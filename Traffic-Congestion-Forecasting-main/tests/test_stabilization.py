"""
Tests for the software-stabilization work: clear errors for bad datasets / missing artifacts, the preflight check
(required vs optional components), the Spark/Java message, the launchers, requirements split and dashboard behaviour.

Nothing here modifies the project: broken scenarios run on temporary COPIES.
Run:   python -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs, needs_all  # noqa: E402
import failure_matrix  # noqa: E402

import numpy as np
import pandas as pd

from src import preflight
from src.app import artifacts, demo
from src.dataProcessing import forecast_data as fd
from src.modelTraining import predictLstm, predictXgboost

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


# ----------------------------------------------------------------------------------------------- dataset validation
def good_frame(days=60, roads=("Alpha Road", "Beta Road")) -> pd.DataFrame:
    """Small SYNTHETIC fixture used only to test the validators (never shown or used as results)."""
    rows = []
    for r in roads:
        for i in range(days):
            rows.append({"Date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "Area Name": "Area",
                         "Road/Intersection Name": r, "Traffic Volume": 1000 + i, "Average Speed": 30.0, "Travel Time Index": 1.2,
                         "Congestion Level": 50.0 + (i % 10), "Road Capacity Utilization": 60.0, "Incident Reports": 1,
                         "Weather Conditions": "Clear"})
    return pd.DataFrame(rows)


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tcf_ds_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, df=None, raw: bytes = None) -> str:
        p = self.tmp / "d.csv"
        if raw is not None:
            p.write_bytes(raw)
        else:
            df.to_csv(p, index=False)
        return str(p)

    def assertRejected(self, path, *words):
        with self.assertRaises(fd.DatasetError) as cm:
            fd.load_raw(path)
        for w in words:
            self.assertIn(w.lower(), str(cm.exception).lower())

    def test_valid_fixture_loads(self):
        self.assertEqual(len(fd.load_raw(self.write(good_frame()))), 120)

    def test_missing_file_is_a_clear_not_found_error(self):
        with self.assertRaises(fd.DatasetNotFoundError) as cm:
            fd.load_raw(str(self.tmp / "nope.csv"))
        self.assertIsInstance(cm.exception, FileNotFoundError)  # old callers that catch FileNotFoundError keep working
        self.assertIn("data/raw", str(cm.exception))

    def test_empty_file(self):
        self.assertRejected(self.write(raw=b""), "empty")

    def test_header_only(self):
        self.assertRejected(self.write(df=good_frame().head(0)), "no rows")

    def test_malformed_binary_file(self):
        self.assertRejected(self.write(raw=bytes(range(256)) * 30))

    def test_missing_column(self):
        self.assertRejected(self.write(good_frame().drop(columns=["Traffic Volume"])), "Traffic Volume", "missing required columns")

    def test_non_numeric_values_name_the_column(self):
        df = good_frame()
        df["Average Speed"] = df["Average Speed"].astype(object)
        df.loc[0, "Average Speed"] = "fast"
        self.assertRejected(self.write(df), "Average Speed", "non-numeric")

    def test_bad_dates(self):
        df = good_frame()
        df.loc[0, "Date"] = "not-a-date"
        self.assertRejected(self.write(df), "not dates")

    def test_missing_values_are_counted_not_imputed(self):
        df = good_frame()
        df.loc[[0, 1, 2], "Traffic Volume"] = np.nan
        self.assertRejected(self.write(df), "Traffic Volume: 3 rows", "nothing is imputed")

    def test_unknown_weather(self):
        df = good_frame()
        df.loc[0, "Weather Conditions"] = "Snow"
        self.assertRejected(self.write(df), "Snow")

    def test_duplicate_rows(self):
        self.assertRejected(self.write(pd.concat([good_frame(), good_frame().head(4)])), "4 duplicate")

    def test_congestion_out_of_range(self):
        df = good_frame()
        df.loc[0, "Congestion Level"] = 250
        self.assertRejected(self.write(df), "0-100")

    def test_too_few_observations_for_any_sample(self):
        with self.assertRaises(fd.DatasetError) as cm:
            fd.load_dataset(self.write(good_frame(days=5)))
        self.assertIn("no usable samples", str(cm.exception).lower())

    def test_dataset_too_small_for_a_meaningful_split(self):
        with self.assertRaises(fd.DatasetError) as cm:
            fd.load_dataset(self.write(good_frame(days=60)))  # ~100 samples < MIN_USABLE_SAMPLES
        self.assertIn("too small", str(cm.exception).lower())

    @needs_data
    def test_the_real_dataset_still_loads(self):
        meta, X, roads, _ = fd.load_dataset()
        self.assertEqual((len(meta), len(roads)), (5754, 16))


# ----------------------------------------------------------------------------------------------- artifacts
class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tcf_art_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_everything_missing_is_listed_with_fix_hints(self):
        gone = artifacts.missing(self.tmp)
        self.assertEqual({n for _, n, _ in gone}, {"model.h5", "scaler.pkl", "lstm_features.json", "xgboost_model.json", "xgboost_features.json"})
        hint = gone[0][2]
        self.assertIn("OVERWRITES", hint)  # retraining is never suggested casually
        self.assertIn("original_backup", hint)  # and the old model folder is explicitly ruled out

    def test_read_json_missing_and_corrupt(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "lstm_features.json.*not found"):
            artifacts.read_json("lstm_features.json", "LSTM feature configuration", self.tmp)
        (self.tmp / "lstm_features.json").write_text("{oops", encoding="utf-8")
        with self.assertRaisesRegex(artifacts.ArtifactError, "not valid JSON"):
            artifacts.read_json("lstm_features.json", "LSTM feature configuration", self.tmp)

    def test_predictors_report_missing_files_instead_of_raw_tracebacks(self):
        hist = pd.DataFrame()
        with self.assertRaisesRegex(artifacts.ArtifactError, "lstm_features.json"):
            predictLstm.predict_next_congestion(hist, model_dir=self.tmp)
        with self.assertRaisesRegex(artifacts.ArtifactError, "xgboost_features.json"):
            predictXgboost.predict_next_congestion(hist, model_dir=self.tmp)
        with self.assertRaisesRegex(artifacts.ArtifactError, "model.h5"):
            predictLstm.preload(self.tmp)
        with self.assertRaisesRegex(artifacts.ArtifactError, "xgboost_model.json"):
            predictXgboost.load_booster(self.tmp)

    @needs_data
    @needs_models
    def test_one_missing_config_only_disables_its_own_model(self):
        for f in ("model.h5", "scaler.pkl", "lstm_features.json", "xgboost_model.json", "xgboost_features.json"):
            shutil.copy(root_dir / "model" / f, self.tmp / f)
        (self.tmp / "lstm_features.json").unlink()
        df = fd.load_raw()
        road = fd.road_names(df)[0]
        hist = df[df[fd.ROAD_COL] == road].tail(12)
        with self.assertRaisesRegex(artifacts.ArtifactError, "lstm_features.json"):
            predictLstm.predict_next_congestion(hist, model_dir=self.tmp)
        same = predictXgboost.predict_next_congestion(hist, model_dir=self.tmp)  # XGBoost is unaffected
        self.assertAlmostEqual(same, predictXgboost.predict_next_congestion(hist), places=9)

    @needs_data
    @needs_models
    def test_model_and_feature_list_from_different_trainings_are_refused(self):
        for f in ("xgboost_model.json", "xgboost_features.json"):
            shutil.copy(root_dir / "model" / f, self.tmp / f)
        cfg = json.loads((self.tmp / "xgboost_features.json").read_text(encoding="utf-8"))
        cfg["xgboost_features"] = cfg["xgboost_features"][::-1]
        (self.tmp / "xgboost_features.json").write_text(json.dumps(cfg), encoding="utf-8")
        df = fd.load_raw()
        hist = df[df[fd.ROAD_COL] == fd.road_names(df)[0]].tail(12)
        with self.assertRaisesRegex(artifacts.ArtifactError, "disagree"):
            predictXgboost.predict_next_congestion(hist, model_dir=self.tmp)


# ----------------------------------------------------------------------------------------------- preflight
@needs_all
class PreflightTests(unittest.TestCase):
    def test_healthy_project_passes_and_groups_are_separated(self):
        checks = preflight.run_checks(fast=True)
        groups = {c.group for c in checks}
        self.assertEqual(groups, {preflight.G_DASH, preflight.G_DEMO, preflight.G_RESULTS, preflight.G_SPARK, preflight.G_TOMTOM})
        self.assertFalse([c for c in checks if c.status == "FAIL"], [c for c in checks if c.status == "FAIL"])
        s = preflight.summarize(checks)
        self.assertTrue(s["dashboard_can_launch"] and s["live_demo_available"])

    def test_optional_groups_never_fail(self):
        with mock.patch.object(preflight, "_java_major", return_value=(None, None)), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TOMTOM_API_KEY", None)
            checks = preflight.run_checks(fast=True)
        java = next(c for c in checks if c.name == "Java")
        self.assertEqual(java.status, "OPT")  # optional component absent: reported, harmless
        self.assertIn("dashboard is unaffected", java.hint)
        s = preflight.summarize(checks)
        self.assertTrue(s["dashboard_can_launch"] and s["live_demo_available"], "missing Java must not block the dashboard")
        self.assertFalse(s["spark_available"])
        self.assertFalse([c for c in checks if c.group in (preflight.G_SPARK, preflight.G_TOMTOM) and c.status in ("FAIL", "WARN")])

    def test_pyspark_missing_is_optional_too(self):
        real = preflight.md.version

        def fake(name):
            if name == "pyspark":
                raise preflight.md.PackageNotFoundError(name)
            return real(name)
        with mock.patch.object(preflight.md, "version", side_effect=fake):
            c = next(c for c in preflight._spark_checks(root_dir) if c.name == "pyspark")
        self.assertEqual(c.status, "OPT")
        self.assertIn("requirements-bigdata.txt", c.hint)

    def test_tomtom_key_is_detected_but_never_printed(self):
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": "SECRET-KEY-123"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = preflight.main(["--fast"])
        self.assertEqual(code, 0)
        self.assertNotIn("SECRET-KEY-123", buf.getvalue())
        self.assertIn("NOT verified", buf.getvalue())
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": ""}):
            self.assertEqual(preflight._tomtom_checks(root_dir)[0].status, "OPT")

    def test_cli_exit_code_and_summary_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = preflight.main(["--fast", "--quiet"])
        self.assertEqual(code, 0)
        self.assertIn("Dashboard can launch ........... YES", buf.getvalue())

    def test_broken_copy_blocks_only_the_demo_not_the_launch(self):
        tmp = Path(tempfile.mkdtemp(prefix="tcf_pf_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        copy = failure_matrix.make_copy(tmp / "c")
        (copy / "model" / "lstm_features.json").unlink()
        r = subprocess.run([sys.executable, "-m", "src.preflight", "--fast", "--json"], cwd=copy, capture_output=True, text=True,
                           env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}, timeout=300)
        data = json.loads(r.stdout)
        self.assertEqual(r.returncode, 1)
        self.assertTrue(data["summary"]["dashboard_can_launch"])
        self.assertFalse(data["summary"]["live_demo_available"])
        bad = [c for c in data["checks"] if c["status"] == "FAIL"]
        self.assertTrue(any("lstm_features.json" in c["name"] and c["hint"] for c in bad))


# ----------------------------------------------------------------------------------------------- Spark / Java
class SparkSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.bigdata import spark_analysis
        cls.sa = spark_analysis

    def test_missing_java_gives_an_actionable_error(self):
        with mock.patch.object(self.sa, "_PYSPARK_IMPORT_ERROR", None), mock.patch.object(preflight, "_java_major", return_value=(None, None)):
            with self.assertRaises(self.sa.SparkSetupError) as cm:
                self.sa.check_prerequisites()
        msg = str(cm.exception)
        self.assertIn("Java was not found", msg)
        self.assertIn("17", msg)
        self.assertIn("dashboard itself does not need Java", msg)

    def test_old_java_is_rejected(self):
        with mock.patch.object(self.sa, "_PYSPARK_IMPORT_ERROR", None), mock.patch.object(preflight, "_java_major", return_value=("java.exe", 11)):
            with self.assertRaisesRegex(self.sa.SparkSetupError, "needs Java 17"):
                self.sa.check_prerequisites()

    def test_missing_pyspark_points_to_the_optional_requirements_file(self):
        with mock.patch.object(self.sa, "_PYSPARK_IMPORT_ERROR", ImportError("no pyspark")):
            with self.assertRaisesRegex(self.sa.SparkSetupError, "requirements-bigdata.txt"):
                self.sa.check_prerequisites()

    @unittest.skipUnless(importlib.util.find_spec("pyspark"), "pyspark not installed")
    def test_cli_without_java_prints_a_message_not_a_traceback(self):
        self.assertEqual(failure_matrix.spark_without_java(), [])


# ----------------------------------------------------------------------------------------------- packaging / launchers
class PackagingTests(unittest.TestCase):
    def test_pyspark_is_optional_and_separate(self):
        def lines(name):  # requirement lines only (comments excluded)
            return [l.split("#")[0].strip().lower() for l in (root_dir / name).read_text(encoding="utf-8").splitlines()
                    if l.split("#")[0].strip()]
        core, big = "\n".join(lines("requirements.txt")), "\n".join(lines("requirements-bigdata.txt"))
        self.assertNotIn("pyspark", core)
        self.assertIn("pyspark==", big)
        self.assertIn("-r requirements.txt", big)
        self.assertIn("requests==", core)  # the dashboard imports the real-time interface at start-up
        self.assertIn("streamlit==", core)

    def test_streamlit_config(self):
        import tomllib
        cfg = tomllib.loads((root_dir / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["client"]["toolbarMode"], "viewer")  # hides the Deploy button
        self.assertIs(cfg["browser"]["gatherUsageStats"], False)

    def test_secrets_are_git_ignored(self):
        self.assertIn(".streamlit/secrets.toml", (root_dir / ".gitignore").read_text(encoding="utf-8"))

    def test_launchers_exist_and_use_the_project_venv_without_activation(self):
        ps1 = (root_dir / "run_dashboard.ps1").read_text(encoding="utf-8")
        bat = (root_dir / "run_dashboard.bat").read_text(encoding="utf-8")
        self.assertIn(r".venv\Scripts\python.exe", ps1)
        self.assertNotIn("Activate.ps1", ps1)
        self.assertIn("'streamlit', 'run', 'app.py'", ps1)
        self.assertIn("run_dashboard.ps1", bat)
        self.assertIn("-ExecutionPolicy Bypass", bat)

    @unittest.skipUnless(POWERSHELL, "PowerShell not available")
    def test_launcher_script_parses(self):
        cmd = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{root_dir / 'run_dashboard.ps1'}',[ref]$null,[ref]$e); if($e.Count){{$e | Out-String; exit 1}}")
        r = subprocess.run([POWERSHELL, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @needs_all
    @unittest.skipUnless(POWERSHELL, "PowerShell not available")
    def test_launcher_check_only_runs_the_full_preflight(self):
        r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root_dir / "run_dashboard.ps1"), "-CheckOnly"],
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-400:])
        self.assertIn("reproduces the saved test prediction", r.stdout)
        self.assertIn("Live prediction demo available . YES", r.stdout)


# ----------------------------------------------------------------------------------------------- display + dashboard
class DisplayTests(unittest.TestCase):
    def test_observation_table_is_rounded_and_complete(self):
        row = pd.Series({"Congestion Level": 100.0, "Average Speed": 38.82219481, "Traffic Volume": 35753.0, "Travel Time Index": 1.5000001,
                         "Road Capacity Utilization": 99.9999, "Incident Reports": 3, "Weather Conditions": "Fog"})
        t = demo.observation_table(row)
        self.assertEqual(len(t), 7)  # no empty trailing row
        self.assertFalse(t.isna().any().any())
        values = dict(zip(t["Field"], t["Day-t value"]))
        self.assertEqual(values["Average Speed"], "38.8")
        self.assertEqual(values["Traffic Volume"], "35,753")
        self.assertEqual(values["Travel Time Index"], "1.50")
        self.assertEqual(values["Incident Reports"], "3")
        import re
        numeric = [v for k, v in values.items() if k != "Weather Conditions"]
        self.assertTrue(all(re.fullmatch(r"[\d,]+(\.\d{1,2})?", v) for v in numeric), numeric)  # at most 2 decimals


@needs_all
class DashboardBehaviourTests(unittest.TestCase):
    def test_healthy_dashboard_has_no_error_boxes_and_the_new_demo_layout(self):
        from streamlit.testing.v1 import AppTest
        os.environ.pop("TOMTOM_API_KEY", None)
        at = AppTest.from_file(str(root_dir / "src" / "ui" / "pages" / "forecasts.py"), default_timeout=300).run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [], "a healthy dashboard must not show any error box (e.g. 'No secrets found')")
        self.assertTrue(any(x.label == "Day-t observation used as input" for x in at.expander))
        at.button(key="demo_predict").click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])
        self.assertTrue(any(x.label == "Input factors and model details" for x in at.expander))
        at.radio(key="demo_source").set_value("What-if scenario (edit day-t conditions)").run()
        self.assertTrue(any(x.label == "Edit the day-t conditions" for x in at.expander))
        self.assertGreaterEqual(len([s for s in at.slider if s.label in demo.EDITABLE_NUMERIC]), 5)

    def test_broken_scenarios_in_fresh_processes(self):
        """A few representative failures end-to-end (the full list is tests/failure_matrix.py)."""
        wanted = ("lstm_features.json missing", "dataset tiny (50 rows)", "metrics.json missing")
        tmp = Path(tempfile.mkdtemp(prefix="tcf_fm_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for i, (name, mutate, banner, err_sub, predict, pre_demo) in enumerate(failure_matrix.SCENARIOS):
            if name not in wanted:
                continue
            copy = failure_matrix.make_copy(tmp / f"c{i}")
            mutate(copy)
            res, pre = failure_matrix.run_app(copy), failure_matrix.run_preflight(copy)
            with self.subTest(name):
                self.assertEqual(failure_matrix.check_scenario(name, banner, err_sub, predict, pre_demo, res, pre), [])
            shutil.rmtree(copy, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

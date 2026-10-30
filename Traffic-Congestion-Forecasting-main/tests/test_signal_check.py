"""
Stage 5A tests for src/analysis/signal_check.py (the signal-gate benchmark).

Most tests use SYNTHETIC data, so they check the method itself: it must report "no usable signal" for data with none and find a
signal of known strength and form. Tests on the real dataset are skipped when the dataset is absent. No model is trained, nothing
is downloaded and no network is used.

Run:   python -m unittest tests.test_signal_check -v
"""
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import signal_check as sc  # noqa: E402
from src.dataProcessing import forecast_data as fd  # noqa: E402

BOOT, PERM = 300, 300  # small, seeded resample counts keep the tests fast


def simulate(seed=0, n_roads=8, n_days=700, obs_prob=0.75, ar=0.0, weekday_amp=0.0, noise_sd=15.0):
    """Synthetic observations (road, date, y) with a KNOWN structure: road level + AR(1) deviation + weekday pattern + noise.
    Rows are observed with probability obs_prob per road and day, so the gaps resemble the real data."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=n_days)
    dow = days.dayofweek.values
    rows = []
    for r in range(n_roads):
        dev = np.zeros(n_days)
        e = rng.normal(0, noise_sd * math.sqrt(1 - ar ** 2), n_days)
        dev[0] = rng.normal(0, noise_sd)
        for t in range(1, n_days):
            dev[t] = ar * dev[t - 1] + e[t]
        y = rng.uniform(40, 90) + dev + weekday_amp * np.sin(2 * np.pi * dow / 7)
        seen = rng.random(n_days) < obs_prob
        rows.append(pd.DataFrame({"road": f"road{r}", "date": days[seen].values, "y": y[seen]}))
    return pd.concat(rows, ignore_index=True)


def run_small(obs, **kw):
    return sc.run(obs, n_boot=BOOT, n_perm=PERM, **kw)


# ======================================================================================================== samples and folds
@needs_data
class RealSampleTests(unittest.TestCase):
    def test_samples_equal_the_projects_own_dataset_and_split(self):
        raw = fd.load_raw()
        meta, X_seq, roads, (tr_end, va_end) = fd.load_dataset()
        s = sc.make_samples(sc.observations_from_raw(raw))
        self.assertEqual(len(s), len(meta))
        pd.testing.assert_series_equal(s["Road"].reset_index(drop=True), meta["Road"].reset_index(drop=True), check_names=False)
        self.assertTrue((s["Date"].values == meta["Date"].values).all())
        self.assertTrue((s["Target Date"].values == meta["Target Date"].values).all())
        self.assertTrue(np.allclose(s["y"].values, meta["y"].values))
        self.assertTrue((s["split"].values == meta["split"].values).all(), "the chronological 70/15/15 split must be the project's own")
        ci = fd.seq_feature_names(roads).index(fd.TARGET_COL)
        self.assertTrue(np.allclose(s["y_t"].values, X_seq[:, -1, ci]))
        self.assertTrue(np.allclose(s["mean_last_7"].values, X_seq[:, :, ci].mean(axis=1), atol=1e-4))

    def test_the_target_is_the_same_road_on_the_next_calendar_day(self):
        s = sc.make_samples(sc.observations_from_raw(fd.load_raw()))
        self.assertTrue(((s["Target Date"] - s["Date"]) == pd.Timedelta(days=1)).all())

    def test_expanding_mean_matches_a_direct_recomputation(self):
        raw = fd.load_raw()
        s = sc.make_samples(sc.observations_from_raw(raw))
        for i in np.linspace(0, len(s) - 1, 25).astype(int):
            row = s.iloc[i]
            hist = raw[(raw[fd.ROAD_COL] == row["Road"]) & (raw[fd.DATE_COL] <= row["Date"])][fd.TARGET_COL]
            self.assertAlmostEqual(row["expanding_mean"], hist.mean(), places=6)


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.s = sc.make_samples(simulate(seed=1))

    def test_folds_are_chronological_disjoint_and_train_strictly_before_test(self):
        folds = sc.rolling_origin_folds(self.s)
        self.assertEqual(len(folds), 5)
        last_end = None
        for f in folds:
            tr, te = self.s[f.train], self.s[f.test]
            self.assertLess(tr["Target Date"].max(), te["Target Date"].min())
            self.assertEqual(te["Target Date"].min(), f.first_test_date)
            self.assertFalse((f.train & f.test).any())
            if last_end is not None:
                self.assertGreater(f.first_test_date, last_end)
            last_end = f.last_test_date
        tests = np.sum([f.test for f in folds], axis=0)
        self.assertTrue((tests <= 1).all(), "no sample may be tested twice")
        self.assertEqual(tests.sum(), (self.s["Target Date"] >= folds[0].first_test_date).sum(), "the folds cover every later date")
        trains = [int(f.train.sum()) for f in folds]
        self.assertEqual(trains, sorted(trains), "the training window only grows")

    def test_warmup_is_training_only(self):
        folds = sc.rolling_origin_folds(self.s)
        dates = np.sort(self.s["Target Date"].unique())
        self.assertGreaterEqual(folds[0].first_test_date, dates[int(len(dates) * sc.WARMUP_FRAC)])

    def test_fewer_folds_when_the_sample_size_does_not_permit_five(self):
        small = sc.make_samples(simulate(seed=2, n_roads=3, n_days=400))
        folds = sc.rolling_origin_folds(small)
        self.assertLess(len(folds), 5)
        self.assertTrue(all(f.test.sum() >= sc.MIN_FOLD_SAMPLES for f in folds))

    def test_too_little_data_is_refused_not_silently_evaluated(self):
        tiny = sc.make_samples(simulate(seed=3, n_roads=2, n_days=120))
        with self.assertRaises(ValueError):
            sc.rolling_origin_folds(tiny)

    @needs_data
    def test_real_data_gives_five_folds_with_enough_samples(self):
        s = sc.make_samples(sc.observations_from_raw(fd.load_raw()))
        folds = sc.rolling_origin_folds(s)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(f.test.sum() >= sc.MIN_FOLD_SAMPLES for f in folds))
        self.assertEqual(sum(int(f.test.sum()) for f in folds), int((s["Target Date"] >= folds[0].first_test_date).sum()))


# ======================================================================================================== no label leakage
class LeakageTests(unittest.TestCase):
    def test_history_baselines_never_see_observations_after_day_t(self):
        obs = simulate(seed=4)
        base = sc.make_samples(obs)
        cutoff = pd.Timestamp("2023-03-01")
        changed = obs.copy()
        later = pd.to_datetime(changed["date"]) > cutoff
        changed.loc[later, "y"] = changed.loc[later, "y"] + 500.0  # a huge change to everything after the cutoff
        alt = sc.make_samples(changed)
        m = base.merge(alt, on=["Road", "Date", "Target Date"], suffixes=("", "_alt"))
        before = m[pd.to_datetime(m["Date"]) <= cutoff]
        self.assertGreater(len(before), 100)
        for col in ("y_t", "mean_last_7", "expanding_mean"):
            self.assertTrue(np.allclose(before[col], before[col + "_alt"]), f"{col} changed when only LATER observations changed")
        after = m[pd.to_datetime(m["Date"]) > cutoff]
        self.assertFalse(np.allclose(after["expanding_mean"], after["expanding_mean_alt"]), "sanity: later rows do change")

    def test_fold_origin_baselines_ignore_every_label_outside_the_training_samples(self):
        s = sc.make_samples(simulate(seed=5))
        f = sc.rolling_origin_folds(s)[2]
        p1 = sc.baseline_predictions(s, f.train, f.test)
        s2 = s.copy()
        s2.loc[~f.train, "y"] = 1e6  # test and future labels are replaced by nonsense
        p2 = sc.baseline_predictions(s2, f.train, f.test)
        pd.testing.assert_frame_equal(p1, p2)

    def test_the_gate_does_not_depend_on_observations_after_the_development_period(self):
        obs = simulate(seed=6)
        s = sc.make_samples(obs)
        val_end = pd.Timestamp(s.loc[s.split == "val", "Target Date"].max())
        changed = obs.copy()
        later = pd.to_datetime(changed["date"]) > val_end
        rng = np.random.default_rng(0)
        changed.loc[later, "y"] = 50 + 40 * np.sin(np.arange(later.sum()) / 5.0) + rng.normal(0, 1, later.sum())
        a, b = run_small(obs), run_small(changed)
        self.assertEqual(json.dumps(a["diagnostics_development_period"], default=sc._py), json.dumps(b["diagnostics_development_period"], default=sc._py))
        self.assertNotEqual(json.dumps(a["diagnostics_all_data_descriptive"], default=sc._py), json.dumps(b["diagnostics_all_data_descriptive"], default=sc._py))


# ======================================================================================================== metrics and bootstrap
class MetricTests(unittest.TestCase):
    def test_basic_metrics(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        m = sc.metrics(y, np.full(4, 25.0))
        self.assertAlmostEqual(m["R2"], 0.0)
        self.assertAlmostEqual(m["MAE"], 10.0)
        self.assertAlmostEqual(m["RMSE"], math.sqrt(125))
        self.assertAlmostEqual(sc.metrics(y, y)["RMSE"], 0.0)

    def test_within_road_correlation(self):
        roads = np.array(["a"] * 4 + ["b"] * 4)
        y = np.array([1, 2, 3, 4, 101, 102, 103, 104.0])
        self.assertAlmostEqual(sc.within_road_corr(roads, y, y + np.where(roads == "a", 0, 500)), 1.0)  # road level offsets do not matter
        self.assertTrue(math.isnan(sc.within_road_corr(roads, y, np.where(roads == "a", 2.0, 102.0))), "a road-constant prediction has no within-road signal")
        self.assertLess(sc.within_road_corr(roads, y, -y), -0.99)

    def test_bootstrap_resamples_whole_dates_and_is_reproducible(self):
        dates = np.repeat(pd.date_range("2024-01-01", periods=20), 3)
        b1, b2 = sc.DateBootstrap(dates, 50, seed=7), sc.DateBootstrap(dates, 50, seed=7)
        self.assertTrue(all((x == y).all() for x, y in zip(b1.idx, b2.idx)))
        for ix in b1.idx:
            counts = np.bincount(np.unique(dates, return_inverse=True)[1][ix], minlength=20)
            self.assertTrue(set(counts % 3) == {0}, "each date contributes all of its rows or none of them")

    def test_identical_predictors_have_a_zero_difference_with_a_zero_width_interval(self):
        rng = np.random.default_rng(0)
        dates = np.repeat(pd.date_range("2024-01-01", periods=40), 4)
        y = rng.normal(50, 10, len(dates))
        p = rng.normal(50, 5, len(dates))
        obs, draws = sc.DateBootstrap(dates, 100, 0).rmse_diff(y, p, p)
        self.assertEqual(obs, 0.0)
        self.assertTrue((draws == 0).all())


# ======================================================================================================== synthetic ground truth
class NoSignalTests(unittest.TestCase):
    def test_data_without_within_road_signal_gets_no_usable_signal(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                rep = run_small(simulate(seed=seed, ar=0.0))
                self.assertEqual(rep["gate"]["verdict"], "NO_USABLE_SIGNAL", rep["gate"]["evidence_of_signal"])
                d = rep["diagnostics_development_period"]
                for a in d["autocorrelation"]:
                    self.assertLess(abs(a["r"]), sc.MIN_AUTOCORR, a)
                for c in d["calendar_effects"]:
                    # adjusted R2 of a many-group effect is noisy (+-0.02), so the decision rule is: large enough AND significant
                    self.assertFalse(c["useful"], c)
                    self.assertGreater(c["permutation_p"], sc.ALPHA / (len(sc.LAGS) + len(sc.CALENDAR_TESTS)), c)
                for name, m in rep["rolling_origin"]["pooled"]["candidates"].items():
                    r = m.get("rmse_minus_reference")
                    if r:
                        self.assertFalse(r["observed"] < 0 and r["excludes_zero_bonferroni"], f"{name} must not beat the expanding mean on noise")

    def test_road_level_differences_alone_are_not_mistaken_for_signal(self):
        # large, persistent differences between roads (like the real pooled lag-1 of ~0.2) but nothing within a road
        obs = simulate(seed=3, ar=0.0)
        pooled = pd.concat([obs, obs.assign(date=pd.to_datetime(obs["date"]))]).sort_values(["road", "date"])
        lag1_pooled = pooled.assign(prev=pooled.groupby("road")["y"].shift(1))[["y", "prev"]].dropna().corr().iloc[0, 1]
        rep = run_small(obs)
        self.assertGreater(lag1_pooled, 0.2, "sanity: the pooled correlation looks like signal")
        self.assertLess(abs(rep["diagnostics_development_period"]["autocorrelation"][0]["r"]), 0.05, "road-standardised lag-1 must not")
        self.assertEqual(rep["gate"]["verdict"], "NO_USABLE_SIGNAL")


class KnownSignalTests(unittest.TestCase):
    def test_strong_temporal_signal_is_found(self):
        rep = run_small(simulate(seed=0, ar=0.8))
        self.assertEqual(rep["gate"]["verdict"], "SIGNAL_FOUND")
        lag1 = rep["diagnostics_development_period"]["autocorrelation"][0]
        self.assertAlmostEqual(lag1["r"], 0.8, delta=0.07)
        self.assertTrue(lag1["useful"])
        # the autocorrelation decays with the lag, as an AR(1) process must
        rs = [a["r"] for a in rep["diagnostics_development_period"]["autocorrelation"]]
        self.assertTrue(rs[0] > rs[1] > rs[2] > rs[3])
        pers = rep["rolling_origin"]["pooled"]["candidates"]["persistence"]
        self.assertLess(pers["rmse_minus_reference"]["observed"], 0)
        self.assertTrue(pers["rmse_minus_reference"]["excludes_zero_bonferroni"], "persistence must beat the mean when ar=0.8")
        self.assertGreater(pers["within_road_corr"], 0.5)

    def test_moderate_signal_is_detected_by_the_diagnostics(self):
        rep = run_small(simulate(seed=1, ar=0.4))
        lag1 = rep["diagnostics_development_period"]["autocorrelation"][0]
        self.assertAlmostEqual(lag1["r"], 0.4, delta=0.07)
        self.assertEqual(rep["gate"]["verdict"], "SIGNAL_FOUND")
        self.assertTrue(any("autocorrelation at lag 1" in e for e in rep["gate"]["evidence_of_signal"]))

    def test_a_weekday_pattern_is_found_as_a_calendar_effect(self):
        rep = run_small(simulate(seed=2, ar=0.0, weekday_amp=12.0))
        self.assertEqual(rep["gate"]["verdict"], "SIGNAL_FOUND")
        wk = next(c for c in rep["diagnostics_development_period"]["calendar_effects"] if c["effect"] == "weekday")
        self.assertTrue(wk["useful"])
        self.assertLess(wk["permutation_p"], 0.01)
        wd = rep["rolling_origin"]["pooled"]["candidates"]["road_weekday_train_mean"]["rmse_minus_reference"]
        self.assertLess(wd["observed"], 0)
        self.assertTrue(wd["excludes_zero_bonferroni"])
        # a weekly pattern necessarily shows up as autocorrelation at a 7-day lag (it is the same structure seen by another test)
        lag7 = next(a for a in rep["diagnostics_development_period"]["autocorrelation"] if a["lag_days"] == 7)
        self.assertGreater(lag7["r"], 0.15)
        self.assertTrue(lag7["useful"])


# ======================================================================================================== the real data (recorded finding)
@needs_data
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = sc.run(n_boot=BOOT, n_perm=PERM)

    def test_recorded_finding_no_usable_signal(self):
        """The current finding on the project's dataset. If the data or method changes this test must be revisited, not weakened."""
        self.assertEqual(self.rep["gate"]["verdict"], "NO_USABLE_SIGNAL", self.rep["gate"]["evidence_of_signal"])
        lag1_all = self.rep["diagnostics_all_data_descriptive"]["autocorrelation"][0]["r"]
        self.assertLess(abs(lag1_all), 0.03, "within-road lag-1 autocorrelation (all data) is about 0.002-0.003")

    def test_gate_uses_development_data_only(self):
        self.assertIn("test labels are not used", self.rep["methodology"]["diagnostics_period"])
        self.assertLess(self.rep["diagnostics_development_period"]["n_observations"], self.rep["diagnostics_all_data_descriptive"]["n_observations"])

    def test_sample_counts_and_split_dates_are_the_projects(self):
        sp = self.rep["samples"]["split_target_dates_and_counts"]
        self.assertEqual(self.rep["samples"]["total"], 5754)
        self.assertEqual([sp[k][2] for k in ("train", "val", "test")], [3997, 901, 856])
        self.assertEqual(sp["test"][:2], ["2024-03-21", "2024-08-09"])
        self.assertEqual(self.rep["rolling_origin"]["n_folds"], 5)

    @needs_outputs
    def test_official_split_baselines_and_saved_models_match_the_saved_metrics(self):
        saved = json.loads((root_dir / "model" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        c = self.rep["official_test"]["candidates"]
        for mine, theirs in (("road_train_mean", "Road-mean baseline"), ("persistence", "Persistence baseline"),
                             ("saved_LSTM", "LSTM"), ("saved_XGBoost", "XGBoost")):
            self.assertAlmostEqual(c[mine]["RMSE"], saved[theirs]["RMSE"], places=3, msg=mine)
            self.assertAlmostEqual(c[mine]["MAE"], saved[theirs]["MAE"], places=3, msg=mine)
            self.assertAlmostEqual(c[mine]["R2"], saved[theirs]["R2"], places=3, msg=mine)

    def test_no_baseline_is_distinguishably_better_than_the_expanding_mean(self):
        for name, m in self.rep["official_test"]["candidates"].items():
            r = m.get("rmse_minus_reference")
            if r:
                self.assertFalse(r["observed"] < 0 and r["excludes_zero_bonferroni"], name)


# ======================================================================================================== isolation and scope
class IsolationTests(unittest.TestCase):
    def test_output_directories_that_hold_protected_files_are_refused(self):
        for bad in ("model", "data", "data/raw", "doucments", "reports", "src", "src/analysis", "tests", "notebooks", "audit_backup", "."):
            with self.subTest(bad), self.assertRaises(ValueError):
                sc.safe_output_dir(root_dir / bad)

    def test_other_directories_are_allowed(self):
        self.assertEqual(sc.safe_output_dir(sc.DEFAULT_OUT), sc.DEFAULT_OUT.resolve())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sc.safe_output_dir(tmp), Path(tmp).resolve())

    def test_outputs_are_exactly_two_files_in_the_chosen_directory(self):
        rep = run_small(simulate(seed=0))
        with tempfile.TemporaryDirectory() as tmp:
            paths = sc.write_outputs(rep, Path(tmp) / "out")
            self.assertEqual(sorted(p.name for p in paths), ["signal_check.json", "signal_check.md"])
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "out").iterdir()), ["signal_check.json", "signal_check.md"])
            loaded = json.loads((Path(tmp) / "out" / "signal_check.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["gate"]["verdict"], rep["gate"]["verdict"])
            md = (Path(tmp) / "out" / "signal_check.md").read_text(encoding="utf-8")
            for needle in ("Verdict:", "Rolling-origin folds", "Limitations", "Reference = expanding per-road mean"):
                self.assertIn(needle, md)

    def test_writing_into_a_protected_directory_raises_before_anything_is_written(self):
        rep = run_small(simulate(seed=0))
        before = sorted(p.name for p in (root_dir / "model").iterdir())
        with self.assertRaises(ValueError):
            sc.write_outputs(rep, root_dir / "model")
        self.assertEqual(sorted(p.name for p in (root_dir / "model").iterdir()), before)

    def test_the_output_is_reproducible_for_the_same_seed(self):
        a, b = run_small(simulate(seed=0)), run_small(simulate(seed=0))
        self.assertEqual(json.dumps(a, default=sc._py, sort_keys=True), json.dumps(b, default=sc._py, sort_keys=True))

    @needs_models
    @needs_data
    def test_a_run_on_the_real_data_leaves_every_protected_model_file_byte_identical(self):
        files = [root_dir / "model" / n for n in ("model.h5", "scaler.pkl", "xgboost_model.json", "metrics.json", "evaluation_report.json",
                                                   "lstm_predictions.csv", "xgboost_predictions.csv", "baseline_predictions.csv")]
        before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
        sc.run(n_boot=50, n_perm=50)
        after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
        self.assertEqual(before, after)

    def test_the_module_trains_nothing_and_uses_no_network_or_live_code(self):
        src = (root_dir / "src" / "analysis" / "signal_check.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if l.lstrip().startswith(("import ", "from ")))
        for banned in ("tensorflow", "keras", "xgboost", "sklearn", "joblib", "torch", "requests", "urllib", "socket", "http", "src.live", "src.ui", "src.app"):
            self.assertNotIn(banned, code, f"signal_check must not import {banned}")
        for banned in (".fit(", ".train(", "to_csv(", "pickle", "joblib", ".save(", "save_model"):
            self.assertNotIn(banned, src, banned)
        self.assertEqual(src.count(".write_text("), 2, "the only writes are the two output files")


# ======================================================================================================== wording corrections
class WordingCorrectionTests(unittest.TestCase):
    """The pooled lag-1 correlation (about 0.21) must not be presented as within-road predictability."""

    @classmethod
    def setUpClass(cls):
        cls.readme = (root_dir / "README.md").read_text(encoding="utf-8")
        from streamlit.testing.v1 import AppTest
        cls.at = AppTest.from_file(str(root_dir / "src" / "ui" / "pages" / "performance.py"), default_timeout=300).run()
        cls.page = " ".join(e.value for g in (cls.at.markdown, cls.at.caption, cls.at.warning, cls.at.info) for e in g)

    def test_readme_explains_the_pooled_correlation_and_the_within_road_value(self):
        flat = " ".join(self.readme.split())
        self.assertNotIn("day-to-day autocorrelation is only about 0.2", flat)
        for needle in ("pooled across roads", "persistent differences between roads", "approximately 0", "within-road lag-1 correlation",
                       "was not regenerated"):
            self.assertIn(needle, flat)

    def test_readme_keeps_the_experimental_label_and_the_baseline_finding(self):
        flat = " ".join(self.readme.split())
        self.assertIn("Status: experimental", flat)
        self.assertIn("neither model beat the simple per-road historical-mean baseline", flat)
        self.assertIn("do **not** beat the road-mean baseline", flat)

    def test_the_figure_quoted_in_the_readme_is_the_one_in_the_saved_benchmark_output(self):
        out = root_dir / "experiments" / "stage5a" / "signal_check.json"
        if not out.is_file():
            self.skipTest("benchmark output not generated yet")
        rep = json.loads(out.read_text(encoding="utf-8"))
        r = rep["diagnostics_all_data_descriptive"]["autocorrelation"][0]
        self.assertIn(f"{r['r']:.3f} on all data", self.readme)
        self.assertIn(f"{r['ci95'][0]:.2f} to +{r['ci95'][1]:.2f}", " ".join(self.readme.split()))

    def test_the_model_insights_page_corrects_the_quoted_figure_without_touching_the_saved_report(self):
        self.assertEqual([e.value for e in self.at.exception], [])
        flat = " ".join(self.page.split())
        for needle in ("pooled day-to-day (lag-1) correlation of about 0.21", "largely driven by persistent differences between roads",
                       "not evidence that next-day changes can be predicted", "The report itself is shown unchanged",
                       "no predictable within-road signal in the Stage 5A benchmark"):
            self.assertIn(needle, flat)
        report = " ".join((root_dir / "model" / "evaluation_report.md").read_text(encoding="utf-8").split())
        self.assertIn("only weakly predictable (lag-1 correlation 0.21)", flat, "the saved report is shown unchanged inside the page")
        own = flat.replace(report, "")  # everything the page says itself, outside the saved report
        self.assertNotIn("only weakly predictable", own)
        self.assertNotIn("lag-1 correlation 0.21", own)

    def test_the_page_still_says_experimental_and_that_neither_model_beats_the_baseline(self):
        flat = " ".join(self.page.split())
        self.assertIn("Experimental models. Honest finding: neither learned model beats the simple per-road historical-mean baseline", flat)
        self.assertIn("the models do not beat the road-mean baseline", flat)
        self.assertIn("Neither model beats the simple", flat)

    def test_saved_evaluation_files_were_not_modified(self):
        rep = (root_dir / "model" / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertIn("lag-1 correlation 0.21", rep, "the saved report is deliberately left as it was")


if __name__ == "__main__":
    unittest.main()

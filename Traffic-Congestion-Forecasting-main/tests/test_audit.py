"""
Audit regression tests. Several tests rebuild the samples, split, baselines and metrics from the raw CSV WITHOUT using the
project's pipeline code, so a bug shared by the pipeline and its own tests cannot hide.
Run:   python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs, needs_all  # noqa: E402

import numpy as np
import pandas as pd

MODEL_DIR = root_dir / "model"
R, C = "Road/Intersection Name", "Congestion Level"


def independent_samples():
    """(road, t, t+1) pairs rebuilt from scratch with plain pandas."""
    raw = pd.read_csv(root_dir / "data" / "raw" / "Banglore_traffic_Dataset.csv", parse_dates=["Date"])
    raw = raw.sort_values([R, "Date"]).reset_index(drop=True)
    raw["i"] = raw.groupby(R).cumcount()
    nxt = raw[[R, "Date", C]].copy()
    nxt["Date"] -= pd.Timedelta(days=1)
    pairs = raw.merge(nxt.rename(columns={C: "y"}), on=[R, "Date"])
    pairs = pairs[pairs["i"] >= 7].copy()  # 7-observation window + one earlier observation for days_since_prev
    pairs["target_date"] = pairs["Date"] + pd.Timedelta(days=1)
    d = np.sort(pairs["target_date"].unique())
    n = len(d)
    tr_end, va_end = d[int(n * 0.70) - 1], d[int(n * 0.85) - 1]
    pairs["split"] = np.where(pairs.target_date <= tr_end, "train",
                              np.where(pairs.target_date <= va_end, "val", "test"))
    return raw, pairs


@needs_all
class IndependentReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.pairs = independent_samples()
        cls.x = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
        cls.l = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")
        cls.b = pd.read_csv(MODEL_DIR / "baseline_predictions.csv")
        cls.test = cls.pairs[cls.pairs.split == "test"].sort_values(["target_date", R]).reset_index(drop=True)
        cls.bs = cls.b.sort_values(["Date", R]).reset_index(drop=True)

    def test_pipeline_sample_counts_and_split_match_independent_rebuild(self):
        from src.dataProcessing import forecast_data as fd
        meta, _, _, (train_end, val_end) = fd.load_dataset()
        self.assertEqual(len(meta), len(self.pairs))
        self.assertEqual(meta["split"].value_counts().to_dict(), self.pairs["split"].value_counts().to_dict())
        self.assertEqual(self.pairs.groupby("split")["target_date"].max().loc["train"], pd.Timestamp(train_end))

    def test_all_prediction_files_cover_exactly_the_independent_test_rows(self):
        keys = self.test["target_date"].dt.strftime("%Y-%m-%d").tolist()
        for name, df_ in (("xgboost", self.x), ("lstm", self.l), ("baseline", self.b)):
            d = df_.sort_values(["Date", R]).reset_index(drop=True)
            self.assertEqual(d["Date"].tolist(), keys, name)
            self.assertEqual(d[R].tolist(), self.test[R].tolist(), name)
            np.testing.assert_allclose(d["Actual Congestion"], self.test["y"], err_msg=name)

    def test_road_mean_baseline_uses_training_targets_only(self):
        train_mean = self.pairs[self.pairs.split == "train"].groupby(R)["y"].mean()
        all_mean = self.pairs.groupby(R)["y"].mean()
        self.assertFalse(np.allclose(train_mean.values, all_mean.values),
                         "guard: train-only and all-data means must differ, otherwise this test cannot detect a leak")
        np.testing.assert_allclose(self.test[R].map(train_mean).values, self.bs["Road Mean"].values)
        self.assertFalse(np.allclose(self.test[R].map(all_mean).values, self.bs["Road Mean"].values))

    def test_persistence_baseline_is_congestion_on_day_t(self):
        np.testing.assert_allclose(self.test[C].values, self.bs["Persistence"].values)

    def test_reported_metrics_match_independent_calculation(self):
        mj = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        y = self.x["Actual Congestion"].values
        preds = {"LSTM": self.l["Predicted Congestion"].values, "XGBoost": self.x["Predicted Congestion"].values,
                 "Persistence baseline": self.b["Persistence"].values, "Road-mean baseline": self.b["Road Mean"].values}
        for k, p in preds.items():
            self.assertAlmostEqual(float(np.mean(np.abs(y - p))), mj[k]["MAE"], places=6, msg=k)
            self.assertAlmostEqual(float(np.sqrt(np.mean((y - p) ** 2))), mj[k]["RMSE"], places=6, msg=k)
            self.assertAlmostEqual(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), mj[k]["R2"],
                                   places=6, msg=k)

    def test_no_prediction_row_uses_a_training_period_target(self):
        first_test_target = self.test["target_date"].min()
        last_train_target = self.pairs[self.pairs.split != "test"]["target_date"].max()
        self.assertGreater(first_test_target, last_train_target)


@needs_all
class EvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = json.loads((MODEL_DIR / "evaluation_report.json").read_text(encoding="utf-8"))
        cls.md = (MODEL_DIR / "evaluation_report.md").read_text(encoding="utf-8")

    def test_report_values_are_computed_not_hardcoded(self):
        raw = pd.read_csv(root_dir / "data" / "raw" / "Banglore_traffic_Dataset.csv")
        f = self.rep["dataset_facts"]
        self.assertEqual(f["rows"], len(raw))
        self.assertEqual(self.rep["dataset"]["rows"], len(raw))
        self.assertAlmostEqual(f["pct_congestion_at_100"], (raw[C] >= 100).mean() * 100, places=6)
        self.assertEqual(self.rep["split"]["n_test_samples"], 856)
        self.assertIs(self.rep["identical_test_rows_for_all_models"], True)

    def test_provenance_and_license_are_not_overstated(self):
        self.assertIn("NOT VERIFIED", self.rep["dataset"]["license"])
        self.assertIn("NOT verified", self.rep["dataset"]["provenance_note"])
        self.assertNotIn("per Kaggle", json.dumps(self.rep))
        for path in ("README.md", "app.py"):
            text = (root_dir / path).read_text(encoding="utf-8")
            self.assertNotIn("open data, per", text, path)
            self.assertNotIn("described there as derived", text, path)

    def test_report_states_uncertainty_and_no_unsupported_ranking(self):
        d = self.rep["uncertainty"]["rmse_difference"]
        for k, v in d.items():
            lo, hi = v["ci95"]
            self.assertLessEqual(lo, v["observed"] + 1e-9)
            self.assertGreaterEqual(hi, v["observed"] - 1e-9)
            self.assertEqual(v["ci_excludes_zero"], not (lo <= 0 <= hi))
        if not any(v["ci_excludes_zero"] for v in d.values()):
            self.assertIn("no ranking among them is supported", self.md)

    def test_old_leaky_metrics_are_not_in_current_outputs(self):
        for text in (self.md, (MODEL_DIR / "saved_metrics.txt").read_text(encoding="utf-8")):
            self.assertNotIn("0.9995", text)
            self.assertNotIn("0.999496", text)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_detects_real_differences(self):
        # Synthetic arrays used ONLY to unit-test the statistics code; they are never shown as results.
        from src.modelTraining.comparePerformancePlots import bootstrap_rmse
        rng = np.random.default_rng(1)
        actual = rng.uniform(0, 100, 400)
        dates = np.repeat(np.arange(100), 4)
        preds = {"LSTM": actual + rng.normal(0, 1, 400), "XGBoost": actual + rng.normal(0, 30, 400),
                 "Road-mean baseline": actual + rng.normal(0, 30, 400), "Persistence baseline": actual * 0}
        a, b = bootstrap_rmse(actual, dates, preds), bootstrap_rmse(actual, dates, preds)
        self.assertEqual(a, b)  # seeded -> identical
        self.assertTrue(a["rmse_difference"]["LSTM - XGBoost"]["ci_excludes_zero"])  # 1 vs 30 points: clearly different
        same = {**preds, "LSTM": preds["XGBoost"] + 0.0}
        self.assertFalse(bootstrap_rmse(actual, dates, same)["rmse_difference"]["LSTM - XGBoost"]["ci_excludes_zero"])


class NotebookAndAppTests(unittest.TestCase):
    @unittest.skipUnless((root_dir / "notebooks" / "test.ipynb").is_file(), "notebooks/test.ipynb not found")
    def test_notebook_no_longer_calls_removed_api(self):
        nb = json.loads((root_dir / "notebooks" / "test.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertNotIn("predict_next_congestion(last_6)", code)
        self.assertIn("predictLstm.predict_next_congestion(history)", code)

    @needs_all
    def test_dashboard_declares_no_winner_when_ci_includes_zero(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(root_dir / "src" / "ui" / "pages" / "performance.py"), default_timeout=300).run()
        self.assertEqual([e.value for e in at.exception], [])
        info = " ".join(i.value for i in at.info)
        self.assertIn("No winner between LSTM and XGBoost", info)
        self.assertFalse(any("Lowest test RMSE of the two models" in s.value for s in at.success))
        self.assertTrue(any("Neither model beats the simple" in w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()

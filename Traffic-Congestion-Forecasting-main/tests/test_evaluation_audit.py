"""
Independent re-verification of the SAVED evaluation (model/*.csv, metrics.json, scaler.pkl) against the raw dataset.

It trains nothing, writes nothing and never touches the model files. It recomputes the numbers by a route that does not go through the
evaluation scripts, so a mistake in how the models were compared (different rows, a baseline that peeks at the test period, a scaler fitted on
more than the training data, a window that contains its own target) would show up here. It checks the saved results; it is not used to choose anything.
"""
import json
import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database)

from _guards import needs_all  # noqa: E402

MODEL_DIR = root_dir / "model"
METRIC_NAMES = ("MAE", "RMSE", "R2")


def _metrics(y, p):
    import numpy as np
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


@needs_all
class SavedEvaluationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from src.dataProcessing import forecast_data as fd
        cls.pd, cls.fd = pd, fd
        cls.xgb = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
        cls.lstm = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")
        cls.base = pd.read_csv(MODEL_DIR / "baseline_predictions.csv")
        cls.saved = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        cls.report = json.loads((MODEL_DIR / "evaluation_report.json").read_text(encoding="utf-8"))
        cls.raw = fd.load_raw()
        cls.meta, cls.X_seq, cls.roads, (cls.train_end, cls.val_end) = fd.load_dataset()

    def test_all_models_are_scored_on_the_identical_test_rows(self):
        key = ["Date", "Road/Intersection Name"]
        import numpy as np
        for other in (self.lstm, self.base):
            self.assertEqual(len(other), len(self.xgb))
            self.assertTrue((other[key].values == self.xgb[key].values).all())
            self.assertTrue(np.allclose(other["Actual Congestion"], self.xgb["Actual Congestion"]))
        self.assertFalse(self.xgb.duplicated(key).any())
        self.assertIs(self.report["identical_test_rows_for_all_models"], True)

    def test_the_saved_metrics_equal_the_metrics_recomputed_from_the_saved_predictions(self):
        y = self.xgb["Actual Congestion"].values
        recomputed = {"XGBoost": _metrics(y, self.xgb["Predicted Congestion"]), "LSTM": _metrics(y, self.lstm["Predicted Congestion"]),
                      "Persistence baseline": _metrics(y, self.base["Persistence"]), "Road-mean baseline": _metrics(y, self.base["Road Mean"])}
        for model, values in recomputed.items():
            for name in METRIC_NAMES:
                with self.subTest(model=model, metric=name):
                    self.assertAlmostEqual(values[name], self.saved[model][name], places=6)

    def test_actual_values_and_the_persistence_baseline_match_the_raw_dataset(self):
        import numpy as np
        idx = self.raw.set_index([self.fd.ROAD_COL, self.fd.DATE_COL])[self.fd.TARGET_COL]
        dates = self.pd.to_datetime(self.xgb["Date"])
        roads = self.xgb["Road/Intersection Name"]
        actual = [idx.get((r, d)) for r, d in zip(roads, dates)]
        previous_day = [idx.get((r, d - self.pd.Timedelta(days=1))) for r, d in zip(roads, dates)]
        self.assertTrue(np.allclose(np.array(actual, float), self.xgb["Actual Congestion"].values))
        self.assertTrue(all(v is not None for v in previous_day), "every test row has an observation on the previous day")
        # persistence = the value known on day t, the day before the target
        self.assertTrue(np.allclose(np.array(previous_day, float), self.base["Persistence"].values))

    def test_the_split_is_chronological_and_the_road_mean_uses_training_targets_only(self):
        import numpy as np
        first_test = self.pd.to_datetime(self.xgb["Date"]).min()
        self.assertLess(self.pd.Timestamp(self.train_end), self.pd.Timestamp(self.val_end))
        self.assertLess(self.pd.Timestamp(self.val_end), first_test)
        train_mean = self.meta[self.meta["split"] == "train"].groupby("Road")["y"].mean()
        self.assertTrue(np.allclose(self.base["Road Mean"].values, self.xgb["Road/Intersection Name"].map(train_mean).values))
        all_mean = self.meta.groupby("Road")["y"].mean()
        self.assertFalse(np.allclose(train_mean.values, all_mean.values), "control: a mean over all data would differ, so a mix-up would be visible")

    def test_every_input_window_ends_the_day_before_its_target(self):
        self.assertTrue(((self.meta["Target Date"] - self.meta["Date"]).dt.days == 1).all())

    def test_the_lstm_scaler_was_fitted_on_training_windows_only(self):
        import joblib
        import numpy as np
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        train = (self.meta["split"] == "train").values
        n_features = self.X_seq.shape[2]
        flat_train = self.X_seq[train].reshape(-1, n_features)
        self.assertTrue(np.allclose(scaler.data_min_, flat_train.min(axis=0)))
        self.assertTrue(np.allclose(scaler.data_max_, flat_train.max(axis=0)))
        flat_all = self.X_seq.reshape(-1, n_features)
        self.assertFalse(np.allclose(scaler.data_min_, flat_all.min(axis=0)) and np.allclose(scaler.data_max_, flat_all.max(axis=0)),
                         "control: it must differ from the all-data range, otherwise this test could not tell the two apart")
        self.assertEqual(list(scaler.feature_names_in_), json.loads((MODEL_DIR / "lstm_features.json").read_text(encoding="utf-8"))["features"])

    def test_the_conclusion_follows_from_the_measured_numbers(self):
        """No learned model is called better than the road-mean baseline unless the saved bootstrap interval supports it."""
        base = self.saved["Road-mean baseline"]["RMSE"]
        for name in ("LSTM", "XGBoost"):
            diff = self.report["uncertainty"]["rmse_difference"][f"{name} - Road-mean baseline"]
            beats = self.saved[name]["RMSE"] < base and diff["ci_excludes_zero"]
            self.assertFalse(beats, f"{name} would be a statistically supported improvement over the baseline; the README and the pages must then change")


if __name__ == "__main__":
    unittest.main()

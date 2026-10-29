"""
Tests of the trained artifacts. Run after both training scripts:   python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs, needs_all  # noqa: E402

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf

from src.dataProcessing import forecast_data as fd
from src.modelTraining import predictLstm, predictXgboost

MODEL_DIR = root_dir / "model"


@needs_all
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fd.load_raw()
        cls.roads = fd.road_names(cls.df)
        cls.meta, cls.X = fd.build_samples(cls.df)
        cls.meta["split"], _, _ = fd.chronological_split(cls.meta)
        cls.lstm_cfg = json.loads((MODEL_DIR / "lstm_features.json").read_text(encoding="utf-8"))
        cls.xgb_cfg = json.loads((MODEL_DIR / "xgboost_features.json").read_text(encoding="utf-8"))
        cls.xgb_pred = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
        cls.lstm_pred = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")

    def test_lstm_input_shape_matches_pipeline(self):
        model = tf.keras.models.load_model(MODEL_DIR / "model.h5", compile=False)
        self.assertEqual(model.input_shape, (None, fd.SEQ_LEN, len(fd.seq_feature_names(self.roads))))
        self.assertEqual(model.output_shape, (None, 1))

    def test_scaler_feature_order_and_train_only_fit(self):
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        names = fd.seq_feature_names(self.roads)
        self.assertEqual(list(scaler.feature_names_in_), names)
        self.assertEqual(self.lstm_cfg["features"], names)
        train = self.X[(self.meta["split"] == "train").values].reshape(-1, len(names))
        allw = self.X.reshape(-1, len(names))
        np.testing.assert_allclose(scaler.data_max_, train.max(axis=0), rtol=1e-5)  # fitted on train only
        np.testing.assert_allclose(scaler.data_min_, train.min(axis=0), rtol=1e-5)
        # sanity: the full-data range would differ somewhere, so this test can actually detect test-set fitting
        self.assertTrue((allw.max(axis=0) != train.max(axis=0)).any() or (allw.min(axis=0) != train.min(axis=0)).any())

    def test_xgboost_features_match_pipeline(self):
        booster = xgb.Booster()
        booster.load_model(str(MODEL_DIR / "xgboost_model.json"))
        tab = fd.tabular_from_sequences(self.X[:5], self.meta["Target Date"][:5], self.roads)
        self.assertEqual(booster.feature_names, list(tab.columns))
        self.assertEqual(booster.feature_names, self.xgb_cfg["xgboost_features"])

    def test_prediction_files_are_aligned_and_on_test_rows_only(self):
        te = self.meta[self.meta["split"] == "test"]
        self.assertEqual(len(self.xgb_pred), len(te))
        self.assertEqual(len(self.lstm_pred), len(te))
        keys = ["Date", "Road/Intersection Name"]
        self.assertTrue((self.xgb_pred[keys].values == self.lstm_pred[keys].values).all())
        self.assertTrue(np.allclose(self.xgb_pred["Actual Congestion"], self.lstm_pred["Actual Congestion"]))
        self.assertEqual(self.xgb_pred["Date"].tolist(), te["Target Date"].dt.strftime("%Y-%m-%d").tolist())
        train_max = self.meta.loc[self.meta["split"] != "test", "Target Date"].max()
        self.assertGreater(pd.Timestamp(self.xgb_pred["Date"].min()), train_max)
        for d in (self.xgb_pred, self.lstm_pred):
            self.assertFalse(d["Predicted Congestion"].isna().any())
            self.assertTrue(d["Predicted Congestion"].between(0, 100).all())

    def _history_for(self, row):
        d = self.df[(self.df[fd.ROAD_COL] == row["Road/Intersection Name"])]
        return d[d[fd.DATE_COL] < pd.Timestamp(row["Date"])].tail(12)  # last observations before the target day

    def test_single_prediction_matches_batch_prediction(self):
        """The inference scripts must reproduce the predictions saved by the training scripts."""
        for i in (0, len(self.xgb_pred) // 2, len(self.xgb_pred) - 1):
            row = self.xgb_pred.iloc[i]
            hist = self._history_for(row)
            self.assertEqual(hist[fd.DATE_COL].iloc[-1], pd.Timestamp(row["Date"]) - pd.Timedelta(days=1))
            self.assertAlmostEqual(predictXgboost.predict_next_congestion(hist), row["Predicted Congestion"], places=3)
            self.assertAlmostEqual(predictLstm.predict_next_congestion(hist),
                                   self.lstm_pred.iloc[i]["Predicted Congestion"], places=2)


if __name__ == "__main__":
    unittest.main()

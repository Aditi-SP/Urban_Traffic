"""
Data-pipeline integrity tests. Run from the project root:   python -m unittest discover -s tests -v
(They need data/raw/Banglore_traffic_Dataset.csv.)
"""
import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs, needs_all  # noqa: E402

import numpy as np
import pandas as pd

from src.dataProcessing import forecast_data as fd


@needs_data
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fd.load_raw()
        cls.roads = fd.road_names(cls.df)
        cls.meta, cls.X, = fd.build_samples(cls.df)
        cls.meta["split"], cls.train_end, cls.val_end = fd.chronological_split(cls.meta)
        cls.names = fd.seq_feature_names(cls.roads)
        cls.lookup = cls.df.set_index([fd.ROAD_COL, fd.DATE_COL])

    def test_shapes(self):
        self.assertEqual(self.X.shape[1], fd.SEQ_LEN)
        self.assertEqual(self.X.shape[2], len(self.names))
        self.assertEqual(len(self.meta), len(self.X))
        self.assertFalse(np.isnan(self.X).any())

    def test_target_is_next_day_of_same_road(self):
        rng = np.random.default_rng(0)
        for k in rng.choice(len(self.meta), 300, replace=False):
            m = self.meta.iloc[k]
            self.assertEqual(m["Target Date"] - m["Date"], pd.Timedelta(days=1))
            actual = self.lookup.loc[(m["Road"], m["Target Date"]), fd.TARGET_COL]
            self.assertAlmostEqual(actual, m["y"], places=4)

    def test_windows_stay_inside_one_road_and_end_at_day_t(self):
        rng = np.random.default_rng(1)
        ci = self.names.index(fd.TARGET_COL)
        for k in rng.choice(len(self.meta), 300, replace=False):
            m = self.meta.iloc[k]
            road_rows = self.df[self.df[fd.ROAD_COL] == m["Road"]]
            road_rows = road_rows[road_rows[fd.DATE_COL] <= m["Date"]].tail(fd.SEQ_LEN)
            self.assertEqual(len(road_rows), fd.SEQ_LEN)
            self.assertEqual(road_rows[fd.DATE_COL].iloc[-1], m["Date"])  # window ends at t
            np.testing.assert_allclose(self.X[k, :, ci], road_rows[fd.TARGET_COL].values, rtol=1e-5)
            onehot = self.X[k, :, [self.names.index(f"Road_{r}") for r in self.roads]].T
            self.assertTrue((onehot.argmax(axis=1) == self.roads.index(m["Road"])).all())
            self.assertTrue((onehot.sum(axis=1) == 1).all())

    def test_chronological_split_has_no_overlap(self):
        s = self.meta.groupby("split")["Target Date"].agg(["min", "max"])
        self.assertLess(s.loc["train", "max"], s.loc["val", "min"])
        self.assertLess(s.loc["val", "max"], s.loc["test", "min"])
        self.assertTrue({"train", "val", "test"} <= set(s.index))

    def test_inputs_never_contain_future_information(self):
        """Corrupt every observation on/after the target day of a sample: its features must not change."""
        rng = np.random.default_rng(2)
        for k in rng.choice(len(self.meta), 5, replace=False):
            m = self.meta.iloc[k]
            bad = self.df.copy()
            future = (bad[fd.ROAD_COL] == m["Road"]) & (bad[fd.DATE_COL] >= m["Target Date"])
            for c in fd.NUMERIC_COLS:
                bad.loc[future, c] = bad.loc[future, c] * 0 + 12345.0
            bad.loc[future, "Weather Conditions"] = "Rain"
            meta2, X2 = fd.build_samples(bad)
            j = meta2.index[(meta2["Road"] == m["Road"]) & (meta2["Date"] == m["Date"])]
            self.assertEqual(len(j), 1)
            np.testing.assert_array_equal(X2[j[0]], self.X[k])  # LSTM window identical
            tab_a = fd.tabular_from_sequences(self.X[[k]], [m["Target Date"]], self.roads)
            tab_b = fd.tabular_from_sequences(X2[[j[0]]], [m["Target Date"]], self.roads)
            pd.testing.assert_frame_equal(tab_a, tab_b)  # XGBoost row identical

    def test_tabular_features_do_not_contain_the_target(self):
        tab = fd.tabular_from_sequences(self.X, self.meta["Target Date"], self.roads)
        y = self.meta["y"].values
        # A leaking feature (like the old rolling mean / ratio) would correlate far above the day-t baseline.
        corr = {c: abs(np.corrcoef(tab[c].values, y)[0, 1]) for c in tab.columns if tab[c].std() > 0}
        self.assertLess(max(corr.values()), 0.95, sorted(corr.items(), key=lambda kv: -kv[1])[:3])
        self.assertEqual(len(tab), len(self.meta))


if __name__ == "__main__":
    unittest.main()

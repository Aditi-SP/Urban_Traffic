"""
Tests for the live demo logic, the real-time interface (unconfigured state + parsing with a mocked HTTP response),
the PySpark outputs and the Streamlit app. Run:   python -m unittest discover -s tests -v

NOTE: nothing here talks to the real traffic provider. The real-time integration is unverified against the live service.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs, needs_all  # noqa: E402

import numpy as np
import pandas as pd

from src.app import demo, realtime
from src.dataProcessing import forecast_data as fd

MODEL_DIR = root_dir / "model"


@needs_all
class DemoLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = demo.DemoContext()
        cls.xgb_pred = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
        cls.lstm_pred = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")
        cls.base_pred = pd.read_csv(MODEL_DIR / "baseline_predictions.csv")

    def test_replay_reproduces_saved_test_predictions_for_every_model(self):
        """Replaying real test windows through the demo must give the same numbers as the evaluation files."""
        for i in (0, 100, 400, len(self.xgb_pred) - 1):
            row = self.xgb_pred.iloc[i]
            t = pd.Timestamp(row["Date"]) - pd.Timedelta(days=1)
            hist = self.ctx.history(row["Road/Intersection Name"], t)
            res = demo.predict_all(hist, self.ctx)
            self.assertEqual(res["target_date"], pd.Timestamp(row["Date"]))
            self.assertAlmostEqual(res["XGBoost"], row["Predicted Congestion"], places=3)
            self.assertAlmostEqual(res["LSTM"], self.lstm_pred.iloc[i]["Predicted Congestion"], places=2)
            self.assertAlmostEqual(res["Persistence baseline"], self.base_pred.iloc[i]["Persistence"], places=3)
            self.assertAlmostEqual(res["Road-mean baseline"], self.base_pred.iloc[i]["Road Mean"], places=3)
            self.assertAlmostEqual(self.ctx.actual_next_day(res["road"], t), row["Actual Congestion"], places=3)

    def test_replay_dates_are_valid_and_test_only_by_default(self):
        road = self.ctx.roads[0]
        dates = self.ctx.replay_dates(road, test_only=True)
        self.assertTrue(len(dates) > 5)
        self.assertTrue(all(pd.Timestamp(d) + pd.Timedelta(days=1) > self.ctx.val_end for d in dates))
        for d in dates[:3]:
            self.assertGreaterEqual(len(self.ctx.history(road, d)), fd.SEQ_LEN + 1)

    def test_invalid_inputs_raise_actionable_errors(self):
        road = self.ctx.roads[0]
        hist = self.ctx.history(road, self.ctx.replay_dates(road)[0])
        with self.assertRaisesRegex(demo.DemoInputError, "at least"):
            demo.predict_all(hist.tail(3), self.ctx)
        bad = hist.copy()
        bad.loc[bad.index[-1], "Traffic Volume"] = np.nan
        with self.assertRaisesRegex(demo.DemoInputError, "missing values"):
            demo.predict_all(bad, self.ctx)
        bad = hist.copy()
        bad[fd.ROAD_COL] = "Nowhere Road"
        with self.assertRaisesRegex(demo.DemoInputError, "Unknown road"):
            demo.predict_all(bad, self.ctx)
        bad = hist.copy()
        bad.loc[bad.index[-1], "Weather Conditions"] = "Snow"
        with self.assertRaisesRegex(demo.DemoInputError, "Weather"):
            demo.predict_all(bad, self.ctx)
        with self.assertRaises(demo.DemoInputError):
            demo.predict_all(hist.iloc[::-1], self.ctx)  # wrong date order

    def test_out_of_range_value_gives_warning_not_silent_acceptance(self):
        road = self.ctx.roads[0]
        hist = self.ctx.history(road, self.ctx.replay_dates(road)[0])
        weird = demo.apply_scenario(hist, {"Traffic Volume": 10 ** 7})
        res = demo.predict_all(weird, self.ctx)
        self.assertTrue(any("outside the range" in w for w in res["warnings"]))

    def test_whatif_changes_only_the_last_row_and_changes_the_output(self):
        road = self.ctx.roads[0]
        hist = self.ctx.history(road, self.ctx.replay_dates(road)[0])
        heavy = demo.apply_scenario(hist, self.ctx.preset_values(road, list(demo.PRESETS)[0]))
        light = demo.apply_scenario(hist, self.ctx.preset_values(road, list(demo.PRESETS)[1]))
        pd.testing.assert_frame_equal(heavy.iloc[:-1], hist.iloc[:-1], check_dtype=False)  # earlier observations untouched
        rh, rl = demo.predict_all(heavy, self.ctx), demo.predict_all(light, self.ctx)
        print(f"\n[what-if] {road}: heavy XGB={rh['XGBoost']:.1f} LSTM={rh['LSTM']:.1f} | "
              f"light XGB={rl['XGBoost']:.1f} LSTM={rl['LSTM']:.1f}")
        self.assertNotEqual(rh["XGBoost"], rl["XGBoost"])
        self.assertNotEqual(rh["LSTM"], rl["LSTM"])

    def test_preset_values_come_from_training_period_only(self):
        road = self.ctx.roads[0]
        vals = self.ctx.preset_values(road, list(demo.PRESETS)[0])
        d = self.ctx.df[(self.ctx.df[fd.ROAD_COL] == road) & (self.ctx.df[fd.DATE_COL] <= self.ctx.train_end)]
        self.assertAlmostEqual(vals["Traffic Volume"], d["Traffic Volume"].quantile(0.9), places=3)
        self.assertAlmostEqual(vals["Average Speed"], d["Average Speed"].quantile(0.1), places=3)  # heavy = slow

    def test_model_version_hashes_exist(self):
        v = demo.model_version()
        self.assertEqual(set(v), {"XGBoost", "LSTM", "LSTM scaler"})
        self.assertTrue(all(len(x) == 10 for x in v.values()))


class RealtimeInterfaceTests(unittest.TestCase):
    """What `realtime` still provides (key lookup, the compatibility gap) and how provider data is obtained: only through the shared
    service in src/live/tomtom.py. All HTTP here is mocked; nothing verifies the real provider."""

    @staticmethod
    def http_ok(payload):
        return mock.Mock(status_code=200, json=lambda: payload)

    def test_unconfigured_state(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(realtime.ENV_KEY, None)
            self.assertIsNone(realtime.get_api_key())
            self.assertIsNone(realtime.get_api_key({}))

    def test_missing_key_is_refused_by_the_shared_service_before_any_request(self):
        from src.live import tomtom
        with mock.patch.object(tomtom.requests, "get") as g:
            for key in (None, "", "   "):
                with self.assertRaises(tomtom.ProviderError) as cm:
                    tomtom.FlowService().get(12.97, 77.59, key)
                self.assertEqual(cm.exception.kind, "no_key")
            g.assert_not_called()

    def test_key_is_read_from_environment_or_secrets(self):
        with mock.patch.dict(os.environ, {realtime.ENV_KEY: "abc"}):
            self.assertEqual(realtime.get_api_key(), "abc")
        os.environ.pop(realtime.ENV_KEY, None)
        self.assertEqual(realtime.get_api_key({realtime.ENV_KEY: "xyz"}), "xyz")

    def test_coordinate_validation_happens_before_any_request(self):
        from src.live import geo, tomtom
        with mock.patch.object(tomtom.requests, "get") as g:
            for lat, lon in ((123, 0), (12.97, 500), (48.85, 2.35)):  # out of range, out of range, Paris
                with self.assertRaises(geo.LocationError):
                    tomtom.FlowService().get(lat, lon, "k")
            g.assert_not_called()

    def test_response_parsing_with_mocked_http_and_missing_field_detection(self):
        """Parsing logic only: this does NOT verify the real provider."""
        from src.live import tomtom
        ok = {"flowSegmentData": {"frc": "FRC1", "currentSpeed": 30, "freeFlowSpeed": 50, "currentTravelTime": 120,
                                  "freeFlowTravelTime": 72, "confidence": 0.9, "roadClosure": False,
                                  "coordinates": {"coordinate": [{"latitude": 12.9716, "longitude": 77.5946},
                                                                 {"latitude": 12.9720, "longitude": 77.5950}]}}}
        with mock.patch.object(tomtom.requests, "get", return_value=self.http_ok(ok)) as g:
            obs, source = tomtom.FlowService().get(12.97, 77.59, "k")
            self.assertEqual(g.call_args.kwargs["params"]["point"], "12.97,77.59")
        self.assertEqual((source, obs.current_speed_kmh), ("provider", 30))
        self.assertFalse(hasattr(obs, "provider_timestamp"))  # the provider gives no observation time; only our fetch time exists
        self.assertGreater(obs.fetched_at, 0)
        bad = {"flowSegmentData": {"currentSpeed": 30}}
        with mock.patch.object(tomtom.requests, "get", return_value=self.http_ok(bad)):
            with self.assertRaises(tomtom.ProviderError) as cm:
                tomtom.FlowService().get(12.97, 77.59, "k")
        self.assertEqual(cm.exception.kind, "invalid_response")

    def test_the_unsafe_direct_client_no_longer_exists(self):
        """The stand-alone request function (no cache, request cap, throttle or cooldown) was removed; nothing may bring it back."""
        self.assertFalse(hasattr(realtime, "fetch_flow_segment"))
        self.assertFalse(hasattr(realtime, "requests"), "realtime must not import an HTTP library")
        self.assertFalse(hasattr(realtime, "RealtimeConfigError"))
        src = Path(realtime.__file__).read_text(encoding="utf-8")
        for needle in ("requests", "urllib", "api.tomtom.com", "flowSegmentData", "def fetch_"):
            self.assertNotIn(needle, src.replace("tomtom.SERVICE", ""), needle)

    def test_prediction_is_always_blocked_for_provider_data(self):
        blockers = " ".join(realtime.prediction_blockers())
        for needed in ("Traffic Volume", "Road Capacity Utilization", "Weather", "history"):
            self.assertIn(needed, blockers)


@needs_data
class SparkOutputTests(unittest.TestCase):
    def test_spark_road_means_match_pandas(self):
        p = root_dir / "reports" / "spark" / "by_road.csv"
        if not p.exists():
            self.skipTest("Spark outputs not generated (python -m src.bigdata.spark_analysis)")
        sp = pd.read_csv(p).set_index("Road/Intersection Name")
        df = fd.load_raw()
        pdm = df.groupby(fd.ROAD_COL)[fd.TARGET_COL].mean()
        pd.testing.assert_series_equal(sp["mean_congestion"].sort_index(), pdm.sort_index().round(3),
                                       check_names=False, atol=2e-3)
        self.assertEqual(int(sp["n_obs"].sum()), len(df))
        s = json.loads((root_dir / "reports" / "spark" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["rows_after_cleaning"], len(df))


@needs_all
class StreamlitAppTests(unittest.TestCase):
    def test_app_runs_and_demo_predict_reset_and_modes(self):
        from streamlit.testing.v1 import AppTest
        os.environ.pop(realtime.ENV_KEY, None)
        # the Traffic Forecasts page (formerly the "Live Prediction Demo" tab of the single-page app)
        at = AppTest.from_file(str(root_dir / "src" / "ui" / "pages" / "forecasts.py"), default_timeout=300).run()
        self.assertEqual([e.value for e in at.exception], [])
        # --- historical replay prediction ---
        at.button(key="demo_predict").click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertIn("demo_result", at.session_state)
        res = at.session_state["demo_result"]
        self.assertEqual(res["source"], "Historical replay")
        shown = {m.label: m.value for m in at.metric}
        self.assertIn("XGBoost", shown)
        self.assertIn("LSTM", shown)
        self.assertAlmostEqual(float(shown["XGBoost"]), res["XGBoost"], places=1)
        # --- what-if ---
        at.radio(key="demo_source").set_value("What-if scenario (edit day-t conditions)").run()
        at.selectbox(key="demo_preset").set_value(list(demo.PRESETS)[0]).run()
        at.button(key="demo_predict").click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(at.session_state["demo_result"]["source"], "What-if scenario")
        self.assertEqual(len(at.session_state["demo_log"]), 2)
        # --- reset ---
        at.button(key="demo_reset").click().run()
        self.assertNotIn("demo_result", at.session_state)
        # --- real-time mode, unconfigured ---
        at.radio(key="demo_mode").set_value("Mode 2: Real-Time Traffic API").run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("No API key configured" in i.value for i in at.info))
        self.assertTrue(any("cannot be used to make a valid prediction" in w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()

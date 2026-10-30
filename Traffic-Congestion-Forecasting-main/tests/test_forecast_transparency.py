"""
Stage 4A tests: (1) the forecasting UI describes the models as experimental and shows the baseline beside them;
(2) Mode 2 of the Traffic Forecasts page uses the SAME provider service (and so the same safeguards) as the Live Traffic Map.

Provider traffic is MOCKED everywhere; no real request is made and no key is needed. A passing test here is not evidence that the real
TomTom endpoint works. All metric values come from the SAVED model/metrics.json; nothing is recomputed or invented.

Run:   python -m unittest tests.test_forecast_transparency -v
"""
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database: tests never touch the real community.db)
os.environ.pop("LIVE_CACHE_TTL_S", None)  # tests must not depend on a local runtime setting; tests that need it set it themselves

import requests  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.app import realtime  # noqa: E402
from src.live import geo, tomtom  # noqa: E402
from src.ui import common  # noqa: E402
from tests.test_live_map import KEY, FakeResp, PageHarness, flow_payload, snapshot  # noqa: E402

FORECASTS = str(root_dir / "src" / "ui" / "pages" / "forecasts.py")
PERFORMANCE = str(root_dir / "src" / "ui" / "pages" / "performance.py")
METRICS = json.loads((root_dir / "model" / "metrics.json").read_text(encoding="utf-8"))
M = METRICS["metrics"]
FETCH = "Fetch provider observation"
MODE2 = "Mode 2: Real-Time Traffic API"
POINT = (12.9159, 77.6240)


def texts(at):
    out = []
    for g in (at.markdown, at.caption, at.info, at.warning, at.error, at.success):
        out += [e.value for e in g]
    return " ".join(out)


def table_rows(at):
    """Every st.table of the page as a list of row-lists of strings."""
    return [[[str(c) for c in row] for row in t.value.values.tolist()] for t in at.table]


# ======================================================================================================== wording
class ComparisonHelperTests(unittest.TestCase):
    def test_table_matches_the_saved_metrics_with_the_baseline_first(self):
        df = common.comparison_table(M)
        self.assertEqual(list(df.columns), ["Model", "MAE (points)", "RMSE (points)", "R²", "RMSE minus road-mean RMSE"])
        self.assertEqual(list(df["Model"]), ["Road-mean baseline (each road's training-period average)", "Persistence baseline (tomorrow = today)",
                                             "XGBoost (experimental)", "LSTM (experimental)"])
        keys = ["Road-mean baseline", "Persistence baseline", "XGBoost", "LSTM"]
        for key, (_, row) in zip(keys, df.iterrows()):
            self.assertEqual(row["MAE (points)"], f"{M[key]['MAE']:.2f}")
            self.assertEqual(row["RMSE (points)"], f"{M[key]['RMSE']:.2f}")
            self.assertEqual(row["R²"], f"{M[key]['R2']:.3f}")
        self.assertEqual(df.iloc[0]["RMSE minus road-mean RMSE"], "reference")
        for key, (_, row) in zip(keys[1:], list(df.iterrows())[1:]):
            self.assertEqual(row["RMSE minus road-mean RMSE"], f"{M[key]['RMSE'] - M['Road-mean baseline']['RMSE']:+.2f}")
        # the figures quoted in the Stage 3 audit
        self.assertEqual([df.iloc[i]["RMSE (points)"] for i in range(4)], ["19.85", "28.23", "20.07", "19.93"])

    def test_missing_rows_are_left_out_not_invented(self):
        df = common.comparison_table({"XGBoost": M["XGBoost"]})
        self.assertEqual(list(df["Model"]), ["XGBoost (experimental)"])
        self.assertEqual(df.iloc[0]["RMSE minus road-mean RMSE"], "n/a")

    def test_notes_use_only_saved_intervals_and_say_they_are_not_recomputed(self):
        text = common.comparison_notes(METRICS)
        self.assertIn(f"{METRICS['n_test_samples']} held-out test samples", text)
        self.assertIn("2024-03-21 to 2024-08-09", text)
        self.assertIn("lower MAE and RMSE and higher R² are better", text)
        for k in ("LSTM - Road-mean baseline", "XGBoost - Road-mean baseline"):
            d = METRICS["uncertainty"]["rmse_difference"][k]
            self.assertIn(f"{d['observed']:+.2f} (95% CI {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})", text)
        self.assertIn("not recomputed here", text)
        self.assertIn("does not show either model to be better or worse than the baseline", text)

    def test_notes_do_not_claim_inclusion_of_zero_when_it_is_not_true_and_do_not_invent_intervals(self):
        excluded = {"metrics": M, "uncertainty": {"rmse_difference": {
            "LSTM - Road-mean baseline": {"observed": -1.0, "ci95": [-2.0, -0.5], "ci_excludes_zero": True},
            "XGBoost - Road-mean baseline": {"observed": 0.2, "ci95": [-0.1, 0.5], "ci_excludes_zero": False}}}}
        self.assertNotIn("does not show either model", common.comparison_notes(excluded))
        bare = common.comparison_notes({"metrics": M})
        self.assertNotIn("95% CI", bare)
        self.assertNotIn("bootstrap", bare)


class ForecastPageWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(os.environ):
            os.environ.pop("TOMTOM_API_KEY", None)
            cls.at = AppTest.from_file(FORECASTS, default_timeout=300).run()
            cls.at.button(key="demo_predict").click().run()

    def test_page_runs(self):
        self.assertEqual([e.value for e in self.at.exception], [])

    def test_the_models_are_described_as_experimental_and_not_better_than_the_baseline(self):
        text = texts(self.at)
        for needle in ("The LSTM and XGBoost models are experimental",
                       "neither beat the simple per-road historical-mean baseline",
                       "does not show that either model predicts day-to-day changes",
                       "performance has not been established: the dataset ends on 2024-08-09",
                       "not live traffic and not validated 2026 forecasts"):
            self.assertIn(needle, text)
        self.assertIn("Experimental model output", text)

    def test_existing_chips_and_target_definition_are_preserved(self):
        text = texts(self.at)
        for needle in ("Historical data 2022-2024", "Not validated for 2026", "Experimental models", "Baselines shown beside the models",
                       "horizon: 1 day ahead (next calendar day)", "target: Congestion Level (0-100)"):
            self.assertIn(needle, text)

    def test_baseline_comparison_table_sits_beside_the_forecast(self):
        rows = next(r for r in table_rows(self.at) if r and r[0][0].startswith("Road-mean baseline"))
        self.assertEqual([r[0] for r in rows], list(common.comparison_table(M)["Model"]))
        self.assertEqual([r[2] for r in rows], [f"{M[k]['RMSE']:.2f}" for k in ("Road-mean baseline", "Persistence baseline", "XGBoost", "LSTM")])
        self.assertIn("(this applies to the whole test set, not to the single forecast above)", texts(self.at))

    def test_forecast_outputs_and_labels_are_unchanged(self):
        shown = {m.label: m.value for m in self.at.metric}
        for label in ("XGBoost", "LSTM", "Persistence baseline", "Road-mean baseline", "Actual next day"):
            self.assertIn(label, shown)
        res = self.at.session_state["demo_result"]
        self.assertEqual((res["source"], res["horizon"]), ("Historical replay", "1 day ahead (next calendar day)"))
        self.assertAlmostEqual(float(shown["XGBoost"]), res["XGBoost"], places=1)


class PerformancePageWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = AppTest.from_file(PERFORMANCE, default_timeout=300).run()

    def test_overview_shows_the_baseline_beside_the_models(self):
        self.assertEqual([e.value for e in self.at.exception], [])
        rows = next(r for r in table_rows(self.at) if r and r[0][0].startswith("Road-mean baseline"))
        self.assertEqual(rows[0][4], "reference")
        self.assertEqual(rows[2][0], "XGBoost (experimental)")
        self.assertEqual(rows[3][0], "LSTM (experimental)")
        self.assertEqual(rows[3][4], f"{M['LSTM']['RMSE'] - M['Road-mean baseline']['RMSE']:+.2f}")

    def test_experimental_status_and_missing_recent_validation_are_stated(self):
        text = texts(self.at)
        self.assertIn("Experimental models. Honest finding: neither learned model beats the simple per-road historical-mean baseline", text)
        self.assertIn("performance on recent real-world conditions has not been established", text)
        self.assertIn("Recent real-world performance has not been established: the dataset ends on 2024-08-09", text)
        self.assertIn("about 43.5% of congestion values are exactly 100", text)

    def test_existing_findings_are_still_there(self):
        text = texts(self.at)
        self.assertIn("No winner between LSTM and XGBoost", text)
        self.assertIn("Neither model beats the simple", text)
        self.assertIn("Not validated for 2026", text)


# ======================================================================================================== Mode 2 routing
class Mode2Harness:
    """Runs the Traffic Forecasts page in Mode 2 with a mocked provider and a given FlowService."""

    def __init__(self, key=KEY, service=None, outcome=None, cache_ttl=None):
        self.key, self.net = key, []
        self.outcome = outcome if outcome is not None else FakeResp(200, flow_payload())
        self.service = service or tomtom.FlowService(ttl=300 if cache_ttl is None else cache_ttl)

    def _get(self, url, **kw):
        self.net.append((url, kw))
        if "tomtom" not in url:
            raise AssertionError(f"unexpected network call: {url}")
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def start(self):
        self._patches = [mock.patch.dict(os.environ), mock.patch.object(requests, "get", side_effect=self._get),
                         mock.patch.object(tomtom, "SERVICE", self.service)]
        for p in self._patches:
            p.start()
        os.environ.pop("TOMTOM_API_KEY", None)
        if self.key:
            os.environ["TOMTOM_API_KEY"] = self.key
        self.at = AppTest.from_file(FORECASTS, default_timeout=300).run()
        self.at.radio(key="demo_mode").set_value(MODE2).run()
        if self.key:
            self.set_point(*POINT)  # the mocked segment lies near this point; a far-away point is (correctly) rejected as invalid
        return self

    def stop(self):
        for p in reversed(self._patches):
            p.stop()

    def set_point(self, lat, lon):
        self.at.number_input[0].set_value(lat)
        self.at.number_input[1].set_value(lon)
        return self

    def fetch(self):
        next(b for b in self.at.button if b.label == FETCH).click()
        self.at.run()
        return self

    def texts(self):
        return texts(self.at)

    def calls(self):
        return [n for n in self.net if "tomtom" in n[0]]


class Mode2RoutingTests(unittest.TestCase):
    def harness(self, **kw):
        h = Mode2Harness(**kw).start()
        self.addCleanup(h.stop)
        return h

    def test_fetch_goes_through_the_shared_service_and_shows_a_labelled_reading(self):
        h = self.harness()
        with mock.patch.object(h.service, "get", wraps=h.service.get) as spy:
            h.fetch()
        self.assertEqual([e.value for e in h.at.exception], [])
        spy.assert_called_once()
        args, kw = spy.call_args
        self.assertEqual((args[2], kw), (KEY, {"force": False}))
        self.assertEqual(h.service.requests_made, 1)
        self.assertEqual(len(h.calls()), 1)
        self.assertEqual(h.calls()[0][1]["params"]["key"], KEY)  # sent server-side only
        rows = dict(next(r for r in table_rows(h.at) if r and r[0][0] == "Provider"))
        self.assertEqual(rows["Current speed (km/h)"], "24")
        self.assertEqual(rows["Provider measurement time"], "not provided by this API")
        self.assertIn("this computer's clock", " ".join(k for k in rows))
        text = h.texts()
        self.assertIn("This is a live observation, not a forecast", text)
        self.assertIn("Prediction blocked. Missing model inputs", text)
        self.assertIn("Traffic data: © TomTom", text)
        self.assertNotIn(KEY, text + repr(table_rows(h.at)))

    def test_there_is_no_direct_provider_call_outside_the_shared_client(self):
        self.assertFalse(hasattr(realtime, "fetch_flow_segment"), "the unsafe stand-alone client must stay removed")
        for f in (root_dir / "src").rglob("*.py"):
            self.assertNotIn("fetch_flow_segment", f.read_text(encoding="utf-8"), f.name)
        users = [f.relative_to(root_dir).as_posix() for f in (root_dir / "src").rglob("*.py")
                 if re.search(r"api\.tomtom\.com|flowSegmentData/|traffic/services", f.read_text(encoding="utf-8"))]
        self.assertEqual(users, ["src/live/tomtom.py"], "TomTom may be contacted from exactly one module")
        for f in (root_dir / "src" / "ui").rglob("*.py"):
            self.assertNotRegex(f.read_text(encoding="utf-8"), r"requests\.(get|post)|import requests", f.name)

    def test_inputs_are_limited_to_the_bengaluru_area(self):
        h = self.harness()
        lat, lon = h.at.number_input[0], h.at.number_input[1]
        self.assertEqual((lat.min, lat.max), (geo.BENGALURU_BBOX[0], geo.BENGALURU_BBOX[2]))
        self.assertEqual((lon.min, lon.max), (geo.BENGALURU_BBOX[1], geo.BENGALURU_BBOX[3]))

    def test_missing_key_shows_setup_help_and_makes_no_request(self):
        h = self.harness(key=None)
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertIn("No API key configured", " ".join(i.value for i in h.at.info))
        self.assertNotIn(FETCH, [b.label for b in h.at.button])
        self.assertEqual((h.net, h.service.requests_made), ([], 0))

    def test_provider_failures_are_explained_without_a_crash_a_reading_or_the_key(self):
        cases = [(requests.Timeout("slow"), "did not answer in time"), (requests.ConnectionError(f"boom key={KEY}"), "Could not reach TomTom"),
                 (FakeResp(401, {}), "rejected the request"), (FakeResp(403, {}), "rejected the request"),
                 (FakeResp(429, {}), "limiting requests"), (FakeResp(400, {}), "no traffic segment"),
                 (FakeResp(503, {}), "temporarily unavailable"), (FakeResp(200, {"unexpected": 1}), "rejected"),
                 (FakeResp(200, json_error=True), "could not be read"),
                 (FakeResp(200, flow_payload(currentSpeed="fast")), "invalid 'currentSpeed'")]
        for outcome, expected in cases:
            with self.subTest(expected):
                h = Mode2Harness(outcome=outcome).start()
                try:
                    h.fetch()
                    self.assertEqual([e.value for e in h.at.exception], [])
                    shown = " ".join(x.value for g in (h.at.error, h.at.warning, h.at.info) for x in g)
                    self.assertIn(expected, shown)
                    self.assertEqual(len(h.calls()), 1)
                    self.assertFalse([r for r in table_rows(h.at) if r and r[0][0] == "Provider"], "no reading may be shown")
                    self.assertIn("No provider reading yet", shown)
                    self.assertNotIn(KEY, h.texts())
                finally:
                    h.stop()

    def test_an_unexpected_service_error_is_contained_and_never_shows_its_text(self):
        h = self.harness()
        with mock.patch.object(h.service, "get", side_effect=RuntimeError(f"boom with {KEY}")):
            h.fetch()
        self.assertEqual([e.value for e in h.at.exception], [])
        text = h.texts()
        self.assertIn("Unexpected problem while requesting traffic (RuntimeError). No reading was produced.", text)
        self.assertNotIn(KEY, text)
        self.assertFalse([r for r in table_rows(h.at) if r and r[0][0] == "Provider"])

    def test_429_starts_the_shared_cooldown_for_this_page(self):
        h = self.harness(outcome=FakeResp(429, {}))
        h.fetch()
        h.set_point(12.95, 77.65).fetch()  # a different point, inside the cooldown
        self.assertEqual(len(h.calls()), 1, "the second request must not reach the provider")
        self.assertIn("wait about", " ".join(w.value for w in h.at.warning))

    def test_request_cap_applies_to_this_page(self):
        h = self.harness(service=tomtom.FlowService(ttl=300, budget=1))
        h.fetch()
        h.set_point(12.95, 77.65).fetch()
        self.assertEqual(len(h.calls()), 1)
        self.assertIn("The limit of 1 provider requests", " ".join(w.value for w in h.at.warning))
        self.assertEqual(len(table_rows(h.at)[0]), 11, "the earlier reading stays on screen")

    def test_with_ttl_zero_an_early_refresh_is_refused_and_the_reading_stays(self):
        h = self.harness(cache_ttl=0)
        h.fetch()
        h.fetch()  # same point again, straight away: this is a refresh
        self.assertEqual(len(h.calls()), 1, "a refused refresh must not reach the provider")
        self.assertIn("once every 30 seconds", " ".join(i.value for i in h.at.info))
        self.assertTrue([r for r in table_rows(h.at) if r and r[0][0] == "Provider"])
        self.assertEqual(h.service._cache, {}, "ttl 0 keeps no provider reading")

    def test_with_the_default_cache_a_quick_refresh_is_answered_from_memory(self):
        h = self.harness(cache_ttl=300)
        h.fetch().fetch()
        self.assertEqual(len(h.calls()), 1)
        self.assertIn("once every 30 seconds", " ".join(i.value for i in h.at.info))

    def test_an_old_reading_is_labelled_stale(self):
        h = self.harness(service=tomtom.FlowService(ttl=300, clock=lambda: time.time() - 700))
        h.fetch()
        self.assertIn("Stale reading", " ".join(w.value for w in h.at.warning))
        self.assertIn("It is not a current observation", " ".join(w.value for w in h.at.warning))

    def test_nothing_is_written_to_disk(self):
        before = snapshot()
        h = self.harness()
        h.fetch()
        self.assertEqual(snapshot() - before, set())


class SharedSafeguardsTests(unittest.TestCase):
    """The Live Traffic Map and the Forecasts page must act on ONE service, so its limits cover both."""

    def run_map(self, service):
        h = PageHarness(key=KEY)
        h.service = service
        h.start()
        self.addCleanup(h.stop)
        return h.search("silk board junction").choose(0).fetch()

    def test_a_reading_fetched_on_the_map_is_served_to_the_forecasts_page_from_the_same_cache(self):
        shared = tomtom.FlowService(ttl=300)
        m = self.run_map(shared)
        self.assertEqual(len(m.tomtom_calls()), 1)
        f = Mode2Harness(service=shared).start()
        self.addCleanup(f.stop)
        lat, lon = m.at.session_state["lm_selected"]["lat"], m.at.session_state["lm_selected"]["lon"]
        f.set_point(round(lat, 5), round(lon, 5)).fetch()
        self.assertEqual(f.calls(), [], "same point inside the cache window: no second provider request")
        self.assertEqual(shared.requests_made, 1)
        self.assertTrue([r for r in table_rows(f.at) if r and r[0][0] == "Provider"])

    def test_the_request_cap_is_one_budget_for_both_pages(self):
        shared = tomtom.FlowService(ttl=300, budget=1)
        self.run_map(shared)
        f = Mode2Harness(service=shared).start()
        self.addCleanup(f.stop)
        f.set_point(12.95, 77.65).fetch()
        self.assertEqual(f.calls(), [])
        self.assertIn("The limit of 1 provider requests", " ".join(w.value for w in f.at.warning))

    def test_a_429_seen_on_the_map_pauses_the_forecasts_page_too(self):
        shared = tomtom.FlowService(ttl=300)
        h = PageHarness(key=KEY, tomtom_outcome=FakeResp(429, {}))
        h.service = shared
        h.start()
        self.addCleanup(h.stop)
        h.search("silk board junction").choose(0).fetch()
        f = Mode2Harness(service=shared).start()
        self.addCleanup(f.stop)
        f.fetch()
        self.assertEqual(f.calls(), [])
        self.assertIn("wait about", " ".join(w.value for w in f.at.warning))

    def test_both_pages_use_the_one_request_helper(self):
        src_map = (root_dir / "src" / "ui" / "live_map.py").read_text(encoding="utf-8")
        src_fc = (root_dir / "src" / "ui" / "forecasts_sections.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"tomtom\.SERVICE\.get\(", src_map + src_fc)), 1, "exactly one call site into the service")
        self.assertIn("live_map.request_reading(", src_fc)
        self.assertIn("request_reading(sel", src_map)


if __name__ == "__main__":
    unittest.main()

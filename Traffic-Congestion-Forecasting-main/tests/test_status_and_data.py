"""
Tests for the status model (src/ui/health.py, src/ui/diagnostics.py, the provider's status()), the homepage summary, the forecast
data-gap messaging, the Model Insights explanation and the read-only new-data validator (src/dataProcessing/check_new_data.py).

All provider behaviour is MOCKED; nothing needs a key, and nothing here says anything about the real TomTom service. Synthetic CSV files
are created only inside temporary folders, to test the validator; they are never placed in the project.

Run:   python -m unittest tests.test_status_and_data -v
"""
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_data, needs_models, needs_outputs  # noqa: E402

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.community import store  # noqa: E402
from src.dataProcessing import check_new_data as cnd  # noqa: E402
from src.dataProcessing import forecast_data as fd  # noqa: E402
from src.live import tomtom  # noqa: E402
from src.ui import common, diagnostics, health  # noqa: E402
from tests.test_live_map import KEY, flow_payload  # noqa: E402

LIVE_PAGE = str(root_dir / "src" / "ui" / "pages" / "live_map.py")
FORECASTS = str(root_dir / "src" / "ui" / "pages" / "forecasts.py")
PERFORMANCE = str(root_dir / "src" / "ui" / "pages" / "performance.py")
METRICS = json.loads((root_dir / "model" / "metrics.json").read_text(encoding="utf-8")) if (root_dir / "model" / "metrics.json").is_file() else {}
NOW = 1_800_000_000.0
STATUS0 = {"requests_made": 0, "budget": 300, "ttl_s": 0, "cooldown_remaining_s": 0.0, "last_success_at": None, "last_error": None}


def md(at):
    return " ".join(m.value for m in at.markdown)


def alltext(at):
    out = []
    for g in (at.markdown, at.caption, at.info, at.warning, at.error, at.success):
        out += [e.value for e in g]
    return " ".join(out)


# ======================================================================================================== provider status()
class ProviderStatusTests(unittest.TestCase):
    def service(self, outcomes, **kw):
        self.t = 1000.0
        it = iter(outcomes)

        def fetcher(lat, lon, key):
            o = next(it)
            if isinstance(o, Exception):
                raise o
            return tomtom.parse_response(flow_payload(), lat, lon, self.t)
        return tomtom.FlowService(fetcher=fetcher, clock=lambda: self.t, ttl=0, **kw)

    def test_a_fresh_service_has_seen_nothing(self):
        s = self.service([]).status()
        self.assertEqual((s["requests_made"], s["last_success_at"], s["last_error"], s["cooldown_remaining_s"]), (0, None, None, 0.0))

    def test_a_valid_response_records_only_the_time(self):
        svc = self.service([None])
        svc.get(12.9159, 77.6240, KEY)
        st = svc.status()
        self.assertEqual((st["requests_made"], st["last_success_at"], st["last_error"]), (1, 1000.0, None))
        self.assertEqual(set(st), {"requests_made", "budget", "ttl_s", "cooldown_remaining_s", "last_success_at", "last_error"}, "no data and no key in status")
        self.assertNotIn(KEY, json.dumps(st))

    def test_a_provider_failure_records_its_kind_and_time_and_keeps_the_last_success(self):
        svc = self.service([None, tomtom.ProviderError("timeout", "slow")])
        svc.get(12.9159, 77.6240, KEY)
        self.t = 1100.0
        with self.assertRaises(tomtom.ProviderError):
            svc.get(12.9159, 77.6240, KEY)
        st = svc.status()
        self.assertEqual((st["last_success_at"], st["last_error"]), (1000.0, ("timeout", 1100.0)))

    def test_safeguard_refusals_are_not_recorded_as_provider_failures(self):
        svc = self.service([None], budget=1)
        svc.get(12.9159, 77.6240, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(12.95, 77.65, KEY)
        self.assertEqual(cm.exception.kind, "budget")
        with self.assertRaises(tomtom.ProviderError) as cm2:
            svc.get(12.9159, 77.6240, KEY, force=True)
        self.assertEqual(cm2.exception.kind, "throttled")
        self.assertIsNone(svc.status()["last_error"], "the provider was not asked, so it cannot have failed")

    def test_a_429_shows_the_cooldown_and_clear_resets_everything(self):
        svc = self.service([tomtom.ProviderError("rate_limited", "x")])
        with self.assertRaises(tomtom.ProviderError):
            svc.get(12.9159, 77.6240, KEY)
        st = svc.status()
        self.assertEqual(st["last_error"][0], "rate_limited")
        self.assertAlmostEqual(st["cooldown_remaining_s"], tomtom.RATE_LIMIT_COOLDOWN_S, delta=1)
        svc.clear()
        st = svc.status()
        self.assertEqual((st["requests_made"], st["last_success_at"], st["last_error"], st["cooldown_remaining_s"]), (0, None, None, 0.0))

    def test_the_status_does_not_change_the_safeguards(self):
        self.assertEqual((tomtom.CACHE_TTL_S, tomtom.MIN_REFRESH_S, tomtom.RUN_BUDGET_DEFAULT, tomtom.RATE_LIMIT_COOLDOWN_S), (300, 30, 300, 60))
        with mock.patch.dict(os.environ, {"LIVE_CACHE_TTL_S": "0"}):
            self.assertEqual(tomtom.FlowService().ttl, 0)


# ======================================================================================================== health items
class HealthItemTests(unittest.TestCase):
    def test_no_key_is_red_and_says_how_to_fix_it(self):
        i = health.provider_item(False, STATUS0, NOW)
        self.assertEqual((i.state, i.chip, i.value, i.scope), ("bad", "Live traffic: not configured", "Not configured", "external"))
        self.assertIn("TOMTOM_API_KEY", i.summary)
        self.assertIn("map and place search still work", i.summary)

    def test_a_key_that_was_never_used_is_unknown_not_working(self):
        i = health.provider_item(True, STATUS0, NOW)
        self.assertEqual((i.state, i.value), ("unknown", "Key found, not tested"))
        self.assertIn("not yet known whether the key works", i.summary)
        self.assertNotEqual(i.state, "ok")

    def test_a_valid_response_is_ok_and_says_the_provider_gives_no_measurement_time(self):
        i = health.provider_item(True, {**STATUS0, "requests_made": 2, "last_success_at": NOW - 30}, NOW)
        self.assertEqual(i.state, "ok")
        self.assertIn("no measurement time", i.summary)
        self.assertIn("this computer's clock", i.summary)

    def test_failures_map_to_honest_states(self):
        for kind, state in (("auth", "bad"), ("invalid_response", "bad"), ("timeout", "warn"), ("network", "warn"), ("server", "warn"),
                            ("rate_limited", "warn"), ("no_segment", "warn"), ("unheard_of", "warn")):
            with self.subTest(kind):
                i = health.provider_item(True, {**STATUS0, "last_error": (kind, NOW - 5)}, NOW)
                self.assertEqual((i.state, i.value), (state, "Provider problem"))
                self.assertIn("No valid response has been received yet", i.summary)

    def test_the_newer_event_wins_and_the_last_valid_response_is_mentioned(self):
        newer_error = health.provider_item(True, {**STATUS0, "last_success_at": NOW - 100, "last_error": ("timeout", NOW - 10)}, NOW)
        self.assertEqual(newer_error.state, "warn")
        self.assertIn("The last valid response was at", newer_error.summary)
        newer_success = health.provider_item(True, {**STATUS0, "last_success_at": NOW - 10, "last_error": ("timeout", NOW - 100)}, NOW)
        self.assertEqual(newer_success.state, "ok")

    def test_the_cooldown_is_mentioned_and_no_secret_or_url_can_appear(self):
        i = health.provider_item(True, {**STATUS0, "last_error": ("rate_limited", NOW - 1), "cooldown_remaining_s": 42.0}, NOW)
        self.assertIn("paused for about 43 s", i.summary)
        for it in (i, health.provider_item(False, STATUS0, NOW), health.provider_item(True, STATUS0, NOW)):
            self.assertNotRegex(" ".join((it.summary, it.detail, it.chip)), r"key=|https?://|api\.tomtom")

    def frame(self, last="2024-08-09"):
        return pd.DataFrame({fd.DATE_COL: pd.to_datetime(["2022-01-01", last]), "x": [1, 2]})

    def test_the_dataset_item_reports_its_real_age(self):
        now = pd.Timestamp("2026-10-29 12:00").timestamp()
        i = health.data_item(self.frame(), None, now)
        self.assertEqual((i.state, i.value, i.scope), ("warn", "Latest 2024-08-09", "data"))
        self.assertEqual(i.chip, "Data: historical, 2022-2024 (latest 2024-08-09)")
        self.assertIn("811 days old", i.summary)
        self.assertIn("cannot reach recent dates", i.summary)
        self.assertIn("no live observation feed", i.detail)

    def test_a_recent_dataset_is_ok(self):
        now = pd.Timestamp("2024-08-20 12:00").timestamp()
        i = health.data_item(self.frame(), None, now)
        self.assertEqual(i.state, "ok")
        self.assertNotIn("cannot reach", i.summary)

    def test_a_dataset_problem_is_red_and_hides_the_local_path(self):
        err = f"Dataset not found: {root_dir}\\data\\raw\\x.csv\nDownload it."
        i = health.data_item(None, err, NOW, root=str(root_dir))
        self.assertEqual((i.state, i.value), ("bad", "Unavailable"))
        self.assertNotIn(str(root_dir), i.summary)
        self.assertIn("<project>", i.summary)
        self.assertNotIn("\n", i.summary)

    def test_model_items(self):
        miss = health.models_item(["model.h5", "scaler.pkl"])
        self.assertEqual((miss.state, miss.value), ("bad", "Unavailable"))
        self.assertIn("model.h5, scaler.pkl", miss.summary)
        self.assertEqual(health.models_item([], ["Could not load the LSTM: boom"]).state, "bad")
        ok = health.models_item([])
        self.assertEqual((ok.state, ok.chip, ok.value), ("warn", "Models: not validated for 2026", "Experimental"))
        self.assertIn("neither model beat", ok.summary)

    def test_community_item_follows_the_store(self):
        self.assertEqual(health.community_item({"state": "ok", "summary": "working; 3 report(s) stored", "count": 3}).value, "3 saved")
        bad = health.community_item({"state": "bad", "summary": "unavailable", "count": None})
        self.assertEqual((bad.state, bad.value), ("bad", "Unavailable"))

    def test_nothing_is_green_when_a_check_failed(self):
        items = health.build_items(has_key=False, provider_status=STATUS0, df=None, data_error="broken", models_missing=["model.h5"],
                                   community={"state": "bad", "summary": "x", "count": None}, now=NOW)
        self.assertEqual([i.key for i in items], ["provider", "data", "models", "community"])
        self.assertEqual({i.state for i in items}, {"bad"})
        self.assertEqual({health.KIND[i.state] for i in items}, {"bad"})

    @needs_data
    @needs_models
    def test_collect_reflects_this_machine_without_a_key_or_a_network_request(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {store.DB_ENV: str(Path(tmp) / "c.db")}), \
                mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()), mock.patch("requests.get", side_effect=AssertionError("network used")):
            os.environ.pop("TOMTOM_API_KEY", None)
            items = {i.key: i for i in health.collect()}
        self.assertEqual(items["provider"].state, "bad")
        self.assertEqual((items["data"].state, items["data"].value), ("warn", "Latest 2024-08-09"))
        self.assertEqual(items["models"].state, "warn")
        self.assertEqual(items["community"].state, "ok")


class DiagnosticsTests(unittest.TestCase):
    def test_file_rows_come_from_the_file_system(self):
        rows = {n: (present, req) for n, present, req in diagnostics.file_rows()}
        self.assertEqual(set(rows), {n for n, _, _ in diagnostics.FILES})
        self.assertTrue(all(isinstance(p, bool) for p, _ in rows.values()))
        self.assertEqual(rows["Dataset"][0], os.path.exists(fd.DATA_PATH))

    def test_a_missing_required_file_is_shown_as_missing_and_the_panel_never_shows_a_path(self):
        def script():
            from unittest import mock as m
            from src.ui import diagnostics as d, health as h
            gone = (root_dir_str + "/model/definitely_absent.csv")
            with m.patch.object(d, "FILES", [("Required thing", lambda: gone, True), ("Optional thing", lambda: gone, False)]):
                d.render([h.provider_item(False, {"requests_made": 0, "budget": 300, "ttl_s": 0, "cooldown_remaining_s": 0.0,
                                                  "last_success_at": None, "last_error": None}, 0.0)])
        src = f"root_dir_str = {str(root_dir)!r}\n" + "import sys\nsys.path.insert(0, root_dir_str)\n" + _function_body(script)
        at = AppTest.from_string(src, default_timeout=60).run()
        self.assertEqual([e.value for e in at.exception], [])
        text = md(at)
        self.assertIn("Required thing: MISSING", text)
        self.assertIn("Optional thing: not present (optional)", text)
        self.assertNotIn(str(root_dir).replace("\\", "/"), text)
        self.assertIn("Live traffic (TomTom)", text)
        self.assertEqual([b.label for b in at.button], ["Check models load", "Run preflight"])


def _function_body(fn):
    import inspect
    import textwrap
    lines = inspect.getsource(fn).splitlines()[1:]
    return textwrap.dedent("\n".join(lines)) + "\n"


# ======================================================================================================== homepage
class HomepageTests(unittest.TestCase):
    def run_page(self, key=None, service=None, db=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = {store.DB_ENV: str(db or Path(tmp.name) / "c.db")}
        with mock.patch.dict(os.environ, env), mock.patch.object(tomtom, "SERVICE", service or tomtom.FlowService()):
            os.environ.pop("TOMTOM_API_KEY", None)
            if key:
                os.environ["TOMTOM_API_KEY"] = key
            return AppTest.from_file(LIVE_PAGE, default_timeout=120).run()

    def test_without_a_key_a_setup_banner_is_shown_at_the_top(self):
        at = self.run_page()
        self.assertEqual([e.value for e in at.exception], [])
        banner, page = at.warning[0].value, md(at)
        for needle in ("Live traffic is not configured.", "TOMTOM_API_KEY", "README"):
            self.assertIn(needle, banner)
        self.assertLessEqual(len(banner), 260, "the banner must stay compact so the map is not pushed down")
        self.assertNotIn(KEY, banner)
        # the full setup instructions stay one click away, in a collapsed expander on the same page
        self.assertIn("Set up the key / why it was not found", [e.label for e in at.expander])
        for needle in ("secrets.toml", "stays on the server", "Ctrl+C", ".\\run_dashboard.ps1", "Why it was not found"):
            self.assertIn(needle, page)
        commands = " ".join(c.value for c in at.code)
        for needle in ("Read-Host", "-AsSecureString", 'SetEnvironmentVariable("TOMTOM_API_KEY"'):
            self.assertIn(needle, commands)
        self.assertNotIn(KEY, page + commands)
        self.assertIn("Not configured", md(at))
        self.assertTrue(next(b for b in at.button if b.label == "Get traffic for this place").disabled)

    def test_with_a_key_there_is_no_banner_and_the_card_says_it_was_never_tested(self):
        at = self.run_page(key=KEY)
        self.assertEqual([w for w in at.warning if "not configured" in w.value], [])
        text = md(at)
        self.assertIn("Key found, not tested", text)
        self.assertNotIn(KEY, text + alltext(at))

    def test_the_provider_card_reports_what_the_shared_service_really_saw(self):
        svc = tomtom.FlowService(fetcher=mock.Mock(side_effect=tomtom.ProviderError("auth", "x")), ttl=0)
        with self.assertRaises(tomtom.ProviderError):
            svc.get(12.9159, 77.6240, KEY)
        text = md(self.run_page(key=KEY, service=svc))
        self.assertIn("Provider problem", text)
        self.assertIn("TomTom did not accept the key", text)

    def test_the_at_a_glance_strip_has_four_verified_cards(self):
        at = self.run_page()
        text = md(at)
        self.assertIn("At a glance", text)
        for label in ("Live traffic", "Historical data", "Models", "Community reports"):
            self.assertIn(f'<div class="lbl">{label}</div>', text)
        self.assertIn("Latest 2024-08-09", text)
        self.assertIn("Experimental", text)
        self.assertEqual([e.value for e in at.exception], [])

    @needs_outputs
    def test_the_model_note_quotes_the_saved_evaluation_exactly(self):
        m = METRICS["metrics"]
        note = " ".join(md(self.run_page()).split())
        for label, key in (("road mean", "Road-mean baseline"), ("LSTM", "LSTM"), ("XGBoost", "XGBoost")):
            self.assertIn(f"{label} {m[key]['RMSE']:.2f}", note)
        self.assertIn("Neither model beat the road mean.", note)
        self.assertIn("Not validated for 2026", note)

    def test_a_missing_evaluation_file_is_stated_not_invented(self):
        from src.ui import live_map
        with mock.patch.object(live_map, "METRICS_JSON", str(root_dir / "model" / "no_such_metrics.json")):
            at = self.run_page()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertIn("was not found, so no model figures are shown", md(at))

    def test_the_community_card_counts_real_reports_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "c.db"
        self.assertIn("0 saved", md(self.run_page(db=db)))
        store.submit("Accident", "Silk Board Junction", "A real test report with enough text.", session_id="s", path=db)
        self.assertIn("1 saved", md(self.run_page(db=db)))

    def test_the_homepage_shows_no_traffic_values_and_no_forecasts(self):
        at = self.run_page()
        text = alltext(at)
        self.assertNotIn("km/h", text)
        self.assertNotIn("Forecast for", text)

    def test_navigation_links_degrade_to_text_when_the_router_is_absent(self):
        at = self.run_page()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertIn('Open "Traffic Forecasts" from the sidebar', alltext(at))


# ======================================================================================================== sidebar
class SidebarTests(unittest.TestCase):
    def test_the_sidebar_chips_are_the_real_health_items_not_fixed_text(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {store.DB_ENV: str(Path(tmp) / "c.db")}):
            os.environ.pop("TOMTOM_API_KEY", None)
            at = AppTest.from_file(str(root_dir / "app.py"), default_timeout=300).run()
        self.assertEqual([e.value for e in at.exception], [])
        side = " ".join(m.value for m in at.sidebar.markdown)
        for i in health.collect():
            self.assertIn(i.chip.replace("&", "&amp;"), side) if i.key != "community" else None
        self.assertIn("Live traffic: not configured", side)
        self.assertEqual([e.label for e in at.sidebar.expander], ["Diagnostics (files, models, data)"])
        self.assertNotIn("Files / Status", side)

    def test_with_a_key_the_sidebar_says_it_is_untested_and_the_key_is_not_shown(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {store.DB_ENV: str(Path(tmp) / "c.db"), "TOMTOM_API_KEY": KEY}), \
                mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()):
            at = AppTest.from_file(str(root_dir / "app.py"), default_timeout=300).run()
        text = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("Live traffic: key found, not yet tested", text)
        self.assertNotIn(KEY, text)


# ======================================================================================================== forecasts
@needs_data
@needs_models
class ForecastDataGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(os.environ):
            os.environ.pop("TOMTOM_API_KEY", None)
            cls.at = AppTest.from_file(FORECASTS, default_timeout=300).run()
            cls.before = md(cls.at)
            cls.at.button(key="demo_predict").click().run()
        cls.last = fd.load_raw()[fd.DATE_COL].max()
        cls.age = (dt.date.today() - cls.last.date()).days

    def test_the_page_runs_and_states_the_real_data_age(self):
        self.assertEqual([e.value for e in self.at.exception], [])
        flat = " ".join(self.before.split())
        self.assertIn(f"{self.age} days before today", flat)
        self.assertIn("2022-01-01 to 2024-08-09", flat)

    def test_observed_replayed_and_forecast_data_are_three_separate_labelled_things(self):
        flat = " ".join(md(self.at).split())
        self.assertIn("Historical observed traffic", flat)
        self.assertIn(f"Until {self.last:%Y-%m-%d}", flat)
        self.assertIn("Observed, but not live.", flat)
        self.assertIn("Replay and what-if", flat)
        self.assertIn("Neither is a forecast of the future, and both are labelled that way.", flat)
        self.assertIn("Future forecasts", flat)
        self.assertIn("no forecast for recent or future dates is made or invented", flat)

    def test_the_latest_reachable_date_is_the_day_after_the_last_observation(self):
        flat = " ".join(alltext(self.at).split())
        self.assertIn(f"the latest date this dataset can reach is {(self.last + pd.Timedelta(days=1)):%Y-%m-%d}", flat)
        self.assertIn("python -m src.dataProcessing.check_new_data", flat)
        self.assertIn("would also require retraining and re-evaluating the models, which needs separate approval", flat)

    def test_no_date_after_the_dataset_can_be_chosen_or_forecast(self):
        for sb in self.at.selectbox:
            if str(sb.key).startswith("demo_date_"):
                self.assertLessEqual(pd.Timestamp(sb.options[-1].split(" ")[0]), self.last, "the last day a replay can start from is the last observed day")
        res = self.at.session_state["demo_result"]
        self.assertLessEqual(res["target_date"], self.last + pd.Timedelta(days=1), "a forecast reaches at most one day after the last observation")

    def test_a_result_states_when_it_was_generated_and_the_latest_data_date(self):
        res = self.at.session_state["demo_result"]
        self.assertAlmostEqual(res["generated_at"], time.time(), delta=300)
        caption = " ".join(" ".join(c.value for c in self.at.caption).split())
        self.assertIn(f"latest data in the dataset: {self.last:%Y-%m-%d}", caption)
        self.assertIn("(this computer's clock)", caption)
        self.assertIn("horizon: 1 day ahead (next calendar day)", caption)
        self.assertIn("Input source: **Historical replay**", caption)

    def test_the_page_never_claims_to_reach_2026_or_to_be_live(self):
        flat = " ".join(alltext(self.at).split())
        self.assertIn("not live traffic and not validated 2026 forecasts", flat)
        self.assertNotIn("live forecast", flat.lower())
        self.assertNotRegex(flat, r"Forecast for [^.]*202[5-6]")


# ======================================================================================================== Model Insights
@needs_outputs
@needs_data
class ModelInsightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = AppTest.from_file(PERFORMANCE, default_timeout=300).run()
        cls.text = " ".join(md(cls.at).split())

    def test_every_figure_comes_from_the_saved_evaluation(self):
        self.assertEqual([e.value for e in self.at.exception], [])
        t, m = self.text, METRICS["metrics"]
        for name in ("LSTM", "XGBoost", "Persistence baseline", "Road-mean baseline"):
            self.assertIn(f"RMSE {m[name]['RMSE']:.2f}", t)
        self.assertIn(f"{METRICS['n_test_samples']} samples", t)
        self.assertIn(f"{METRICS['test_period'][0]} to {METRICS['test_period'][1]}", t)

    def test_the_provenance_block_documents_dataset_split_and_comparison(self):
        t = self.text
        for needle in ("How these numbers were produced", "Kaggle `preethamgouda/banglore-city-traffic-dataset`", "8,936 rows", "2022-01-01 to 2024-08-09",
                       "one chronological split by target date", "held-out test set", "the same rows for every model",
                       "road-mean baseline (each road's training-period average) and persistence (tomorrow = today)"):
            self.assertIn(needle, t)
        self.assertIn("provenance and licence are unverified", t)

    def test_training_and_held_out_numbers_are_distinguished(self):
        self.assertIn("Training versus held-out", self.text)
        self.assertIn("They are not performance; only held-out test figures count.", self.text)

    def test_the_page_says_historical_results_do_not_prove_live_accuracy(self):
        self.assertIn("It does **not** show accuracy on live traffic or on 2025-2026.", self.text)

    def test_the_metric_explanations_exist_and_use_the_real_baseline_mae(self):
        self.assertIn("What do MAE, RMSE and R² mean?", [e.label for e in self.at.expander])
        t = self.text
        mae = METRICS["metrics"]["Road-mean baseline"]["MAE"]
        for needle in ("**MAE** (mean absolute error)", "**RMSE** (root mean squared error)", "**R²**", "0-100 scale",
                       f"For the road-mean baseline it is {mae:.1f}"):
            self.assertIn(needle, t)
        self.assertIn("beats the simple baselines on the **same** held-out rows", t)

    def test_the_experimental_finding_and_the_corrected_correlation_note_remain(self):
        flat = " ".join(alltext(self.at).split())
        self.assertIn("Experimental models. Honest finding: neither learned model beats the simple per-road historical-mean baseline", flat)
        self.assertIn("approximately 0 (0.003 in the Stage 5A benchmark)", flat)


# ======================================================================================================== read-only validator
COLS = ["Date", "Road/Intersection Name", "Weather Conditions", "Congestion Level", "Average Speed", "Traffic Volume",
        "Travel Time Index", "Road Capacity Utilization", "Incident Reports"]


def write_csv(path, start, days, roads=("Road A", "Road B"), **over):
    rows = []
    for d in range(days):
        for r in roads:
            row = {"Date": (pd.Timestamp(start) + pd.Timedelta(days=d)).strftime("%Y-%m-%d"), "Road/Intersection Name": r, "Weather Conditions": "Clear",
                   "Congestion Level": 60.0 + d, "Average Speed": 30.0, "Traffic Volume": 20000, "Travel Time Index": 1.3,
                   "Road Capacity Utilization": 80.0, "Incident Reports": 1}
            row.update(over)
            rows.append(row)
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return Path(path)


class NewDataValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.current = write_csv(self.dir / "current.csv", "2024-01-01", 10)  # the stand-in for the project's dataset: ends 2024-01-10

    def check(self, name, start, days, target=None, **kw):
        return cnd.check(write_csv(self.dir / name, start, days, **kw), target=target, existing_path=self.current)

    def test_a_valid_continuation_extends_the_dates_and_reports_what_it_reaches(self):
        r = self.check("new.csv", "2024-01-11", 5, target="2024-01-15")
        self.assertEqual((r["rows"], r["roads"], r["days_after_current_last"], r["overlap_rows"]), (10, 2, 5, 0))
        self.assertTrue(r["reaches_target"])
        self.assertEqual(r["notes"], [])

    def test_a_file_that_ends_before_the_target_does_not_reach_it(self):
        r = self.check("new.csv", "2024-01-11", 5, target="2024-02-01")
        self.assertFalse(r["reaches_target"])
        self.assertTrue(any("ends 17 day(s) before the target date 2024-02-01" in n for n in r["notes"]))

    def test_no_newer_data_is_reported_as_such(self):
        r = self.check("same.csv", "2024-01-01", 10, target="2026-10-28")
        self.assertEqual((r["days_after_current_last"], r["reaches_target"], r["overlap_rows"]), (0, False, 20))
        text = " ".join(r["notes"])
        self.assertIn("contains nothing after the current last observation", text)
        self.assertIn("20 rows repeat a date and road", text)

    def test_a_gap_unknown_roads_and_overlap_are_flagged(self):
        r = self.check("gap.csv", "2024-01-20", 3, roads=("Road A", "Road Z"), target="2024-01-22")
        text = " ".join(r["notes"])
        self.assertIn("gap of 9 day(s)", text)
        self.assertIn("Road Z", text)
        self.assertTrue(r["unknown_roads"] == ["Road Z"])

    def test_invalid_files_are_rejected_with_the_projects_own_rules(self):
        cases = {"missing column": lambda p: pd.read_csv(write_csv(p, "2024-01-11", 3)).drop(columns=["Traffic Volume"]).to_csv(p, index=False),
                 "bad date": lambda p: write_csv(p, "2024-01-11", 3, Date="not-a-date"),
                 "non-numeric": lambda p: write_csv(p, "2024-01-11", 3, **{"Average Speed": "fast"}),
                 "congestion over 100": lambda p: write_csv(p, "2024-01-11", 3, **{"Congestion Level": 140.0}),
                 "unknown weather": lambda p: write_csv(p, "2024-01-11", 3, **{"Weather Conditions": "Meteor shower"}),
                 "empty": lambda p: Path(p).write_text(""), "header only": lambda p: pd.DataFrame(columns=COLS).to_csv(p, index=False),
                 "not a csv": lambda p: Path(p).write_bytes(b"\x00\x01\x02" * 100)}
        for name, make in cases.items():
            with self.subTest(name):
                p = self.dir / f"{name.replace(' ', '_')}.csv"
                make(p)
                with self.assertRaises(fd.DatasetError):
                    cnd.check(p, existing_path=self.current)
        dup = self.dir / "dup.csv"
        pd.concat([pd.read_csv(write_csv(dup, "2024-01-11", 2)), pd.read_csv(dup)]).to_csv(dup, index=False)
        with self.assertRaises(fd.DatasetError):
            cnd.check(dup, existing_path=self.current)
        miss = write_csv(self.dir / "miss.csv", "2024-01-11", 3)
        frame = pd.read_csv(miss)
        frame.loc[0, "Congestion Level"] = None
        frame.to_csv(miss, index=False)
        with self.assertRaises(fd.DatasetError):
            cnd.check(miss, existing_path=self.current)

    def test_the_cli_exit_codes_and_messages(self):
        good = write_csv(self.dir / "good.csv", "2024-01-11", 5)
        out = StringIO()
        with redirect_stdout(out), mock.patch.object(fd, "DATA_PATH", str(self.current)):
            code = cnd.main([str(good), "--target", "2024-01-15"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Valid file: 10 rows, 2 roads, 2024-01-11 to 2024-01-15.", text)
        self.assertIn("reaches 2024-01-15: yes", text)
        self.assertIn("Nothing was changed.", text)
        self.assertIn("needs approval", text)
        bad = write_csv(self.dir / "bad.csv", "2024-01-11", 3, Date="nope")
        out = StringIO()
        with redirect_stdout(out), mock.patch.object(fd, "DATA_PATH", str(self.current)):
            self.assertEqual(cnd.main([str(bad)]), 1)
        self.assertIn("NOT VALID", out.getvalue())
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(cnd.main([str(self.dir / "absent.csv")]), 1)

    def test_the_validator_is_read_only(self):
        new = write_csv(self.dir / "new.csv", "2024-01-11", 5)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dir.iterdir()}
        cnd.check(new, existing_path=self.current)
        with redirect_stdout(StringIO()), mock.patch.object(fd, "DATA_PATH", str(self.current)):
            cnd.main([str(new)])
        self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dir.iterdir()}, before, "no file may be created or changed")

    def test_the_module_cannot_write_merge_train_or_use_the_network(self):
        src = (root_dir / "src" / "dataProcessing" / "check_new_data.py").read_text(encoding="utf-8")
        for banned in (".to_csv(", "write_text", "write_bytes", "open(", "shutil", "requests", "urllib", "socket", ".fit(", "save_model", "concat("):
            self.assertNotIn(banned, src, banned)

    @needs_data
    def test_the_real_dataset_has_nothing_newer_and_does_not_reach_yesterday(self):
        r = cnd.check(fd.DATA_PATH, target=dt.date.today() - dt.timedelta(days=1))
        self.assertEqual(r["days_after_current_last"], 0)
        self.assertFalse(r["reaches_target"])
        self.assertEqual(str(r["current_last"].date()), "2024-08-09")

    @needs_models
    @needs_data
    def test_running_the_validator_leaves_model_files_byte_identical(self):
        files = [root_dir / "model" / n for n in ("model.h5", "scaler.pkl", "xgboost_model.json", "metrics.json")]
        before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
        cnd.check(fd.DATA_PATH)
        self.assertEqual([hashlib.sha256(f.read_bytes()).hexdigest() for f in files], before)


if __name__ == "__main__":
    unittest.main()

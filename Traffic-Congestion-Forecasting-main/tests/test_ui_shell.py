"""
Stage 0 (UI shell) tests: navigation definition, every page loads, honest labelling, nothing invented, and the existing
functionality is still reachable under the new sections.

Each page script is run on its own with Streamlit's AppTest (in Streamlit 1.40 the test harness does not render the
page selected through st.navigation); the router (app.py) is tested for launch, sidebar and status; real page switching
is checked in a real browser by tests/browser_check.py.
Run:   python -m unittest discover -s tests -v
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _guards import needs_all, needs_data  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ui import navigation, theme  # noqa: E402

EXPECTED_TITLES = ["Live Traffic Map", "Traffic Forecasts", "Community Reports", "Road & Incident Details",
                   "Model Insights"]
KEY = "TOMTOM_API_KEY"


def all_text(at) -> list:
    """Every visible string of a page, one entry per element (so context checks stay local)."""
    out = []
    for group in (at.markdown, at.caption, at.subheader, at.header, at.title, at.text, at.info, at.warning, at.error, at.success):
        out += [e.value for e in group]
    out += [f"{m.label} {m.value}" for m in at.metric]
    out += [x.label for x in at.expander] + [t.label for t in at.tabs] + [b.label for b in at.button]
    return out


def open_page(title: str, env_key=None):
    """Run the script of the page with this title (optionally with a provider key in the environment)."""
    page = next(p for p in navigation.PAGES if p.title == title)
    with mock.patch.dict(os.environ):
        os.environ.pop(KEY, None)
        if env_key:
            os.environ[KEY] = env_key
        return AppTest.from_file(str(root_dir / page.path), default_timeout=300).run()


class NavigationDefinitionTests(unittest.TestCase):
    def test_exactly_the_five_approved_sections_in_order(self):
        self.assertEqual([p.title for p in navigation.PAGES], EXPECTED_TITLES)

    def test_one_default_landing_page_and_unique_urls(self):
        self.assertEqual([p.title for p in navigation.PAGES if p.default], ["Live Traffic Map"])
        urls = [p.url_path for p in navigation.PAGES]
        self.assertEqual(len(urls), len(set(urls)))

    def test_every_page_script_exists_and_has_an_icon(self):
        for p in navigation.PAGES:
            self.assertTrue((root_dir / p.path).is_file(), p.path)
            self.assertTrue(p.icon.startswith(":material/"), p.icon)

    def test_app_py_is_only_a_router_and_the_launch_command_is_unchanged(self):
        src = (root_dir / "app.py").read_text(encoding="utf-8")
        self.assertIn("st.navigation", src)
        self.assertLess(len(src.splitlines()), 80)  # the old 800-line single page was split into src/ui/
        ps1 = (root_dir / "run_dashboard.ps1").read_text(encoding="utf-8")
        self.assertIn("'streamlit', 'run', 'app.py'", ps1)


class ThemeTests(unittest.TestCase):
    def test_dark_navy_theme_is_configured(self):
        import tomllib
        cfg = tomllib.loads((root_dir / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["theme"]["base"], "dark")
        self.assertEqual(cfg["theme"]["backgroundColor"].lower(), theme.NAVY.lower())
        self.assertEqual(cfg["client"]["toolbarMode"], "viewer")  # earlier setting preserved
        self.assertIs(cfg["browser"]["gatherUsageStats"], False)

    def test_status_colours_are_green_amber_red(self):
        self.assertEqual((theme.GREEN, theme.AMBER, theme.RED), ("#22C55E", "#F59E0B", "#EF4444"))

    def test_components_escape_html(self):
        html = theme.card_html("<b>label</b>", "<script>alert(1)</script>", "<img src=x onerror=1>", "bad")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("tc-bad", html)
        self.assertIn("tc-info", theme.chip_html("x", "not-a-kind"))  # unknown kinds fall back safely


class ShellPagesTests(unittest.TestCase):
    """Pages that must work with NOTHING configured (no TomTom key) and must not invent anything."""

    @classmethod
    def setUpClass(cls):
        cls.pages = {t: open_page(t) for t in ("Live Traffic Map", "Community Reports")}

    def test_missing_optional_key_does_not_break_any_page(self):
        for title, at in self.pages.items():
            self.assertEqual([e.value for e in at.exception], [], title)
            self.assertEqual([e.value for e in at.error], [], title)

    def test_live_map_says_live_traffic_is_not_configured_and_shows_no_data(self):
        at = self.pages["Live Traffic Map"]
        text = " ".join(all_text(at))
        self.assertIn("Live traffic is not configured", text)
        self.assertIn("Live traffic not configured", text)
        self.assertIn("No reading yet", text)
        # nothing invented: no native charts/tables/metrics, no readings, no coordinates (the basemap is an embedded Leaflet iframe)
        for kind in ("plotly_chart", "deck_gl_json_chart", "arrow_data_frame", "arrow_table", "map"):
            self.assertEqual(len(at.get(kind)), 0, f"unexpected {kind} on the live map")
        self.assertEqual(len(at.metric), 0)
        self.assertNotIn("km/h", text)
        self.assertIsNone(re.search(r"\d{1,2}\.\d{3,}\s*,\s*\d{2,3}\.\d{3,}", text), "coordinates must not appear")

    def test_community_page_does_not_pretend_to_work(self):
        at = self.pages["Community Reports"]
        text = " ".join(all_text(at))
        self.assertIn("Not implemented yet", text)
        self.assertIn("not available yet", text)
        self.assertIn("No sample reports are shown", text)
        for kind in ("file_uploader", "camera_input", "arrow_data_frame", "plotly_chart"):
            self.assertEqual(len(at.get(kind)), 0, f"unexpected {kind}: uploads/reports must not exist yet")
        self.assertEqual(len(at.button), 0)

    def test_stage0_creates_no_database_and_no_upload_folder(self):
        for pattern in ("**/*.db", "**/*.sqlite", "**/*.sqlite3"):
            self.assertEqual([p for p in root_dir.glob(pattern) if ".venv" not in str(p)], [])
        self.assertFalse((root_dir / "data" / "community").exists())
        self.assertFalse((root_dir / "data" / "uploads").exists())

    def test_provider_key_is_never_displayed(self):
        at = open_page("Live Traffic Map", env_key="FAKE-SECRET-KEY-9876")
        self.assertEqual([e.value for e in at.exception], [])
        blob = " ".join(all_text(at))
        self.assertNotIn("FAKE-SECRET-KEY-9876", blob)
        self.assertIn("Traffic provider key found", blob)  # honest: a key is not live data
        self.assertIn("No reading yet", blob)
        self.assertNotIn("Live traffic not configured", blob)
        # the router's sidebar status must not leak the key either
        with mock.patch.dict(os.environ, {KEY: "FAKE-SECRET-KEY-9876"}):
            router = AppTest.from_file(str(root_dir / "app.py"), default_timeout=300).run()
        side = " ".join(m.value for m in router.sidebar.markdown)
        self.assertNotIn("FAKE-SECRET-KEY-9876", side)
        self.assertIn("Provider key found, not yet verified", side)


class RouterTests(unittest.TestCase):
    def test_router_launches_without_a_key_and_shows_status_in_the_sidebar(self):
        env = {k: v for k, v in os.environ.items() if k != KEY}
        with mock.patch.dict(os.environ, env, clear=True):
            at = AppTest.from_file(str(root_dir / "app.py"), default_timeout=300).run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])
        side = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("Live traffic: not configured", side)
        self.assertIn("Data: historical, 2022-2024", side)
        self.assertIn("Models: not validated for 2026", side)
        self.assertEqual([e.label for e in at.sidebar.expander], ["Files / Status"])  # Plot Controls now appear only on the pages that use them


@needs_all
class DataPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = {t: open_page(t) for t in ("Traffic Forecasts", "Road & Incident Details", "Model Insights")}

    def test_pages_load_without_exceptions_or_error_boxes(self):
        for title, at in self.pages.items():
            self.assertEqual([e.value for e in at.exception], [], title)
            self.assertEqual([e.value for e in at.error], [], title)

    def test_forecasts_page_is_labelled_historical_and_not_validated_for_2026(self):
        text = " ".join(all_text(self.pages["Traffic Forecasts"]))
        for needle in ("Historical data 2022-2024", "Not validated for 2026", "not live traffic", "Baselines shown beside the models",
                       "2024-03-21 to 2024-08-09", "Historical replay and what-if forecasts"):
            self.assertIn(needle, text)

    def test_existing_prediction_flow_still_works_from_the_new_page(self):
        at = open_page("Traffic Forecasts")
        at.button(key="demo_predict").click().run()
        self.assertEqual([e.value for e in at.exception], [])
        res = at.session_state["demo_result"]
        self.assertEqual(res["source"], "Historical replay")
        self.assertTrue(all(0 <= res[k] <= 100 for k in ("XGBoost", "LSTM", "Persistence baseline", "Road-mean baseline")))
        text = " ".join(all_text(at))
        self.assertIn("Input source: Historical replay", text.replace("**", ""))
        self.assertIn("Persistence baseline", text)  # baseline stays visible next to the models

    def test_details_page_live_values_are_unavailable_and_history_is_real(self):
        at = self.pages["Road & Incident Details"]
        text = " ".join(all_text(at))
        self.assertIn("will be added in a later stage", text)
        self.assertEqual(text.count('<div class="val">Not available</div>'), 4)  # the four live-status cards: speed, travel time, closure, incidents
        import pandas as pd
        from src.dataProcessing import forecast_data as fd
        df = fd.load_raw()
        road = at.selectbox(key="details_road").value
        expected = f"{df[df[fd.ROAD_COL] == road][fd.TARGET_COL].mean():.1f}"
        self.assertIn(expected, text)  # the historical number comes from the real file, not invented
        self.assertIn("not live conditions", text)

    def test_performance_page_keeps_the_existing_evaluation(self):
        import json
        at = self.pages["Model Insights"]
        labels = [t.label for t in at.tabs]
        for tab in ("Overview", "Model Results", "Data & Wrangling", "Historical Analysis", "Limitations & Sources"):
            self.assertIn(tab, labels)
        m = json.loads((root_dir / "model" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        text = " ".join(all_text(at))
        for name in ("LSTM", "XGBoost", "Persistence baseline", "Road-mean baseline"):
            self.assertIn(f"RMSE {m[name]['RMSE']:.2f}", text)
            self.assertIn(f"R² {m[name]['R2']:.3f}", text)
        self.assertIn("No winner between LSTM and XGBoost", text)  # existing statistical honesty preserved
        self.assertIn("Neither model beats the simple", text)
        self.assertIn("Not validated for 2026", text)

    def test_no_page_presents_historical_or_demo_data_as_live(self):
        """Every use of the word 'live' must be a label, a negation or a 'planned / later / needs' statement."""
        allowed_labels = ["Live Traffic Map", "Live provider", "Live observations", "Live status", "Live details", "Live traffic: not configured",
                          "Live traffic not configured", "Live traffic is not configured yet"]
        safe = ("not ", "no ", "not ", "never", "later", "planned", "until", "once", "needs", "need ", "cannot", "unverified",
                "unavailable", "does not", "none", "key found", "only")
        for title, at in self.pages.items():
            for chunk in all_text(at):
                clean = chunk
                for lab in allowed_labels:
                    clean = clean.replace(lab, "")
                for m in re.finditer(r"\blive\b", clean, flags=re.I):
                    ctx = re.sub(r"[*_`]", "", clean[max(0, m.start() - 160): m.end() + 160]).lower()  # ignore markdown emphasis
                    self.assertTrue(any(s in ctx for s in safe), f"{title}: possibly mislabelled as live: ...{ctx}...")
        for title, at in self.pages.items():  # and never a forecast called live / real-time
            blob = " ".join(all_text(at)).lower()
            for bad in ("live forecast", "live prediction", "real-time forecast", "live demo"):
                self.assertNotIn(bad, blob, f"{title}: {bad!r}")


if __name__ == "__main__":
    unittest.main()

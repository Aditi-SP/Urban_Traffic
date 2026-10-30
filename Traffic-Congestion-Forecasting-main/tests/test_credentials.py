"""
Tests for the TomTom key configuration (src/live/credentials.py), the provider's error classification, the safe self-test
(src/live/check_key.py) and the launcher's handling of saved Windows settings.

Only FAKE keys are used and every provider answer is MOCKED: nothing here contacts TomTom, and nothing here says anything about whether
a real key works. The real Windows environment is never changed (tests that need it patch the lookup).

Run:   python -m unittest tests.test_credentials -v
"""
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database: tests never touch the real community.db)
os.environ.pop("LIVE_CACHE_TTL_S", None)

import requests  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import preflight  # noqa: E402
from src.live import check_key, credentials, geo, tomtom  # noqa: E402
from src.ui import common, health, live_map  # noqa: E402
from tests.test_live_map import CONSTRUCTED_404, REC_BAD_KEY, REC_BAD_POINT, REC_NO_POINT, REC_NO_ROAD, KEY, FakeResp, PageHarness, flow_payload  # noqa: E402

LAUNCHER = root_dir / "run_dashboard.ps1"
NOTHING_SAVED = (None, None)
COMPACT_BANNER_CHARS = 260   # the banner was 688 characters (528 px tall at 800 px wide) and pushed the map off-screen; now 122 to 208


def creds(env=None, secrets=None, home_secrets=None):
    """credentials.load() on a throw-away project folder and home folder, with an explicit environment (the real one is not read)."""
    with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
        for base, text in ((proj, secrets), (home, home_secrets)):
            if text is not None:
                (Path(base) / ".streamlit").mkdir()
                (Path(base) / ".streamlit" / "secrets.toml").write_text(text, encoding="utf-8")
        return credentials.load(project_root=proj, home=home, environ=env or {})


# ======================================================================================================== value handling
class NormaliseTests(unittest.TestCase):
    def test_usable_values(self):
        self.assertEqual(credentials.normalise("abc123"), ("abc123", None))
        self.assertEqual(credentials.normalise("  abc123 \r\n"), ("abc123", None))
        self.assertEqual(credentials.normalise('"abc123"'), ("abc123", None))
        self.assertEqual(credentials.normalise("'abc123'"), ("abc123", None))
        self.assertEqual(credentials.normalise('  " abc123 "  '), ("abc123", None))

    def test_unusable_values_say_why(self):
        for bad, why in ((None, None), ("", "empty"), ("   ", "empty"), ('""', "empty"), ("abc def", "spaces"), ("abc\tdef", "spaces"),
                         ("abéc", "cannot be part of a key"), (123, "not text"), ("ab\x00c", "cannot be part of a key")):
            with self.subTest(repr(bad)):
                key, reason = credentials.normalise(bad)
                self.assertIsNone(key)
                if why:
                    self.assertIn(why, reason)


# ======================================================================================================== where the key is found
class LoadTests(unittest.TestCase):
    def test_the_environment_variable_is_used_and_the_value_is_never_in_any_report(self):
        c = creds({"TOMTOM_API_KEY": KEY})
        self.assertTrue(c.configured)
        self.assertEqual((c.key, c.source), (KEY, "environment variable"))
        shown = " ".join([str(c), repr(c), *c.report_lines(), c.hint_text(), *map(str, c.problems), *map(str, c.hints)])
        self.assertNotIn(KEY, shown)
        self.assertNotIn(str(len(KEY)), c.report_lines()[0], "not even the length")
        self.assertIn("configured (source: environment variable)", c.report_lines()[0])

    def test_quotes_and_spaces_around_the_value_are_ignored_and_reported(self):
        c = creds({"TOMTOM_API_KEY": f'  "{KEY}" '})
        self.assertEqual(c.key, KEY)
        self.assertIn("surrounding spaces or quotes", c.hint_text())
        self.assertNotIn(KEY, c.hint_text())

    def test_an_unusable_environment_value_is_a_reported_problem_not_a_silent_miss(self):
        for value, why in (("", "empty"), ("   ", "empty"), ("two words", "spaces")):
            with self.subTest(value):
                c = creds({"TOMTOM_API_KEY": value})
                self.assertFalse(c.configured)
                self.assertIn("exists but cannot be used", c.hint_text())
                self.assertIn(why, c.hint_text())

    def test_a_secrets_file_is_used_when_the_environment_has_nothing(self):
        c = creds({}, secrets=f'TOMTOM_API_KEY = "{KEY}"\n')
        self.assertEqual((c.key, c.source), (KEY, "secrets file"))
        self.assertNotIn(KEY, " ".join(c.report_lines()))
        self.assertIn("project secrets file .streamlit/secrets.toml: file found", " ".join(c.report_lines()))

    def test_the_environment_wins_over_the_secrets_file(self):
        c = creds({"TOMTOM_API_KEY": "from-env"}, secrets='TOMTOM_API_KEY = "from-file"\n')
        self.assertEqual((c.key, c.source), ("from-env", "environment variable"))

    def test_the_home_secrets_file_is_the_last_resort(self):
        self.assertEqual(creds({}, home_secrets='TOMTOM_API_KEY = "from-home"\n').key, "from-home")
        self.assertEqual(creds({}, secrets='TOMTOM_API_KEY = "proj"\n', home_secrets='TOMTOM_API_KEY = "home"\n').key, "proj")

    def test_a_broken_secrets_file_is_explained_and_its_content_is_not_echoed(self):
        c = creds({}, secrets=f'TOMTOM_API_KEY: {KEY}\n[oops\n')
        self.assertFalse(c.configured)
        self.assertIn("could not be read", c.hint_text())
        self.assertIn('TOMTOM_API_KEY = "your-key"', c.hint_text())
        self.assertNotIn(KEY, c.hint_text() + " ".join(c.report_lines()))

    def test_a_secrets_file_without_the_entry_names_the_similar_entries_but_not_their_values(self):
        c = creds({}, secrets='TOMTOM_KEY = "SECRETVALUE123"\nother = 1\n')
        self.assertFalse(c.configured)
        self.assertIn("has no TOMTOM_API_KEY entry (it has: TOMTOM_KEY)", c.hint_text())
        self.assertNotIn("SECRETVALUE123", c.hint_text())

    def test_a_secrets_value_that_is_not_text_or_is_empty_is_reported(self):
        self.assertIn("is not text", creds({}, secrets="TOMTOM_API_KEY = 12345\n").hint_text())
        self.assertIn("is empty", creds({}, secrets='TOMTOM_API_KEY = ""\n').hint_text())

    def test_similarly_named_variables_are_listed_by_name_only(self):
        c = creds({"TOMTOM_KEY": "SECRETVALUE1", "TOM_TOM_API_KEY": "SECRETVALUE2", "PATH": "x"})
        self.assertEqual(c.similar_names, ["TOMTOM_KEY", "TOM_TOM_API_KEY"])
        text = " ".join(c.report_lines())
        self.assertIn("similar name exist (TOMTOM_KEY, TOM_TOM_API_KEY)", text)
        self.assertNotIn("SECRETVALUE", text)
        self.assertFalse(c.configured)

    def test_a_key_saved_in_windows_but_missing_from_this_process_gets_the_exact_fix(self):
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=("user", "WINDOWSSECRET")):
            os.environ.pop("TOMTOM_API_KEY", None)
            c = credentials.load()
        self.assertFalse(c.configured)
        text = c.hint_text()
        for needle in ("saved in your Windows user environment", "does not have it", "VS Code", ".\\run_dashboard.ps1"):
            self.assertIn(needle, text)
        self.assertNotIn("WINDOWSSECRET", text + " ".join(c.report_lines()))

    def test_no_windows_hint_when_the_process_has_the_key(self):
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}), mock.patch.object(credentials, "_windows_lookup", return_value=("user", "x")):
            self.assertEqual(credentials.load().hints, [])

    def test_nothing_anywhere_says_so_plainly(self):
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=NOTHING_SAVED):
            os.environ.pop("TOMTOM_API_KEY", None)
            c = credentials.load(project_root=tempfile.gettempdir() + "/none_here", home=tempfile.gettempdir() + "/none_here")
        self.assertIn("No TOMTOM_API_KEY was found in this process", c.hint_text())

    def test_the_windows_lookup_never_returns_a_secret_value(self):
        with self.assertRaises(ValueError):
            credentials.windows_env_value("TOMTOM_API_KEY")
        self.assertIn(credentials.windows_env_scope("TC_NO_SUCH_VARIABLE_XYZ"), (None,))

    def test_a_setting_that_differs_from_the_saved_windows_value_is_flagged(self):
        with mock.patch.object(credentials, "windows_env_value", side_effect=lambda n: "0" if n == "LIVE_CACHE_TTL_S" else None):
            lines = credentials.settings_lines({"LIVE_CACHE_TTL_S": 300, "LIVE_MAX_REQUESTS": 300})
            self.assertIn("WARNING", lines[0])
            self.assertIn("saved Windows value is 0", lines[0])
            self.assertNotIn("WARNING", lines[1])
            self.assertNotIn("WARNING", credentials.settings_lines({"LIVE_CACHE_TTL_S": 0, "LIVE_MAX_REQUESTS": 300})[0])

    def test_the_module_never_prints_or_logs(self):
        src = (root_dir / "src" / "live" / "credentials.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"\bprint\(|logging|logger")


class ProcessStartTimeTests(unittest.TestCase):
    """D5: Diagnostics used to show the time the credentials MODULE was imported as the "process start". It is now the operating system's
    creation time of the process, fixed for the life of the process."""

    CHILD = """
import sys, time
sys.path.insert(0, {root!r})
time.sleep(3)   # the process exists for 3 s BEFORE it imports the module
from src.live import credentials
imported = time.time()
print(credentials.STARTED_AT, credentials.STARTED_AT_EXACT, imported, credentials.process_start_time()[0])
"""

    def test_the_time_is_the_process_creation_time_not_the_module_import_time(self):
        before = time.time()
        out = subprocess.run([sys.executable, "-I", "-c", self.CHILD.format(root=str(root_dir))], capture_output=True, text=True, timeout=120)
        after = time.time()
        self.assertEqual(out.returncode, 0, out.stderr[-300:])
        started, exact, imported, again = out.stdout.split()
        started, imported, again = float(started), float(imported), float(again)
        self.assertEqual(exact, "True")
        self.assertGreaterEqual(started, before - 1.0, "not before this test launched the process")
        self.assertLessEqual(started, after)
        self.assertGreaterEqual(imported - started, 2.5, "the import happened >= 3 s after the process was created: the old code would report the import time")
        self.assertLess(abs(again - started), 0.01, "asking again returns the same moment")

    def test_the_value_is_fixed_for_the_life_of_the_process(self):
        first = credentials.STARTED_AT
        time.sleep(1.1)
        self.assertEqual(credentials.STARTED_AT, first)
        self.assertLess(abs(credentials.process_start_time()[0] - first), 0.01)   # even a re-import would compute the same moment
        self.assertLessEqual(credentials.STARTED_AT, time.time() - 1.0)

    def test_fallback_is_marked_inexact(self):
        with mock.patch.object(sys, "platform", "unknown-os"):
            ts, exact = credentials.process_start_time()
        self.assertFalse(exact)
        self.assertLessEqual(abs(ts - time.time()), 5)


class ShortHintTests(unittest.TestCase):
    """The one-sentence summary used in the (compact) setup banner; the full reasons stay in hint_text() and Diagnostics."""

    def test_nothing_to_say_when_the_key_is_configured_or_no_reason_is_known(self):
        self.assertEqual(creds({"TOMTOM_API_KEY": KEY}).short_hint(), "")
        self.assertEqual(creds({}).short_hint(), "")

    def test_a_key_saved_in_windows_gets_the_exact_fix(self):
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=("user", "WINSECRET")):
            os.environ.pop("TOMTOM_API_KEY", None)
            c = credentials.load()
        self.assertEqual(c.windows_scope, "user")
        self.assertIn("Saved in Windows but not loaded here", c.short_hint())
        self.assertIn(".\\run_dashboard.ps1", c.short_hint())
        self.assertNotIn("WINSECRET", c.short_hint() + c.hint_text())

    def test_a_problem_or_a_similar_name_is_summarised_in_one_sentence(self):
        self.assertIn("cannot be used", creds({"TOMTOM_API_KEY": "two words"}).short_hint())
        self.assertIn("TOMTOM_KEY", creds({"TOMTOM_KEY": "SECRETVALUE"}).short_hint())
        self.assertNotIn("SECRETVALUE", creds({"TOMTOM_KEY": "SECRETVALUE"}).short_hint())

    def test_the_summary_never_exceeds_the_limit_and_is_a_prefix_of_nothing_secret(self):
        long = creds({}, secrets='TOMTOM_API_KEY: ' + 'x' * 400 + '\n[oops\n')
        hint = long.short_hint()
        self.assertLessEqual(len(hint), 150)
        self.assertNotIn("x" * 20, hint)
        self.assertTrue(hint.endswith("\u2026") or len(hint) < 150)


# ======================================================================================================== wiring
class WiringTests(unittest.TestCase):
    def test_the_dashboards_key_loader_uses_the_same_rules(self):
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": f"  '{KEY}'  "}):
            self.assertEqual(common.tomtom_key(), KEY)
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=NOTHING_SAVED):
            os.environ.pop("TOMTOM_API_KEY", None)
            self.assertIsNone(common.tomtom_key())

    def test_the_status_item_carries_the_reason_the_key_is_missing(self):
        i = health.provider_item(False, health.STATUS0 if hasattr(health, "STATUS0") else
                                 {"requests_made": 0, "budget": 300, "ttl_s": 0, "cooldown_remaining_s": 0.0, "last_success_at": None, "last_error": None},
                                 0.0, hint="Environment variables with a similar name exist (TOMTOM_KEY).")
        self.assertIn("TOMTOM_KEY", i.summary)
        self.assertEqual(i.state, "bad")

    def page(self, env, lookup=NOTHING_SAVED):
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=lookup), \
                mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()):
            os.environ.pop("TOMTOM_API_KEY", None)
            os.environ.update(env)
            return AppTest.from_file(str(root_dir / "src" / "ui" / "pages" / "live_map.py"), default_timeout=120).run()

    def test_the_banner_explains_a_misspelled_variable_without_showing_its_value(self):
        at = self.page({"TOMTOM_KEY": "SECRETVALUE777"})
        banner, page = at.warning[0].value, " ".join(m.value for m in at.markdown)
        self.assertIn("Live traffic is not configured.", banner)
        self.assertIn("TOMTOM_KEY", banner)               # the one-sentence reason is in the banner ...
        self.assertLessEqual(len(banner), COMPACT_BANNER_CHARS, "the banner must stay compact so the map is not pushed down")
        self.assertIn("Why it was not found", page)       # ... and the full explanation is on the page, in the collapsed expander
        self.assertIn("TOMTOM_KEY", page)
        self.assertNotIn("SECRETVALUE777", page + banner)

    def test_the_banner_tells_a_stale_terminal_how_to_fix_it(self):
        at = self.page({}, lookup=("user", "WINSECRET"))
        banner, page = at.warning[0].value, " ".join(m.value for m in at.markdown)
        for needle in ("Saved in Windows", ".\\run_dashboard.ps1"):       # the most useful fix is right in the banner
            self.assertIn(needle, banner)
        self.assertLessEqual(len(banner), COMPACT_BANNER_CHARS)
        for needle in ("Why it was not found", "Windows user environment", "VS Code", ".\\run_dashboard.ps1"):  # the full reasons are one click away
            self.assertIn(needle, page)
        self.assertNotIn("WINSECRET", banner + page)

    def test_a_valid_environment_key_removes_the_banner_and_is_never_shown(self):
        at = self.page({"TOMTOM_API_KEY": KEY})
        self.assertEqual([w for w in at.warning if "not configured" in w.value], [])
        self.assertNotIn(KEY, " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption))

    def test_diagnostics_show_the_key_report_the_process_start_and_the_settings_without_the_value(self):
        src = f"import sys\nsys.path.insert(0, {str(root_dir)!r})\nfrom src.ui import diagnostics\ndiagnostics.render()\n"
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}), mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()):
            at = AppTest.from_string(src, default_timeout=120).run()
        text = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
        self.assertEqual([e.value for e in at.exception], [])
        for needle in ("TomTom key configuration", "TomTom key: configured (source: environment variable)", "This dashboard process started at",
                       "LIVE_CACHE_TTL_S in use by this dashboard", "LIVE_MAX_REQUESTS in use by this dashboard"):
            self.assertIn(needle, text)
        self.assertNotIn(KEY, text)

    def test_the_two_diagnostics_buttons_show_what_they_are_doing(self):
        from tests.test_live_map import RecordedUi
        src = f"import sys\nsys.path.insert(0, {str(root_dir)!r})\nfrom src.ui import diagnostics\ndiagnostics.render_tools()\n"
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}), mock.patch.object(common, "load_models", return_value=(None, False, [])), \
                mock.patch.object(preflight, "run_checks", return_value=[]), RecordedUi() as ui:
            at = AppTest.from_string(src, default_timeout=120).run()
            at.button(key="diag_models").click().run()
            at.button(key="diag_preflight").click().run()
        self.assertIn("Loading the trained models (the first time can take 10 to 20 seconds)...", ui.spinners)
        self.assertIn("Running the preflight checks...", ui.spinners)
        self.assertEqual([e.value for e in at.exception], [])

    def test_diagnostics_show_the_exact_process_start_time_not_a_later_one(self):
        src = f"import sys\nsys.path.insert(0, {str(root_dir)!r})\nfrom src.ui import diagnostics\ndiagnostics.render()\n"
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}), mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()):
            at = AppTest.from_string(src, default_timeout=120).run()
        text = " ".join(c.value for c in at.caption)
        self.assertTrue(credentials.STARTED_AT_EXACT, "this platform must report the real process creation time")
        clock = time.strftime("%H:%M:%S", time.localtime(credentials.STARTED_AT))
        self.assertIn(f"This dashboard process started at {clock} and serves port", text)

    def test_diagnostics_do_not_claim_a_start_time_when_the_os_cannot_give_one(self):
        src = f"import sys\nsys.path.insert(0, {str(root_dir)!r})\nfrom src.ui import diagnostics\ndiagnostics.render()\n"
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}), mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()),                 mock.patch.object(credentials, "STARTED_AT_EXACT", False):
            at = AppTest.from_string(src, default_timeout=120).run()
        text = " ".join(c.value for c in at.caption)
        self.assertIn("This dashboard process loaded this code at", text)
        self.assertNotIn("process started at", text)

    def test_preflight_uses_the_same_rules_and_never_prints_the_key(self):
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": KEY}):
            ok = preflight._tomtom_checks(root_dir)[0]
        self.assertEqual(ok.status, "OK")
        self.assertIn("check_key --request", ok.detail)
        self.assertNotIn(KEY, ok.detail)
        with mock.patch.dict(os.environ, {"TOMTOM_API_KEY": "two words"}):
            bad = preflight._tomtom_checks(root_dir)[0]
        self.assertEqual(bad.status, "OPT")
        self.assertIn("cannot be used", bad.detail)


# ======================================================================================================== provider errors
def fetch(resp=None, exc=None, key=KEY):
    get = mock.Mock(side_effect=exc) if exc else mock.Mock(return_value=resp)
    return tomtom.fetch_flow(12.9159, 77.6240, key, http_get=get)


class ProviderErrorClassificationTests(unittest.TestCase):
    def expect(self, resp=None, exc=None):
        with self.assertRaises(tomtom.ProviderError) as cm:
            fetch(resp, exc)
        return cm.exception

    def test_every_failure_has_its_own_kind_and_a_message_that_says_what_to_do(self):
        cases = [
            (FakeResp(401, REC_BAD_KEY), "auth", "missing valid authentication credentials"),   # RECORDED from the live service
            (FakeResp(401, {}), "auth", "HTTP 401"),
            (FakeResp(403, {"error": "Forbidden", "detailedError": {"code": "x", "message": "Invalid key"}}), "auth", "answer suggests the key is invalid or has expired"),
            (FakeResp(403, {"detailedError": {"message": "API key not authorized for this product"}}), "forbidden", "enable the Traffic API"),
            (FakeResp(403, {"error": "Not enabled"}), "forbidden", "answer suggests this key may not be enabled for the Traffic Flow API"),
            (FakeResp(403, {"detailedError": {"message": "Over the limit"}}), "quota", "suggests the key's request quota may be used up"),
            (FakeResp(403, {"error": "daily quota exceeded"}), "quota", "usage and plan"),
            (FakeResp(403, {}), "auth", "did not say which"),
            (FakeResp(403, {"weird": "shape"}), "auth", "did not say which"),
            (FakeResp(403, []), "auth", "did not say which"),
            (FakeResp(429, {}), "rate_limited", "limiting requests"),
            (FakeResp(400, REC_NO_ROAD), "no_segment", "no traffic segment"),                  # RECORDED: a point with no road near it
            (FakeResp(400, REC_BAD_POINT), "unexpected_status", "HTTP 400"),                    # RECORDED: other 400s are not "no segment"
            (FakeResp(400, REC_NO_POINT), "unexpected_status", "does not recognise"),
            (FakeResp(400, {}), "unexpected_status", "HTTP 400"),
            (FakeResp(404, CONSTRUCTED_404), "unexpected_status", "HTTP 404"),                  # RECORDED for an unknown service version (values constructed)
            (FakeResp(596, {}), "unexpected_status", "HTTP 596"),                               # documented as "Unknown version of the service"
            (FakeResp(500, {}), "server", "temporarily unavailable"),
            (FakeResp(503, {}), "server", "temporarily unavailable"),
            (FakeResp(200, json_error=True), "invalid_response", "could not be read"),
            (FakeResp(200, {"flowSegmentData": "nope"}), "invalid_response", "rejected")]
        for resp, kind, fragment in cases:
            with self.subTest(status=resp.status_code, kind=kind):
                e = self.expect(resp)
                self.assertEqual(e.kind, kind)
                self.assertIn(fragment, e.message)

    def test_the_documented_403_meaning_is_quoted_and_no_unobserved_cause_is_added_as_fact(self):
        """TomTom's documentation (docs.tomtom.com, Flow Segment Data, checked 2026-10-30) describes a 403 as 'The supplied API Key is not valid for this
        request.' That sentence is quoted; the possible reasons that follow it are only possibilities, and the answer is said not to say which."""
        msg = self.expect(FakeResp(403, {})).message
        self.assertIn("The supplied API Key is not valid for this request.", msg)
        self.assertIn("Possible reasons", msg)
        self.assertIn("did not say which", msg)
        self.assertNotIn("suggests", msg)

    def test_an_observed_401_says_what_tomtom_said_and_adds_no_cause(self):
        msg = self.expect(FakeResp(401, REC_BAD_KEY)).message
        self.assertIn("HTTP 401", msg)
        self.assertIn("missing valid authentication credentials", msg)
        self.assertNotIn("expired", msg)          # 'expired' was never observed for a 401
        self.assertNotIn("quota", msg)

    def test_a_403_cause_is_never_stated_as_fact_and_the_unverified_wording_is_admitted(self):
        """The wording of a real HTTP 403 has never been observed, so a cause taken from its body is only ever a suggestion; an unrecognised
        body gets the neutral message; only the 401 (observed against the live service) is stated plainly."""
        for body in ({"error": "Not enabled"}, {"detailedError": {"message": "Over the limit"}}, {"error": "Invalid key"}):
            msg = self.expect(FakeResp(403, body)).message
            self.assertIn("HTTP 403", msg)
            self.assertIn("suggests", msg)
        for body in ({"error": "Not enabled"}, {"detailedError": {"message": "Over the limit"}}):
            self.assertIn("has not been verified", self.expect(FakeResp(403, body)).message)
        for body in ({}, {"weird": "shape"}, []):
            neutral = self.expect(FakeResp(403, body)).message
            self.assertIn("did not say which", neutral)
            self.assertNotIn("suggests", neutral)
        self.assertNotIn("HTTP 403", self.expect(FakeResp(401, {"error": "Not enabled"})).message)   # a 401 stays the plain "not accepted"
        self.assertEqual(self.expect(FakeResp(401, {"error": "Over the limit"})).kind, "auth")      # and its body cannot turn it into something else
        for exc, kind, fragment in ((requests.Timeout("slow"), "timeout", "did not answer in time"),
                                    (requests.ConnectionError("down"), "network", "Could not reach TomTom")):
            e = self.expect(exc=exc)
            self.assertEqual(e.kind, kind)
            self.assertIn(fragment, e.message)

    def test_missing_credentials_are_refused_before_any_request(self):
        for key in (None, "", "   "):
            get = mock.Mock()
            with self.assertRaises(tomtom.ProviderError) as cm:
                tomtom.fetch_flow(12.9, 77.6, key, http_get=get)
            self.assertEqual(cm.exception.kind, "no_key")
            get.assert_not_called()

    def test_the_five_credential_and_limit_failures_are_told_apart(self):
        kinds = {self.expect(FakeResp(*a)).kind for a in ((401, {}), (403, {"error": "not enabled"}), (403, {"error": "over the limit"}), (429, {}))}
        self.assertEqual(kinds, {"auth", "forbidden", "quota", "rate_limited"})
        self.assertEqual(self.expect(exc=requests.ConnectionError("x")).kind, "network")
        self.assertEqual(self.expect(FakeResp(400, REC_NO_ROAD)).kind, "no_segment")

    def test_the_key_never_appears_even_when_the_provider_echoes_it(self):
        echoes = [FakeResp(403, {"error": f"Invalid key {KEY}", "detailedError": {"message": f"key={KEY} not authorized"}}),
                  FakeResp(401, {"message": KEY}), FakeResp(403, {"error": f"{KEY} over the limit"}), FakeResp(500, {"error": KEY})]
        for resp in echoes:
            e = self.expect(resp)
            for text in (e.message, str(e), repr(e)):
                self.assertNotIn(KEY, text)
        leaky = requests.ConnectionError(f"HTTPSConnectionPool(host='api.tomtom.com'): url: /x?key={KEY}")
        e = self.expect(exc=leaky)
        self.assertNotIn(KEY, e.message + str(e))
        self.assertIsNone(e.__cause__)

    def test_the_new_kinds_reach_the_status_and_the_page_messages(self):
        self.assertEqual((live_map.SEVERITY["forbidden"], live_map.SEVERITY["quota"]), ("error", "warning"))
        status = {"requests_made": 1, "budget": 300, "ttl_s": 0, "cooldown_remaining_s": 0.0, "last_success_at": None}
        forb = health.provider_item(True, {**status, "last_error": ("forbidden", 1000.0)}, 1010.0)
        quota = health.provider_item(True, {**status, "last_error": ("quota", 1000.0)}, 1010.0)
        self.assertEqual((forb.state, quota.state), ("bad", "warn"))
        self.assertIn("enable the Traffic API", forb.summary)
        self.assertIn("quota", quota.summary)

    def test_the_live_page_shows_each_failure_and_never_a_reading(self):
        cases = [(FakeResp(403, {"error": "not enabled for this product"}), "may not be enabled for the Traffic Flow API"),
                 (FakeResp(403, {"error": "over the limit"}), "quota may be used up"), (FakeResp(401, REC_BAD_KEY), "missing valid authentication credentials"),
                 (requests.ConnectionError(f"x key={KEY}"), "Could not reach TomTom"), (FakeResp(400, REC_NO_ROAD), "no traffic segment"),
                 (FakeResp(404, CONSTRUCTED_404), "does not recognise")]
        for outcome, expected in cases:
            with self.subTest(expected):
                h = PageHarness(key=KEY, tomtom_outcome=outcome).start()
                try:
                    h.search("silk board junction").choose(0).fetch()
                    shown = " ".join(x.value for g in (h.at.error, h.at.warning, h.at.info) for x in g)
                    self.assertIn(expected, shown)
                    self.assertEqual(h.specs[-1]["lines"], [])
                    self.assertNotIn(KEY, h.texts() + shown)
                    self.assertNotIn("km/h", h.texts())
                finally:
                    h.stop()

    def test_a_successful_mocked_response_is_still_accepted(self):
        obs = fetch(FakeResp(200, flow_payload()))
        self.assertEqual(obs.current_speed_kmh, 24)


# ======================================================================================================== the self-test
class CheckKeyTests(unittest.TestCase):
    def run_cli(self, argv, env, get=None):
        out = StringIO()
        with mock.patch.dict(os.environ), mock.patch.object(credentials, "_windows_lookup", return_value=NOTHING_SAVED):
            os.environ.pop("TOMTOM_API_KEY", None)
            os.environ.update(env)
            code = check_key.main(argv, http_get=get or mock.Mock(side_effect=AssertionError("no request expected")), out=out)
        return code, out.getvalue()

    def test_no_key_means_exit_1_and_no_request(self):
        code, text = self.run_cli(["--request"], {})
        self.assertEqual(code, 1)
        self.assertIn("RESULT: NOT CONFIGURED", text)

    def test_a_key_without_request_makes_no_network_call_and_does_not_claim_it_works(self):
        code, text = self.run_cli([], {"TOMTOM_API_KEY": KEY})
        self.assertEqual(code, 0)
        self.assertIn("That does not prove it works", text)
        self.assertNotIn(KEY, text)

    def test_one_request_on_success_reports_the_reading_and_scope_of_the_result(self):
        get = mock.Mock(return_value=FakeResp(200, flow_payload()))
        code, text = self.run_cli(["--request"], {"TOMTOM_API_KEY": KEY}, get)
        self.assertEqual((code, get.call_count), (0, 1))
        self.assertIn("RESULT: OK", text)
        self.assertIn("current speed 24 km/h", text)
        self.assertIn("says nothing about other roads", text)
        self.assertNotIn(KEY, text)
        (url,), kw = get.call_args
        self.assertTrue(url.startswith("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/"))
        geo.validate_location(*map(float, kw["params"]["point"].split(",")))
        self.assertEqual(kw["params"]["key"], KEY)  # sent to TomTom by the server, never printed

    def test_every_failure_is_labelled_and_exits_2(self):
        cases = [(FakeResp(401, {}), "KEY NOT ACCEPTED"), (FakeResp(403, {"error": "not enabled"}), "KEY MAY NOT BE ENABLED FOR THE TRAFFIC FLOW API"),
                 (FakeResp(403, {"error": "over the limit"}), "QUOTA MAY BE USED UP"), (FakeResp(429, {}), "RATE LIMITED"),
                 (FakeResp(503, {}), "TOMTOM PROBLEM"), (FakeResp(400, REC_NO_ROAD), "NO TRAFFIC DATA NEAR THE POINT"), (FakeResp(404, CONSTRUCTED_404), "UNEXPECTED ANSWER"),
                 (FakeResp(200, json_error=True), "UNREADABLE ANSWER")]
        for resp, label in cases:
            with self.subTest(label):
                code, text = self.run_cli(["--request"], {"TOMTOM_API_KEY": KEY}, mock.Mock(return_value=resp))
                self.assertEqual(code, 2)
                self.assertIn(f"REQUEST FAILED ({label})", text)
                self.assertNotIn(KEY, text)
        for exc, label in ((requests.Timeout(f"slow key={KEY}"), "TIMEOUT"), (requests.ConnectionError(f"down key={KEY}"), "NO CONNECTION")):
            code, text = self.run_cli(["--request"], {"TOMTOM_API_KEY": KEY}, mock.Mock(side_effect=exc))
            self.assertEqual(code, 2)
            self.assertIn(label, text)
            self.assertNotIn(KEY, text)

    def test_an_unexpected_error_prints_only_its_type(self):
        code, text = self.run_cli(["--request"], {"TOMTOM_API_KEY": KEY}, mock.Mock(side_effect=RuntimeError(f"boom {KEY}")))
        self.assertEqual(code, 2)
        self.assertIn("unexpected RuntimeError", text)
        self.assertNotIn(KEY, text)
        self.assertNotIn("boom", text)

    def test_a_malformed_value_is_explained_with_exit_1(self):
        code, text = self.run_cli([], {"TOMTOM_API_KEY": "two words"})
        self.assertEqual(code, 1)
        self.assertIn("exists but cannot be used", text)

    def test_the_module_never_stores_or_prints_the_key(self):
        src = (root_dir / "src" / "live" / "check_key.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"print\(.*\.key|say\(.*\.key|f\".*\{cred\.key")


# ======================================================================================================== the launcher
@unittest.skipUnless(os.name == "nt" and LAUNCHER.is_file(), "Windows PowerShell launcher only")
class LauncherTests(unittest.TestCase):
    def test_the_launch_command_is_unchanged_and_the_key_is_never_printed(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("'streamlit', 'run', 'app.py'", text)
        self.assertIn("Import-MissingEnvironment -Names 'TOMTOM_API_KEY', 'LIVE_CACHE_TTL_S', 'LIVE_MAX_REQUESTS'", text)
        for line in text.splitlines():
            if "Write-Host" in line:
                self.assertNotRegex(line, r"\$env:TOMTOM_API_KEY|\$v\b|GetEnvironmentVariable\('TOMTOM_API_KEY'", line)

    def run_ps(self, body, env=None):
        script = ("$src = Get-Content -Raw '" + str(LAUNCHER) + "'\n"
                  "$m = [regex]::Match($src, '(?s)function Import-MissingEnvironment \\{.*?\\n\\}\\r?\\n')\n"
                  ". ([scriptblock]::Create($m.Value))\n" + body)
        full_env = dict(os.environ)
        full_env.update(env or {})
        return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], capture_output=True, text=True,
                              timeout=60, env=full_env).stdout.strip()

    def test_a_saved_value_is_loaded_only_when_the_terminal_has_none(self):
        out = self.run_ps("""
            $env:TC_T_ABSENT = $null; $env:TC_T_PRESENT = 'already-here'
            $fake = { param($n, $s) if ($n -eq 'TC_T_ABSENT' -and $s -eq 'User') { 'from-user' } elseif ($n -eq 'TC_T_MACH' -and $s -eq 'Machine') { 'from-machine' } elseif ($n -eq 'TC_T_BOTH') { if ($s -eq 'User') { 'user-wins' } else { 'machine-loses' } } }
            $r = @(Import-MissingEnvironment -Names 'TC_T_ABSENT','TC_T_PRESENT','TC_T_MACH','TC_T_BOTH','TC_T_NONE_X' -Getter $fake)
            ($r | ForEach-Object { "$($_.Name)<-$($_.Scope)" }) -join ','
            "$env:TC_T_ABSENT|$env:TC_T_PRESENT|$env:TC_T_MACH|$env:TC_T_BOTH"
        """)
        first, second = out.splitlines()[-2:]
        self.assertEqual(first, "TC_T_ABSENT<-User,TC_T_MACH<-Machine,TC_T_BOTH<-User")
        self.assertEqual(second, "from-user|already-here|from-machine|user-wins")

    def test_blank_saved_values_are_not_loaded(self):
        out = self.run_ps("""
            $env:TC_T_BLANK = $null
            $r = @(Import-MissingEnvironment -Names 'TC_T_BLANK' -Getter { param($n, $s) '   ' })
            "count=$($r.Count) value=[$env:TC_T_BLANK]"
        """)
        self.assertEqual(out.splitlines()[-1], "count=0 value=[]")

    def test_check_only_reports_the_key_status_without_the_value(self):
        env = {"TOMTOM_API_KEY": KEY, "TF_CPP_MIN_LOG_LEVEL": "3"}
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER), "-CheckOnly"], capture_output=True, text=True,
                           timeout=240, env={**os.environ, **env}, cwd=str(root_dir))
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-500:])
        self.assertIn("TomTom key: found in the environment (value not shown).", out)
        self.assertIn("configured via the environment variable (value not shown)", out)
        self.assertNotIn(KEY, out)


if __name__ == "__main__":
    unittest.main()

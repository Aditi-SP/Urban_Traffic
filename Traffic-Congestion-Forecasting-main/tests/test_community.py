"""
Tests for Community Reports (src/community/store.py and src/ui/community.py).

Every test uses a TEMPORARY database (never data/community/community.db). Nothing is downloaded and no network is used.

Run:   python -m unittest tests.test_community -v
"""
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database: tests never touch the real community.db)

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.community import store  # noqa: E402

PAGE = str(root_dir / "src" / "ui" / "pages" / "community.py")
GOOD = ("Accident", "Silk Board Junction", "Two cars are blocking the left lane near the flyover.")
REAL_DB = root_dir / "data" / "community" / "community.db"


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "sub" / "c.db"

    def save(self, *args, **kw):
        a = args or GOOD
        return store.submit(*a, session_id=kw.pop("session_id", "s1"), path=self.db, **kw)

    # ---- viewing never creates anything
    def test_reading_an_absent_database_creates_nothing_and_shows_no_reports(self):
        self.assertEqual((store.recent(path=self.db), store.count(path=self.db)), ([], 0))
        self.assertFalse(self.db.exists())
        self.assertFalse(self.db.parent.exists())

    # ---- saving
    def test_a_saved_report_is_read_back_and_listed(self):
        r, outcome = self.save()
        self.assertEqual((outcome, r.id, r.category, r.place, r.status), ("saved", 1, "Accident", "Silk Board Junction", "unverified"))
        self.assertTrue(self.db.is_file())
        self.assertEqual(store.count(path=self.db), 1)
        listed = store.recent(path=self.db)
        self.assertEqual(listed, [r])
        self.assertAlmostEqual(r.created_at, time.time(), delta=5)  # the server sets the time, not the user

    def test_reports_persist_across_connections_and_come_newest_first(self):
        self.save("Accident", "Hosur Road", "First report about the junction being blocked.", now=1000.0)
        self.save("Roadwork", "Tumkur Road", "Second report: lane closed for resurfacing work.", now=2000.0)
        self.save("Other", "CMH Road", "Third report with enough characters here.", now=3000.0)
        self.assertEqual([r.place for r in store.recent(path=self.db)], ["CMH Road", "Tumkur Road", "Hosur Road"])
        self.assertEqual([r.place for r in store.recent(2, path=self.db)], ["CMH Road", "Tumkur Road"])

    def test_the_same_token_cannot_create_two_rows(self):
        a, o1 = self.save(token="tok")
        b, o2 = self.save(token="tok")
        self.assertEqual((o1, o2, a.id, b.id, store.count(path=self.db)), ("saved", "already_saved", 1, 1, 1))

    def test_nothing_is_stored_when_a_report_is_refused(self):
        with self.assertRaises(store.ReportError):
            store.submit("Accident", "", "long enough description", session_id="s", path=self.db)
        self.assertEqual(store.count(path=self.db), 0)

    # ---- validation
    def test_invalid_input_is_rejected_with_a_reason(self):
        cases = [(("Nope", "Place", "long enough description"), "category"), (("Accident", "", "long enough description"), "road or place"),
                 (("Accident", "ab", "long enough description"), "too short"), (("Accident", "x" * 121, "long enough description"), "too long"),
                 (("Accident", "Place", "short"), "description is too short"), (("Accident", "Place", "y" * 501), "too long"),
                 (("Accident", "Place\x00x", "long enough description"), "not allowed"), (("Accident", "Place", "bad \x07 text here"), "not allowed"),
                 (("Accident", None, "long enough description"), "required"), ((None, "Place", "long enough description"), "category")]
        for args, needle in cases:
            with self.subTest(args), self.assertRaises(store.ReportError) as cm:
                self.save(*args)
            self.assertEqual(cm.exception.kind, "invalid")
            self.assertIn(needle, cm.exception.message)
        self.assertEqual(store.count(path=self.db), 0)

    def test_text_is_cleaned_and_sql_is_never_built_from_input(self):
        evil = "Robert'); DROP TABLE reports;-- <b>x</b>"
        r, _ = self.save("Other", "  Silk   Board  ", evil + " and more text")
        self.assertEqual(r.place, "Silk Board")
        self.assertEqual(r.description, evil + " and more text")  # stored literally, as plain text
        self.assertEqual(store.count(path=self.db), 1)  # the table still exists
        ml, _ = self.save("Other", "CMH Road", "line one here\n\n\n\nline two here")
        self.assertEqual(ml.description, "line one here\n\nline two here")

    # ---- safeguards for an anonymous service
    def test_an_identical_report_inside_the_window_is_refused_but_allowed_later(self):
        self.save(now=1000.0)
        with self.assertRaises(store.ReportError) as cm:
            self.save(now=1000.0 + 60, session_id="other")
        self.assertEqual(cm.exception.kind, "duplicate")
        self.assertEqual(store.count(path=self.db), 1)
        again, outcome = self.save(now=1000.0 + store.DUPLICATE_WINDOW_S + 1, session_id="other")
        self.assertEqual(outcome, "saved")
        self.assertEqual(store.count(path=self.db), 2)

    def test_a_session_is_rate_limited_and_the_limit_expires(self):
        for i in range(store.SESSION_LIMIT[0]):
            self.save("Other", f"Place {i}", f"Report number {i} with enough text", now=1000.0 + i)
        with self.assertRaises(store.ReportError) as cm:
            self.save("Other", "Another place", "One report too many for this session", now=1010.0)
        self.assertEqual(cm.exception.kind, "rate_limited")
        self.assertEqual(store.count(path=self.db), store.SESSION_LIMIT[0])
        self.save("Other", "Another place", "Sent after the window has passed", now=1000.0 + store.SESSION_LIMIT[1] + 10)  # no error
        self.save("Other", "Different session place", "A different session is not limited", now=1010.0, session_id="someone else")

    def test_the_whole_service_has_an_hourly_and_a_total_cap(self):
        with mock.patch.object(store, "GLOBAL_LIMIT", (3, 3600)):
            for i in range(3):
                self.save("Other", f"Place {i}", f"Report number {i} with enough text", now=1000.0 + i, session_id=f"s{i}")
            with self.assertRaises(store.ReportError) as cm:
                self.save("Other", "One more", "Over the hourly cap for everyone", now=1010.0, session_id="s9")
            self.assertEqual(cm.exception.kind, "rate_limited")
        with mock.patch.object(store, "MAX_REPORTS", 1):
            with self.assertRaises(store.ReportError) as cm:
                self.save("Other", "Overflow", "Over the total cap of the store", now=9_000_000.0, session_id="s10")
            self.assertEqual(cm.exception.kind, "full")

    # ---- failures are visible, safe and do not lose or invent data
    def test_an_unusable_database_gives_a_safe_actionable_error(self):
        blocker = Path(self.tmp.name) / "afile"
        blocker.write_text("x")
        bad = blocker / "c.db"  # a path below a regular file can never be created
        with self.assertRaises(store.ReportError) as cm:
            store.submit(*GOOD, session_id="s", path=bad)
        self.assertEqual(cm.exception.kind, "storage")
        self.assertIn("data/community", cm.exception.message)
        self.assertNotIn(self.tmp.name, cm.exception.message, "no absolute path in a user-facing message")
        self.assertNotIn("Traceback", cm.exception.message)

    def test_a_corrupt_database_is_an_error_not_an_empty_list(self):
        self.db.parent.mkdir(parents=True)
        self.db.write_bytes(b"this is not a sqlite database at all" * 20)
        for call in (lambda: store.recent(path=self.db), lambda: store.count(path=self.db), lambda: store.submit(*GOOD, session_id="s", path=self.db)):
            with self.assertRaises(store.ReportError) as cm:
                call()
            self.assertEqual(cm.exception.kind, "storage")
        self.assertEqual(store.health(path=self.db)["state"], "bad")

    def test_a_database_from_a_newer_version_is_refused(self):
        self.db.parent.mkdir(parents=True)
        con = sqlite3.connect(self.db)
        con.execute(f"PRAGMA user_version = {store.SCHEMA_VERSION + 1}")
        con.commit()
        con.close()
        with self.assertRaises(store.ReportError) as cm:
            store.count(path=self.db)
        self.assertIn("newer version", cm.exception.message)

    def test_a_report_that_cannot_be_read_back_is_not_reported_as_saved(self):
        real = store._connect
        calls = {"n": 0}

        def flaky(path, create):
            calls["n"] += 1
            return None if calls["n"] == 2 else real(path, create)  # the read-back connection finds nothing
        with mock.patch.object(store, "_connect", side_effect=flaky), self.assertRaises(store.ReportError) as cm:
            self.save()
        self.assertEqual(cm.exception.kind, "not_saved")

    def test_the_schema_has_no_personal_fields_and_is_versioned(self):
        self.save()
        from contextlib import closing
        with closing(sqlite3.connect(self.db)) as con:
            cols = [r[1] for r in con.execute("PRAGMA table_info(reports)")]
            self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], store.SCHEMA_VERSION)
            row = con.execute("SELECT submitter FROM reports").fetchone()[0]
        self.assertEqual(cols, ["id", "created_at", "category", "place", "description", "status", "token", "fingerprint", "submitter"])
        self.assertNotIn("s1", row)  # the session id is hashed, not stored

    def test_health_reports_the_real_state(self):
        self.assertEqual(store.health(path=self.db)["count"], 0)
        self.assertIn("no reports yet", store.health(path=self.db)["summary"])
        self.save()
        h = store.health(path=self.db)
        self.assertEqual((h["state"], h["count"]), ("ok", 1))

    def test_the_module_has_no_edit_delete_or_network_code(self):
        src = (root_dir / "src" / "community" / "store.py").read_text(encoding="utf-8")
        for banned in ("DELETE ", "UPDATE ", "DROP ", "requests", "urllib", "socket", "http"):
            self.assertNotIn(banned, src.replace("hashlib", ""), banned)

    def test_the_default_database_is_git_ignored(self):
        text = (root_dir / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/community/", text)


# ======================================================================================================== the page
class PageHarness:
    def __init__(self, tmp, db=None):
        self.path = db or Path(tmp) / "pg" / "c.db"
        self.patch = mock.patch.dict(os.environ, {store.DB_ENV: str(self.path)})
        self.patch.start()
        self.at = AppTest.from_file(PAGE, default_timeout=60).run()

    def stop(self):
        self.patch.stop()

    def send(self, category="Accident", place="Silk Board Junction", description="Two cars are blocking the left lane near the flyover."):
        self.at.selectbox(key="cr_category").set_value(category)
        self.at.text_input(key="cr_place").set_value(place)
        self.at.text_area(key="cr_description").set_value(description)
        next(b for b in self.at.button if b.label == "Send report").click()
        self.at.run()
        return self

    def page_text(self):
        out = []
        for g in (self.at.markdown, self.at.caption, self.at.info, self.at.warning, self.at.error, self.at.success):
            out += [e.value for e in g]
        return " ".join(out)


class PageTests(unittest.TestCase):
    def harness(self, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        h = PageHarness(tmp.name, **kw)
        self.addCleanup(h.stop)
        return h

    def test_with_no_reports_it_says_so_and_invents_nothing(self):
        h = self.harness()
        self.assertEqual([e.value for e in h.at.exception], [])
        text = h.page_text()
        self.assertIn("No community reports have been saved yet", text)
        self.assertIn("No sample reports exist", text)
        self.assertNotIn("tc-card tc-info tc-tight", text, "no report cards may exist")
        self.assertFalse(h.path.exists(), "viewing must not create the database")
        self.assertEqual([b.label for b in h.at.button], ["Report Traffic with Photo", "Send report"], "the photo-report button and the text form; no edit or delete controls")
        for kind in ("file_uploader", "camera_input"):
            self.assertEqual(len(h.at.get(kind)), 0)

    def test_success_is_shown_only_after_the_report_is_stored_and_it_is_listed(self):
        h = self.harness().send()
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertEqual(len(h.at.success), 1)
        self.assertIn("Saved: your report (number 1) was stored on this computer", h.at.success[0].value)
        self.assertIn("anonymous and unverified", h.at.success[0].value)
        self.assertEqual(store.count(path=h.path), 1, "the success message must match a real row")
        text = h.page_text()
        self.assertIn("Silk Board Junction", text)
        self.assertIn("unverified", text)
        self.assertEqual(h.at.text_input(key="cr_place").value, "")  # the fields are cleared only after a confirmed save

    def test_a_report_survives_a_page_refresh_and_a_second_visitor_sees_it(self):
        h = self.harness().send()
        again = AppTest.from_file(PAGE, default_timeout=60).run()  # a brand-new session = the page reloaded or another visitor
        self.assertIn("Silk Board Junction", " ".join(m.value for m in again.markdown))
        self.assertEqual(len(again.success), 0, "a new session must not show the earlier save message")
        self.assertIn("1", " ".join(m.value for m in again.markdown if "Reports saved" in m.value))

    def test_invalid_input_shows_an_actionable_error_and_keeps_what_was_typed(self):
        h = self.harness().send(place="Silk Board Junction", description="short")
        self.assertEqual(len(h.at.success), 0)
        self.assertIn("The description is too short: write at least 10 characters.", h.at.error[0].value)
        self.assertEqual(h.at.text_input(key="cr_place").value, "Silk Board Junction")
        self.assertEqual(h.at.text_area(key="cr_description").value, "short")
        self.assertFalse(h.path.exists())

    def test_duplicates_and_rate_limits_are_explained_not_hidden(self):
        h = self.harness().send()
        h.send()  # identical content again
        self.assertIn("An identical report was already sent", h.at.error[0].value)
        self.assertEqual(store.count(path=h.path), 1)

    def test_report_text_is_shown_as_plain_text(self):
        h = self.harness().send(place="<img src=x onerror=alert(1)>", description="Look at <script>alert(1)</script> and [a link](http://example.com) **bold**")
        md = " ".join(m.value for m in h.at.markdown)
        self.assertNotIn("<script>", md)
        self.assertNotIn("<img", md)
        self.assertIn("&lt;script&gt;", md)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", md)

    def test_old_reports_are_marked_possibly_outdated_and_the_order_is_newest_first(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "c.db"
        store.submit("Accident", "Old place", "An old report from two days ago here.", session_id="a", path=db, now=time.time() - 2 * 86400)
        store.submit("Accident", "New place", "A fresh report from a few seconds ago.", session_id="b", path=db, now=time.time() - 5)
        h = PageHarness(tmp.name, db=db)
        self.addCleanup(h.stop)
        md = " ".join(m.value for m in h.at.markdown)
        self.assertLess(md.index("New place"), md.index("Old place"))
        self.assertEqual(md.count("possibly outdated"), 2 + 0 if "Reports older than a day are marked as possibly outdated" in md else 1)  # one card + the subtitle text

    def test_an_unavailable_database_disables_sending_and_explains_why(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "c.db"
        db.write_bytes(b"not a database" * 30)
        h = PageHarness(tmp.name, db=db)
        self.addCleanup(h.stop)
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertTrue(any("report database is unavailable" in e.value for e in h.at.error))
        self.assertTrue(next(b for b in h.at.button if b.label == "Send report").disabled)
        self.assertNotIn(tmp.name, " ".join(e.value for e in h.at.error), "no absolute path shown")

    def test_the_real_database_is_never_touched_by_these_tests(self):
        self.harness().send()
        self.assertFalse(REAL_DB.exists() and REAL_DB.stat().st_mtime > time.time() - 120, "tests must use temporary databases")


if __name__ == "__main__":
    unittest.main()

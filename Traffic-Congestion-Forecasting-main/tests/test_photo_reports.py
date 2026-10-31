"""
Community traffic PHOTO reports (src/community/photos.py, src/ui/photo_capture.py, the Community Reports page and the Live Traffic Map).

Every test uses its own temporary database and photo folder; the real data/community/community.db is never touched (tests/_isolation.py).
Time is controlled with the `now` argument (the SERVER clock), never by sleeping.
"""
import base64
import io
from contextlib import closing
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
import _isolation  # noqa: E402,F401

from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.community import photos, store  # noqa: E402
from src.ui import leaflet_map, live_map, photo_capture, photo_cards  # noqa: E402
from tests.test_live_map import APP, KEY, PageHarness  # noqa: E402

AREA_G = (12.9159, 77.6240)                 # where "User A" stands (Silk Board area)
NORTH_08_KM = (12.9159 + 0.8 / 111.32, 77.6240)   # 0.8 km from Area G
NORTH_30_KM = (12.9159 + 3.0 / 111.32, 77.6240)   # 3 km from Area G
T0 = 1_800_000_000.0                         # a fixed "server time"
PAGE_COMMUNITY = str(root_dir / "src" / "ui" / "pages" / "community.py")


def make_image(fmt="JPEG", size=(800, 600), color=(200, 30, 30), **save):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt, **save)
    return buf.getvalue()


def data_url(raw, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "community.db"
        patcher = mock.patch.dict(os.environ, {store.DB_ENV: str(self.db)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._n = 0

    def photo(self, at=AREA_G, *, session="A", description="Heavy traffic near junction", now=T0, accuracy=25, color=None):
        """Store one photo (each call a different picture, so the duplicate rule never interferes)."""
        self._n += 1
        c = color or ((37 * self._n) % 256, (91 * self._n) % 256, (53 * self._n) % 256)
        return photos.submit_photo(make_image(color=c), at[0], at[1], description, session_id=session, accuracy_m=accuracy, now=now, path=self.db)[0]


# ======================================================================================================== 1-3, 6: a valid submission, the position, the time
class SubmissionTests(Base):
    def test_a_valid_photo_is_stored_and_read_back(self):
        report, outcome = photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "Heavy traffic near junction", session_id="A", accuracy_m=25, now=T0, path=self.db)
        self.assertEqual(outcome, "saved")
        self.assertEqual((report.latitude, report.longitude, report.accuracy_m), (AREA_G[0], AREA_G[1], 25.0))
        self.assertEqual(report.description, "Heavy traffic near junction")
        self.assertEqual(report.status, "unverified")
        self.assertRegex(report.report_id, r"^[0-9a-f]{32}$")
        self.assertTrue(photos.read_image(report, path=self.db).startswith(b"\xff\xd8"), "a real JPEG was stored")
        self.assertEqual([r.report_id for r in photos.history(path=self.db)], [report.report_id])

    def test_a_data_url_from_the_browser_component_is_accepted(self):
        report, _ = photos.submit_photo(data_url(make_image()), AREA_G[0], AREA_G[1], "", session_id="A", accuracy_m=40, now=T0, path=self.db)
        self.assertEqual(report.description, "")

    def test_every_report_has_the_required_fields_and_the_reporter_is_not_exposed(self):
        report = self.photo()
        for field in ("report_id", "created_at", "expires_at", "latitude", "longitude", "description", "image_file", "status"):
            self.assertTrue(hasattr(report, field))
        self.assertEqual(report.created_utc, "2027-01-15T08:00:00+00:00")
        self.assertNotIn("A", {report.report_id, report.image_file}, "the session id is not part of anything shown")
        with closing(sqlite3.connect(str(self.db))) as con:
            who = con.execute("SELECT submitter FROM photo_reports").fetchone()[0]
        self.assertNotIn("A", who.replace("0", ""), "only a one-way pseudonym is stored") if len(who) != 1 else None
        self.assertEqual(len(who), 16)
        self.assertFalse(hasattr(report, "submitter"))

    def test_a_repeated_token_is_idempotent(self):
        a = photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=10, token="t1", now=T0, path=self.db)
        b = photos.submit_photo(make_image(color=(1, 2, 3)), AREA_G[0], AREA_G[1], "y", session_id="A", accuracy_m=10, token="t1", now=T0 + 1, path=self.db)
        self.assertEqual((a[1], b[1], a[0].report_id), ("saved", "already_saved", b[0].report_id))
        self.assertEqual(len(photos.history(path=self.db)), 1)


class LocationValidationTests(Base):
    def attempt(self, lat, lon, accuracy=25):
        with self.assertRaises(photos.PhotoError) as cm:
            photos.submit_photo(make_image(), lat, lon, "x", session_id="A", accuracy_m=accuracy, now=T0, path=self.db)
        self.assertEqual(cm.exception.kind, "location")
        self.assertEqual(photos.history(path=self.db), [], "nothing is stored when the position is refused")
        self.assertFalse((self.db.parent / "photos").exists() and any((self.db.parent / "photos").iterdir()), "and no file is left behind")
        return cm.exception.message

    def test_missing_gps_coordinates_are_refused_with_a_clear_reason(self):
        for lat, lon in ((None, None), (AREA_G[0], None), (None, AREA_G[1])):
            self.assertIn("No GPS position was received", self.attempt(lat, lon))

    def test_invalid_coordinates_are_refused(self):
        for lat, lon in (("12.9", "77.6"), (float("nan"), 77.6), (12.9, float("inf")), (True, False), (95.0, 77.6), (12.9, 200.0), ([12.9], {"x": 1})):
            with self.subTest(lat=lat, lon=lon):
                self.attempt(lat, lon)

    def test_a_position_outside_the_map_area_is_refused_unless_the_scope_is_widened(self):
        self.assertIn("outside the Bengaluru area", self.attempt(28.61, 77.20))       # Delhi
        with mock.patch.dict(os.environ, {photos.ENV_SCOPE: "any"}):
            report, _ = photos.submit_photo(make_image(), 28.61, 77.20, "x", session_id="A", accuracy_m=20, now=T0, path=self.db)
        self.assertEqual(report.latitude, 28.61)

    def test_the_gps_accuracy_must_be_reported_and_good_enough(self):
        self.assertIn("did not report how accurate", self.attempt(AREA_G[0], AREA_G[1], accuracy=None))
        self.assertIn("too imprecise", self.attempt(AREA_G[0], AREA_G[1], accuracy=5000))
        self.attempt(AREA_G[0], AREA_G[1], accuracy=-1)
        with mock.patch.dict(os.environ, {photos.ENV_MAX_ACCURACY_M: "6000"}):
            photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=5000, now=T0, path=self.db)   # configurable


class TimestampTests(Base):
    def test_the_upload_time_is_the_server_time_and_in_utc(self):
        report = self.photo(now=T0)
        self.assertEqual(report.created_at, T0)
        self.assertEqual(report.expires_at, T0 + 600)
        with closing(sqlite3.connect(str(self.db))) as con:
            iso = con.execute("SELECT created_at_utc FROM photo_reports").fetchone()[0]
        self.assertEqual(iso, "2027-01-15T08:00:00+00:00")
        self.assertTrue(iso.endswith("+00:00"), "UTC")

    def test_without_an_explicit_time_the_server_clock_is_used(self):
        before = time.time()
        report = photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=20, path=self.db)[0]
        self.assertTrue(before <= report.created_at <= time.time())

    def test_the_reporters_device_clock_has_no_way_in(self):
        """The component's value has no time field that the server reads, and submit_photo takes none from the browser."""
        info = photos.explain_capture({"image": data_url(make_image()), "lat": 12.9, "lon": 77.6, "accuracy_m": 10, "captured_at": 1, "device_time": 1, "now": 1})
        self.assertNotIn("captured_at", info)
        import inspect
        self.assertNotIn("created_at", inspect.signature(photos.submit_photo).parameters)
        self.assertNotIn("timestamp", inspect.signature(photos.submit_photo).parameters)


# ======================================================================================================== 4, 5: what is accepted
class ImageValidationTests(Base):
    def refused(self, raw, kind):
        with self.assertRaises(photos.PhotoError) as cm:
            photos.submit_photo(raw, AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=20, now=T0, path=self.db)
        self.assertEqual(cm.exception.kind, kind)
        self.assertEqual(photos.history(path=self.db), [])
        return cm.exception.message

    def test_formats_other_than_jpeg_png_and_webp_are_refused(self):
        self.assertIn("Only JPEG, PNG or WebP", self.refused(make_image("GIF"), "invalid_format"))
        self.assertIn("Only JPEG, PNG or WebP", self.refused(make_image("BMP"), "invalid_format"))
        self.assertIn("Only JPEG, PNG or WebP", self.refused(make_image("TIFF"), "invalid_format"))

    def test_png_and_webp_are_accepted_and_stored_as_jpeg(self):
        for fmt in ("PNG", "WEBP"):
            report = self.photo(color=(10 if fmt == "PNG" else 20, 90, 200)) if False else photos.submit_photo(
                make_image(fmt, color=(10 if fmt == "PNG" else 20, 90, 200)), AREA_G[0], AREA_G[1], "x", session_id=fmt, accuracy_m=20, now=T0, path=self.db)[0]
            with Image.open(io.BytesIO(photos.read_image(report, path=self.db))) as im:
                self.assertEqual(im.format, "JPEG")

    def test_files_that_are_not_images_are_refused_whatever_they_claim_to_be(self):
        for raw in (b"hello world", b"%PDF-1.4 fake", b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", b"MZ\x90\x00 an executable", b"\x00" * 500):
            with self.subTest(raw[:12]):
                self.refused(raw, "invalid_image")
        self.assertIn("No photo was received", self.refused(b"", "invalid_image"))

    def test_a_corrupt_or_truncated_image_is_refused(self):
        good = make_image()
        self.refused(good[: len(good) // 2], "invalid_image")
        self.refused(good[:20] + b"\x00" * 200, "invalid_image")

    def test_a_data_url_that_is_not_valid_base64_or_not_an_image_is_refused(self):
        for value in ("data:image/jpeg;base64,@@@not base64@@@", "data:text/html;base64,PGI+", "not a data url", "", None, 42):
            with self.subTest(str(value)[:20]), self.assertRaises(photos.PhotoError):
                photos.decode_data_url(value)

    def test_an_oversized_upload_is_refused_before_it_is_decoded(self):
        self.assertIn("too large", self.refused(b"\xff\xd8\xff" + b"0" * (9 * 1024 * 1024), "too_large"))
        with self.assertRaises(photos.PhotoError) as cm:
            photos.decode_data_url("data:image/jpeg;base64," + "A" * (12 * 1024 * 1024))
        self.assertEqual(cm.exception.kind, "too_large")

    def test_the_size_limit_is_configurable_and_the_default_is_8_mb(self):
        self.assertEqual(photos.max_upload_bytes(), 8 * 1024 * 1024)
        big = make_image(size=(2400, 1800), color=(3, 4, 5), quality=100)
        with mock.patch.dict(os.environ, {photos.ENV_MAX_MB: "0.1"}):
            self.assertEqual(photos.max_upload_bytes(), int(0.1 * 1024 * 1024))
            self.refused(make_image(size=(1600, 1200), color=(9, 9, 9)) * 40, "too_large")
        self.assertGreater(len(big), 0)

    def test_an_image_with_too_many_pixels_or_too_few_is_refused(self):
        with mock.patch.object(photos, "MAX_PIXELS", 10_000):
            self.assertIn("too many pixels", self.refused(make_image(size=(200, 200)), "too_large"))
        self.assertIn("too small", self.refused(make_image(size=(10, 10)), "invalid_image"))

    def test_animated_images_are_refused(self):
        buf = io.BytesIO()
        frames = [Image.new("RGB", (100, 100), c) for c in ((255, 0, 0), (0, 255, 0))]
        frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
        self.refused(buf.getvalue(), "invalid_format")
        webp = io.BytesIO()
        frames[0].save(webp, "WEBP", save_all=True, append_images=frames[1:])
        self.refused(webp.getvalue(), "invalid_format")

    def test_the_stored_photo_carries_no_metadata_and_is_downscaled(self):
        im = Image.new("RGB", (3000, 2000), (50, 120, 200))
        exif = Image.Exif()
        exif[0x010F] = "SecretPhoneMaker"
        exif[0x8825] = {1: "N", 2: (12.0, 55.0, 0.0), 3: "E", 4: (77.0, 37.0, 0.0)}     # a GPS block in the file
        buf = io.BytesIO()
        im.save(buf, "JPEG", exif=exif)
        self.assertIn(b"SecretPhoneMaker", buf.getvalue(), "the test photo really contains metadata")
        report, _ = photos.submit_photo(buf.getvalue(), AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=20, now=T0, path=self.db)
        stored = photos.read_image(report, path=self.db)
        self.assertNotIn(b"SecretPhoneMaker", stored)
        with Image.open(io.BytesIO(stored)) as out:
            self.assertEqual(len(out.getexif()), 0, "no EXIF at all")
            self.assertLessEqual(max(out.size), photos.MAX_SIDE)
        with Image.open(io.BytesIO(photos.read_image(report, thumb=True, path=self.db))) as t:
            self.assertLessEqual(max(t.size), photos.THUMB_SIDE)

    def test_the_description_is_plain_text_and_limited(self):
        for bad in ("x" * 201, "bell\x07char", 12345):
            with self.subTest(str(bad)[:12]), self.assertRaises(photos.PhotoError) as cm:
                photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], bad, session_id="A", accuracy_m=20, now=T0, path=self.db)
            self.assertEqual(cm.exception.kind, "invalid")
        r = photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "  two\nlines   here  ", session_id="A", accuracy_m=20, now=T0, path=self.db)[0]
        self.assertEqual(r.description, "two lines here")

    def test_file_names_are_random_and_inside_the_photo_folder(self):
        report = self.photo(description="../../evil")
        self.assertRegex(report.image_file, r"^[0-9a-f]{32}\.jpg$")
        files = sorted(p.name for p in (self.db.parent / "photos").iterdir())
        self.assertEqual(files, sorted([report.image_file, report.image_file[:-4] + "_t.jpg"]))
        forged = photos.PhotoReport("x", T0, T0 + 600, 1, 1, 1, "", "../../secret.jpg")
        self.assertIsNone(photos.read_image(forged, path=self.db), "a name from the database is checked before it is opened")


# ======================================================================================================== 7, 8: ten-minute freshness
class FreshnessTests(Base):
    def test_a_report_is_fresh_for_ten_minutes_and_then_expired(self):
        r = self.photo(now=T0)
        self.assertEqual(photos.fresh_seconds(), 600)
        for dt_s, fresh in ((0, True), (59, True), (599, True), (599.9, True), (600, False), (601, False), (3600, False)):
            with self.subTest(seconds_after_upload=dt_s):
                self.assertEqual(r.is_fresh(T0 + dt_s), fresh)
                self.assertEqual(r.state(T0 + dt_s), "fresh" if fresh else "expired")

    def test_the_default_results_contain_only_fresh_reports(self):
        r = self.photo(now=T0)
        self.assertEqual([n.report.report_id for n in photos.nearby(*AREA_G, now=T0 + 599, path=self.db)], [r.report_id])
        self.assertEqual(photos.nearby(*AREA_G, now=T0 + 601, path=self.db), [], "after ten minutes it is NOT offered as current traffic")
        self.assertEqual(photos.count_fresh(now=T0 + 599, path=self.db), 1)
        self.assertEqual(photos.count_fresh(now=T0 + 601, path=self.db), 0)

    def test_expired_reports_stay_in_the_history_and_are_labelled_expired(self):
        r = self.photo(now=T0)
        hist = photos.history(path=self.db)
        self.assertEqual([h.report_id for h in hist], [r.report_id])
        self.assertEqual(hist[0].state(T0 + 700), "expired")
        self.assertIn("Expired", photos.state_text(hist[0], T0 + 700))
        self.assertEqual(len(photos.nearby(*AREA_G, now=T0 + 700, include_expired=True, path=self.db)), 1)

    def test_the_freshness_duration_is_configurable(self):
        with mock.patch.dict(os.environ, {photos.ENV_FRESH_MIN: "2"}):
            r = self.photo(now=T0)
            self.assertEqual(r.expires_at, T0 + 120)
            self.assertEqual(len(photos.nearby(*AREA_G, now=T0 + 119, path=self.db)), 1)
            self.assertEqual(len(photos.nearby(*AREA_G, now=T0 + 121, path=self.db)), 0)
        with mock.patch.dict(os.environ, {photos.ENV_FRESH_MIN: "30"}):
            self.assertEqual(photos.fresh_seconds(), 1800)
        for junk in ("abc", "-5", "nan", ""):
            with mock.patch.dict(os.environ, {photos.ENV_FRESH_MIN: junk}):
                self.assertGreater(photos.fresh_seconds(), 0, "an unusable setting falls back to a safe value")

    def test_a_later_change_of_the_setting_does_not_resurrect_an_expired_report(self):
        r = self.photo(now=T0)
        with mock.patch.dict(os.environ, {photos.ENV_FRESH_MIN: "120"}):
            self.assertEqual(photos.nearby(*AREA_G, now=T0 + 700, path=self.db), [], "the expiry time was fixed when the photo was uploaded")
        self.assertEqual(r.expires_at, T0 + 600)

    def test_the_wording_for_the_age_of_a_photo(self):
        r = self.photo(now=T0)
        for dt_s, text in ((0, "Uploaded just now"), (30, "Uploaded just now"), (60, "Uploaded 1 minute ago"), (180, "Uploaded 3 minutes ago"),
                           (480, "Uploaded 8 minutes ago"), (3 * 3600, "Uploaded 3 hours ago")):
            self.assertEqual(photos.uploaded_text(r, T0 + dt_s), text)

    def test_old_photo_files_are_deleted_after_the_retention_time_but_the_entry_stays(self):
        r = self.photo(now=T0)
        self.assertEqual(photos.purge(now=T0 + 3600, path=self.db), 0, "within the retention time nothing is deleted")
        self.assertIsNotNone(photos.read_image(r, path=self.db))
        self.assertEqual(photos.purge(now=T0 + 25 * 3600, path=self.db), 1)
        self.assertEqual(list((self.db.parent / "photos").iterdir()), [])
        left = photos.history(path=self.db)
        self.assertEqual((len(left), left[0].image_file), (1, ""))
        self.assertIsNone(photos.read_image(left[0], path=self.db))


# ======================================================================================================== 9-12: proximity, never the place name
class ProximityTests(Base):
    def test_a_report_within_two_kilometres_is_found_and_its_distance_is_reported(self):
        self.photo(AREA_G)
        found = photos.nearby(*NORTH_08_KM, now=T0 + 240, path=self.db)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].distance_km, 0.8, delta=0.02)
        self.assertEqual(photos.distance_text(found[0].distance_km), "0.8 km from your searched location")
        self.assertEqual(photos.uploaded_text(found[0].report, T0 + 240), "Uploaded 4 minutes ago")

    def test_a_report_outside_the_radius_is_never_returned(self):
        self.photo(AREA_G)
        self.assertEqual(photos.nearby(*NORTH_30_KM, now=T0 + 60, path=self.db), [])
        self.assertEqual(photos.nearby(*NORTH_30_KM, 2.0, now=T0 + 60, path=self.db), [])
        # the boundary is exact: just inside is found, just outside is not
        inside = (AREA_G[0] + 1.99 / 111.32, AREA_G[1])
        outside = (AREA_G[0] + 2.01 / 111.32, AREA_G[1])
        self.assertEqual(len(photos.nearby(*inside, 2.0, now=T0 + 60, path=self.db)), 1)
        self.assertEqual(len(photos.nearby(*outside, 2.0, now=T0 + 60, path=self.db)), 0)

    def test_the_radius_is_configurable_and_starts_at_two_kilometres(self):
        self.assertEqual(photos.radius_km(), 2.0)
        self.photo(AREA_G)
        self.assertEqual(photos.nearby(*NORTH_30_KM, now=T0 + 60, path=self.db), [])
        with mock.patch.dict(os.environ, {photos.ENV_RADIUS_KM: "5"}):
            self.assertEqual(photos.radius_km(), 5.0)
            self.assertEqual(len(photos.nearby(*NORTH_30_KM, now=T0 + 60, path=self.db)), 1, "the same report is found once the radius allows it")
        self.assertEqual(len(photos.nearby(*NORTH_30_KM, 3.5, now=T0 + 60, path=self.db)), 1)

    def test_distance_decides_not_the_place_name(self):
        r = self.photo(AREA_G, description="Silk Board Junction")      # the description names a place
        far = (13.0358, 77.5970)                                         # Hebbal: a different area (about 12 km away)
        self.assertEqual(photos.nearby(*far, now=T0 + 60, path=self.db), [], "a matching name elsewhere does not matter")
        near_other_name = photos.nearby(*NORTH_08_KM, now=T0 + 60, path=self.db)
        self.assertEqual([n.report.report_id for n in near_other_name], [r.report_id], "and nearby is found without any name match")

    def test_searching_an_area_with_no_reports_returns_nothing(self):
        self.assertEqual(photos.nearby(*AREA_G, now=T0, path=self.db), [], "no database at all")
        self.photo(AREA_G)
        self.assertEqual(photos.nearby(13.0358, 77.5970, now=T0 + 60, path=self.db), [], "a database with reports elsewhere")

    def test_several_reports_newest_first_and_the_expired_one_is_left_out(self):
        old = self.photo(AREA_G, now=T0)
        new = self.photo(NORTH_08_KM, now=T0 + 500, session="B")
        found = photos.nearby(*AREA_G, now=T0 + 550, path=self.db)
        self.assertEqual([n.report.report_id for n in found], [new.report_id, old.report_id])
        found = photos.nearby(*AREA_G, now=T0 + 650, path=self.db)         # the first one expired at T0 + 600
        self.assertEqual([n.report.report_id for n in found], [new.report_id])

    def test_two_users_two_sessions_user_b_sees_what_user_a_uploaded(self):
        a = self.photo(AREA_G, session="user-A-browser", description="Heavy traffic near junction", now=T0)
        # "User B" is a different session, searching a different (nearby) point; nothing links the two except the position
        found = photos.nearby(*NORTH_08_KM, now=T0 + 240, path=self.db)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].report.report_id, found[0].report.description), (a.report_id, "Heavy traffic near junction"))
        self.assertFalse(hasattr(found[0].report, "submitter"), "user A's identity is not available to user B")


# ======================================================================================================== rate limits and storage rules
class SafetyTests(Base):
    def test_one_session_cannot_flood_the_service(self):
        for _ in range(photos.SESSION_LIMIT[0]):
            self.photo(session="flood")
        with self.assertRaises(photos.PhotoError) as cm:
            self.photo(session="flood")
        self.assertEqual(cm.exception.kind, "rate_limited")
        self.photo(session="somebody-else")                                   # another session is not affected
        self.photo(session="flood", now=T0 + photos.SESSION_LIMIT[1] + 1)    # and the limit ends with the window

    def test_the_whole_service_has_a_limit_too(self):
        for i in range(photos.GLOBAL_LIMIT[0]):
            self.photo(session=f"s{i}")
        with self.assertRaises(photos.PhotoError) as cm:
            self.photo(session="one-more")
        self.assertEqual(cm.exception.kind, "rate_limited")

    def test_the_same_picture_twice_within_its_fresh_window_is_refused(self):
        raw = make_image(color=(1, 77, 130))
        photos.submit_photo(raw, AREA_G[0], AREA_G[1], "a", session_id="A", accuracy_m=20, now=T0, path=self.db)
        with self.assertRaises(photos.PhotoError) as cm:
            photos.submit_photo(raw, AREA_G[0], AREA_G[1], "b", session_id="B", accuracy_m=20, now=T0 + 30, path=self.db)
        self.assertEqual(cm.exception.kind, "duplicate")
        photos.submit_photo(raw, AREA_G[0], AREA_G[1], "later", session_id="B", accuracy_m=20, now=T0 + 700, path=self.db)   # after it expired it is allowed

    def test_a_refused_report_leaves_no_files_behind(self):
        with self.assertRaises(photos.PhotoError):
            photos.submit_photo(b"nope", AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=20, now=T0, path=self.db)
        folder = self.db.parent / "photos"
        self.assertFalse(folder.exists() and any(folder.iterdir()))

    def test_a_storage_failure_is_reported_and_nothing_is_kept(self):
        with mock.patch.object(photos, "_write_atomic", side_effect=OSError("disk full")), self.assertRaises(photos.PhotoError) as cm:
            photos.submit_photo(make_image(), AREA_G[0], AREA_G[1], "x", session_id="A", accuracy_m=20, now=T0, path=self.db)
        self.assertEqual(cm.exception.kind, "storage")
        self.assertNotIn("disk full", cm.exception.message, "no internal details are shown")
        self.assertEqual(photos.history(path=self.db), [])

    def test_viewing_never_creates_or_migrates_anything(self):
        photos.nearby(*AREA_G, path=self.db)
        photos.history(path=self.db)
        photos.count_fresh(path=self.db)
        photos.purge(path=self.db)
        self.assertFalse(self.db.exists())
        self.assertFalse((self.db.parent / "photos").exists())

    def test_an_older_text_report_database_is_migrated_only_by_a_write_and_keeps_its_reports(self):
        store.submit("Accident", "Silk Board Junction", "An older text report that must survive.", session_id="s", path=self.db)
        with closing(sqlite3.connect(str(self.db))) as con:           # make it look like a database from before photos existed
            con.execute("DROP TABLE photo_reports")
            con.execute("PRAGMA user_version = 1")
        before = self.db.read_bytes()
        self.assertEqual(photos.nearby(*AREA_G, path=self.db), [])
        self.assertEqual(photos.history(path=self.db), [])
        self.assertEqual(store.count(path=self.db), 1)
        self.assertEqual(self.db.read_bytes(), before, "reading an old database does not change it")
        self.photo()                                           # the first photo (a write) migrates it
        with closing(sqlite3.connect(str(self.db))) as con:
            self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], store.SCHEMA_VERSION)
        self.assertEqual(store.count(path=self.db), 1, "the text report is still there")
        self.assertEqual(len(photos.history(path=self.db)), 1)


# ======================================================================================================== 14, 15: permissions, as the component reports them
class CaptureMessageTests(unittest.TestCase):
    OK = {"image": data_url(make_image()), "lat": 12.9159, "lon": 77.6240, "accuracy_m": 30.0, "loc_status": "ok", "cam_status": "ok"}

    def test_a_complete_capture_is_ready(self):
        info = photos.explain_capture(self.OK)
        self.assertTrue(info["ready"])
        self.assertEqual(info["messages"], [])

    def test_location_permission_denied_is_explained_and_blocks_submission(self):
        info = photos.explain_capture({**self.OK, "lat": None, "lon": None, "accuracy_m": None, "loc_status": "denied"})
        self.assertFalse(info["ready"])
        self.assertTrue(any("Location permission was denied" in m and "GPS position" in m for m in info["messages"]))

    def test_other_location_problems_are_explained(self):
        for status, words in (("insecure", "secure page"), ("unavailable", "could not determine"), ("timeout", "too long"), ("unsupported", "cannot give out")):
            info = photos.explain_capture({"loc_status": status})
            self.assertFalse(info["ready"])
            self.assertTrue(any(words in m for m in info["messages"]), status)

    def test_camera_permission_denied_is_explained_and_offers_the_alternative(self):
        info = photos.explain_capture({"cam_status": "camera_denied", "loc_status": "ok", "lat": 12.9, "lon": 77.6, "accuracy_m": 10})
        self.assertFalse(info["ready"], "no photo yet")
        self.assertTrue(any("Camera permission was denied" in m and "choose an existing photo" in m for m in info["messages"]))

    def test_a_missing_camera_and_a_bad_photo_are_explained(self):
        self.assertTrue(any("No usable camera" in m for m in photos.explain_capture({"cam_status": "camera_unavailable"})["messages"]))
        self.assertTrue(any("cannot be used" in m for m in photos.explain_capture({"image_error": "unsupported_image"})["messages"]))
        self.assertTrue(any("too large" in m for m in photos.explain_capture({"image_error": "image_too_large"})["messages"]))

    def test_a_photo_without_a_position_or_a_position_without_a_photo_is_not_ready(self):
        self.assertFalse(photos.explain_capture({**self.OK, "lat": None})["ready"])
        self.assertFalse(photos.explain_capture({**self.OK, "accuracy_m": None})["ready"])
        self.assertFalse(photos.explain_capture({**self.OK, "image": None})["ready"])
        self.assertFalse(photos.explain_capture({**self.OK, "image": "javascript:alert(1)"})["ready"])

    def test_nothing_malformed_can_break_it(self):
        for junk in (None, 5, "x", [], {"lat": "12.9", "lon": "77.6", "accuracy_m": "5", "image": 7}, {"loc_status": {"a": 1}}):
            with self.subTest(str(junk)[:20]):
                self.assertFalse(photos.explain_capture(junk)["ready"])


# ======================================================================================================== the Community Reports page
class CommunityPageTests(Base):
    READY = {"image": data_url(make_image(color=(20, 140, 60))), "lat": AREA_G[0], "lon": AREA_G[1], "accuracy_m": 35.0, "loc_status": "ok", "cam_status": "ok"}

    def run_page(self, value=None, steps=()):
        with mock.patch.object(photo_capture, "capture", return_value=value):
            at = AppTest.from_file(PAGE_COMMUNITY, default_timeout=120).run()
            for step in steps:
                step(at)
        return at

    def open_flow(self, at):
        at.button(key="cr_photo_open_btn").click().run()

    def test_the_photo_button_is_prominent_and_opens_the_flow_with_the_reason_for_location(self):
        at = self.run_page()
        self.assertIn("Report Traffic with Photo", [b.label for b in at.button])
        self.assertEqual([e.value for e in at.exception], [])
        with mock.patch.object(photo_capture, "capture", return_value=None):
            at.button(key="cr_photo_open_btn").click().run()
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Why location is required", text)
        self.assertIn("You cannot type a position in", text)
        self.assertTrue(at.button(key="cr_photo_submit").disabled, "no photo and no position yet")
        self.assertIn("Short description (optional)", [t.label for t in at.text_input])

    def test_submitting_a_ready_capture_saves_it_and_shows_it_as_fresh(self):
        at = self.run_page(self.READY)
        with mock.patch.object(photo_capture, "capture", return_value=self.READY):
            at.button(key="cr_photo_open_btn").click().run()
            self.assertFalse(at.button(key="cr_photo_submit").disabled)
            at.text_input(key="cr_photo_desc").set_value("Heavy traffic near junction").run()
            at.button(key="cr_photo_submit").click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("Saved: your photo report" in s.value for s in at.success), [s.value for s in at.success])
        stored = photos.history(path=self.db)
        self.assertEqual(len(stored), 1)
        self.assertEqual((stored[0].description, stored[0].latitude, stored[0].longitude), ("Heavy traffic near junction", AREA_G[0], AREA_G[1]))
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Heavy traffic near junction", text)
        self.assertIn("Uploaded just now", text)
        self.assertIn("Fresh", text)

    def test_location_permission_denied_shows_the_explanation_and_cannot_be_submitted(self):
        denied = {"image": self.READY["image"], "lat": None, "lon": None, "accuracy_m": None, "loc_status": "denied", "cam_status": "ok"}
        at = self.run_page(denied)
        with mock.patch.object(photo_capture, "capture", return_value=denied):
            at.button(key="cr_photo_open_btn").click().run()
        self.assertTrue(any("Location permission was denied" in w.value for w in at.warning))
        self.assertTrue(at.button(key="cr_photo_submit").disabled)
        self.assertEqual(photos.history(path=self.db), [])

    def test_camera_permission_denied_shows_the_explanation_and_the_alternative(self):
        denied = {"image": None, "lat": AREA_G[0], "lon": AREA_G[1], "accuracy_m": 20.0, "loc_status": "ok", "cam_status": "camera_denied"}
        at = self.run_page(denied)
        with mock.patch.object(photo_capture, "capture", return_value=denied):
            at.button(key="cr_photo_open_btn").click().run()
        self.assertTrue(any("Camera permission was denied" in w.value and "choose an existing photo" in w.value for w in at.warning))
        self.assertTrue(at.button(key="cr_photo_submit").disabled)

    def test_a_server_side_refusal_is_shown_and_nothing_is_stored(self):
        bad = {**self.READY, "lat": 28.61, "lon": 77.20}              # a position far outside the map area
        at = self.run_page(bad)
        with mock.patch.object(photo_capture, "capture", return_value=bad):
            at.button(key="cr_photo_open_btn").click().run()
            at.button(key="cr_photo_submit").click().run()
        self.assertTrue(any("outside the Bengaluru area" in e.value for e in at.error), [e.value for e in at.error])
        self.assertEqual(photos.history(path=self.db), [])

    def test_a_forged_payload_with_a_non_image_is_refused_by_the_server(self):
        forged = {**self.READY, "image": data_url(b"<script>alert(1)</script>", "image/jpeg")}
        at = self.run_page(forged)
        with mock.patch.object(photo_capture, "capture", return_value=forged):
            at.button(key="cr_photo_open_btn").click().run()
            at.button(key="cr_photo_submit").click().run()
        self.assertTrue(any("not a readable photo" in e.value for e in at.error), [e.value for e in at.error])
        self.assertEqual(photos.history(path=self.db), [])

    def test_the_page_lists_fresh_photos_and_keeps_expired_ones_in_the_history(self):
        fresh = self.photo(AREA_G, description="Fresh one", now=time.time() - 120)
        stale = self.photo(NORTH_08_KM, description="Stale one", now=time.time() - 1800, session="B")
        at = self.run_page()
        recent = " ".join(m.value for m in at.markdown)
        self.assertIn("Fresh one", recent)
        self.assertIn("Stale one", recent, "the history is on the page, inside its expander")
        self.assertIn("Photo history (1 expired)", [e.label for e in at.expander])
        self.assertNotIn("No recent community traffic reports.", recent)
        self.assertEqual({fresh.state(time.time()), stale.state(time.time())}, {"fresh", "expired"})

    def test_with_no_photos_the_page_says_so_and_invents_nothing(self):
        at = self.run_page()
        self.assertTrue(any("No recent community traffic reports" in i.value for i in at.info))
        self.assertEqual(len(at.get("imgs") if hasattr(at, "get") else []), 0)

    def test_a_description_cannot_inject_markup(self):
        self.photo(AREA_G, description="<script>alert(1)</script><img src=x onerror=alert(2)>", now=time.time() - 30)
        at = self.run_page()
        text = " ".join(m.value for m in at.markdown)
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertNotIn("<img src=x", text)
        self.assertIn("&lt;script&gt;", text)


# ======================================================================================================== 13 and the map: the Live Traffic Map
class LiveMapPhotoTests(Base):
    def harness(self, **kw):
        h = PageHarness(script=APP, **kw).start()
        self.addCleanup(h.stop)
        return h

    def test_user_b_searches_the_area_and_sees_user_as_photo_on_the_map_and_below_it(self):
        a = self.photo(NORTH_08_KM, session="user-A", description="Heavy traffic near junction", now=time.time() - 240)
        h = self.harness(key=KEY).search("silk board junction").choose(0)      # user B: a different session, searching the place
        markers = [m for m in h.specs[-1]["markers"] if m.get("kind") == "community"]
        self.assertEqual(len(markers), 1)
        popup = markers[0]["popup"]
        self.assertEqual(popup["description"], "Heavy traffic near junction")
        self.assertEqual(popup["uploaded"], "Uploaded 4 minutes ago")
        self.assertRegex(popup["distance"], r"^\d\.\d km from your searched location$")
        self.assertTrue(popup["image"].startswith("data:image/jpeg;base64,"))
        self.assertIn("Community photo", popup["source"])
        self.assertTrue(markers[0]["fresh"] and popup["fresh"])
        text = " ".join(m.value for m in h.at.markdown)
        self.assertIn("Heavy traffic near junction", text)
        self.assertIn("Uploaded 4 minutes ago", text)
        self.assertIn("km from your searched location", text)
        self.assertIn("Fresh", text)
        self.assertNotIn(a.report_id, text, "no internal id and no reporter are shown")

    def test_a_circle_shows_the_searched_area_with_the_configured_radius(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        circles = h.specs[-1]["circles"]
        self.assertEqual(len(circles), 1)
        self.assertEqual(circles[0]["radius_m"], 2000.0)

    def test_reports_outside_the_radius_do_not_appear(self):
        self.photo(NORTH_30_KM, description="Far away", now=time.time() - 60)        # 3 km from Area G
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertEqual([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"], [])
        self.assertNotIn("Far away", " ".join(m.value for m in h.at.markdown))
        self.assertTrue(any("No recent community traffic reports for this area." in i.value for i in h.at.info))

    def test_expired_reports_do_not_appear_as_current_traffic(self):
        self.photo(AREA_G, description="Too old", now=time.time() - 700)
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertEqual([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"], [])
        self.assertNotIn("Too old", " ".join(m.value for m in h.at.markdown))
        self.assertTrue(any("No recent community traffic reports for this area." in i.value for i in h.at.info))

    def test_an_area_with_no_reports_says_so(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertTrue(any(i.value == "No recent community traffic reports for this area." for i in h.at.info))

    def test_before_a_place_is_chosen_there_are_no_community_markers(self):
        self.photo(AREA_G, now=time.time() - 60)
        h = self.harness(key=KEY)
        self.assertEqual([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"], [])

    def test_the_community_layer_can_be_switched_off_without_touching_the_provider_layers(self):
        self.photo(AREA_G, now=time.time() - 60)
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertEqual(len(h.specs[-1]["lines"]), 1)
        h.at.button_group(key="lm_layers").set_value(["Traffic segment", "Selected place"]).run()
        self.assertEqual([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"], [])
        self.assertEqual(h.specs[-1]["circles"], [])
        self.assertEqual(len(h.specs[-1]["lines"]), 1)

    def test_community_photos_and_provider_traffic_stay_visibly_separate(self):
        self.photo(AREA_G, now=time.time() - 60)
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        spec = h.specs[-1]
        self.assertEqual(len(spec["lines"]), 1, "the provider reading is a coloured line")
        self.assertNotIn("popup", spec["lines"][0])
        kinds = {m.get("kind", "place") for m in spec["markers"]}
        self.assertEqual(kinds, {"place", "community"})
        caption = " ".join(c.value for c in h.at.caption)
        self.assertIn("Two different sources", caption)
        self.assertIn("TomTom traffic (provider)", caption)

    def test_the_radius_control_changes_what_is_found(self):
        self.photo(NORTH_30_KM, description="Three km away", now=time.time() - 60)
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertEqual([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"], [])
        h.at.slider(key="lm_photo_radius").set_value(4.0).run()
        self.assertEqual(len([m for m in h.specs[-1]["markers"] if m.get("kind") == "community"]), 1)

    def test_the_page_refreshes_the_photos_on_a_timer_without_reloading_the_application(self):
        self.assertEqual(photos.refresh_seconds(), 30)
        import inspect
        src = inspect.getsource(live_map)
        self.assertIn("@st.fragment(run_every=photos.refresh_seconds())", src)
        with mock.patch.dict(os.environ, {photos.ENV_REFRESH_S: "60"}):
            self.assertEqual(photos.refresh_seconds(), 60)

    def test_a_broken_photo_store_never_breaks_the_map(self):
        with mock.patch.object(photos, "nearby", side_effect=RuntimeError("boom")):
            h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertTrue(any("could not be read" in w.value for w in h.at.warning))


class MapMarkerInteractionTests(unittest.TestCase):
    """The popup of a community marker (the 'marker interaction'): its content, and that user text can never become markup. The real click in a browser
    is covered by tests/browser_check.py and the photo-report browser check."""

    def spec(self, description="Heavy traffic near junction"):
        item = photos.NearbyPhoto(photos.PhotoReport("r" * 32, T0, T0 + 600, 12.9231, 77.6240, 25.0, description, ""), 0.8)
        marker = photo_cards.marker_for(item, T0 + 240)
        marker["popup"]["image"] = data_url(make_image(size=(120, 90)))
        return leaflet_map.build_spec([12.9159, 77.624], 15, [marker], [], None, [{"lat": 12.9159, "lon": 77.624, "radius_m": 2000, "color": "#8B5CF6"}])

    def test_the_marker_carries_everything_the_popup_shows(self):
        marker = self.spec()["markers"][0]
        self.assertEqual(marker["kind"], "community")
        self.assertEqual(marker["popup"]["description"], "Heavy traffic near junction")
        self.assertEqual(marker["popup"]["uploaded"], "Uploaded 4 minutes ago")
        self.assertEqual(marker["popup"]["distance"], "0.8 km from your searched location")
        self.assertIn("Fresh", marker["popup"]["state"])
        self.assertIn("not verified", marker["popup"]["location"])

    def test_the_embedded_map_builds_the_popup_from_text_nodes_not_html(self):
        doc = leaflet_map.html_document(self.spec())
        self.assertIn("function popupFor", doc)
        self.assertIn("d.textContent = text", doc)
        self.assertIn("L.divIcon", doc)
        self.assertIn("bindPopup(popupFor", doc)
        popup_js = doc[doc.index("function popupFor"):doc.index("spec.markers.forEach")]
        self.assertNotIn("innerHTML", popup_js)
        self.assertIn("data:image/jpeg;base64,", popup_js, "only our own JPEG data URLs are used as an image source")

    def test_a_malicious_description_stays_data(self):
        evil = "</script><script>alert(1)</script><img src=x onerror=alert(2)>"
        doc = leaflet_map.html_document(self.spec(evil))
        self.assertNotIn("</script><script>alert(1)", doc, "the data cannot close the script tag")
        self.assertEqual(doc.count("<script>alert(1)"), 0)

    def test_an_expired_marker_is_drawn_differently_and_labelled(self):
        item = photos.NearbyPhoto(photos.PhotoReport("e" * 32, T0, T0 + 600, 12.9, 77.6, 10.0, "old", ""), 0.5)
        marker = photo_cards.marker_for(item, T0 + 900)
        self.assertFalse(marker["fresh"])
        self.assertIn("Expired", marker["popup"]["state"])


class ComponentFileTests(unittest.TestCase):
    def test_the_capture_component_asks_for_the_rear_camera_and_real_geolocation_only(self):
        html = (root_dir / "src" / "ui" / "components" / "photo_capture" / "index.html").read_text(encoding="utf-8")
        for needle in ('capture="environment"', 'facingMode: { ideal: "environment" }', "navigator.geolocation.getCurrentPosition", "enableHighAccuracy: true",
                       "NotAllowedError", "camera_denied", '"denied"', "streamlit:setComponentValue", "image/*"):
            self.assertIn(needle, html)
        self.assertNotIn('type="number"', html, "there is no way to type coordinates in")
        self.assertNotIn("latitude\" type=", html)
        for external in ('src="http', 'href="http', "fetch(", "XMLHttpRequest", "import(", "<link", "@import"):
            self.assertNotIn(external, html, "the component loads nothing from outside and sends nothing anywhere itself")

    def test_the_component_wrapper_is_registered(self):
        self.assertTrue((root_dir / "src" / "ui" / "components" / "photo_capture" / "index.html").is_file())
        self.assertTrue(callable(photo_capture.capture))


if __name__ == "__main__":
    unittest.main()

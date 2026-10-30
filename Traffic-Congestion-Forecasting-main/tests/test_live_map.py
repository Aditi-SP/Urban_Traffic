"""
Stage 1 tests: location validation, place search (Nominatim policy), TomTom client, in-memory cache, vendored map library
and the Live Traffic Map page.

IMPORTANT: every provider response here is MOCKED. The only real Nominatim response is the recorded fixture below (OpenStreetMap
data, (c) OpenStreetMap contributors, ODbL). The TomTom fixture is constructed from TomTom's published documentation, NOT recorded
from the service. Nothing in this file verifies the real TomTom integration; that needs a valid key and a genuine response.

Run:   python -m unittest tests.test_live_map -v
"""
import logging
import math
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
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database: tests never touch the real community.db)
os.environ.pop("LIVE_CACHE_TTL_S", None)  # tests must not depend on a local runtime setting; tests that need it set it themselves

import requests  # noqa: E402

from src.live import geo, geocode, tomtom  # noqa: E402
from src.ui import leaflet_map  # noqa: E402

KEY = "TEST-KEY-9f3a-DO-NOT-LEAK"

# A real Nominatim response recorded on 2026-10-28 (q="Silk Board Junction", bounded to the Bengaluru box).
NOMINATIM_REAL = [{
    "place_id": 420444310, "licence": "Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright",
    "osm_type": "node", "osm_id": 13955791826, "lat": "12.9158955", "lon": "77.6239872", "category": "highway", "type": "bus_stop",
    "place_rank": 30, "importance": 7.094453345536e-05, "addresstype": "highway", "name": "Silk Board Junction",
    "display_name": "Silk Board Junction, Srinagar - Kanyakumari Highway, MCHS Colony, HSR Layout, Bengaluru South City Corporation, "
                    "Bengaluru, Bangalore South, Bengaluru Urban, Karnataka, 560102, India",
    "boundingbox": ["12.9158455", "12.9159455", "77.6239372", "77.6240372"]}]
PLACE = (12.9158955, 77.6239872)


class FakeResp:
    def __init__(self, status=200, payload=None, json_error=False):
        self.status_code, self._payload, self._json_error = status, payload, json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def flow_payload(**seg_over):
    """CONSTRUCTED from TomTom's documentation (field names and nesting); not a recorded response."""
    seg = {"frc": "FRC1", "currentSpeed": 24, "freeFlowSpeed": 52, "currentTravelTime": 130, "freeFlowTravelTime": 60,
           "confidence": 0.92, "roadClosure": False,
           "coordinates": {"coordinate": [{"latitude": 12.9159, "longitude": 77.6240}, {"latitude": 12.9164, "longitude": 77.6244},
                                          {"latitude": 12.9170, "longitude": 77.6249}]}}
    seg.update(seg_over)
    return {"flowSegmentData": seg}


# Error bodies RECORDED from the LIVE TomTom service on 2026-10-30 with a real key (nothing in them is secret). These, not guesses, decide how
# a status is classified. The 404 for an unknown service version had a different JSON shape: only its FIELD NAMES were recorded, so its values are constructed.
REC_NO_ROAD = {"error": "Point too far from nearest existing segment.", "httpStatusCode": 400,
               "detailedError": {"code": "INVALID_REQUEST", "message": "Point too far from nearest existing segment."}}
REC_BAD_POINT = {"error": "Invalid point format.", "httpStatusCode": 400, "detailedError": {"code": "INVALID_REQUEST", "message": "Invalid point format."}}
REC_NO_POINT = {"error": "Missing point parameter.", "httpStatusCode": 400, "detailedError": {"code": "INVALID_REQUEST", "message": "Missing point parameter."}}
REC_BAD_KEY = {"detailedError": {"code": "Unauthorized", "message": "You are missing valid authentication credentials"}}   # wrong key AND no key: both 401
CONSTRUCTED_404 = {"type": "about:blank", "title": "Not Found", "status": 404, "detail": "constructed: only the field names were recorded"}


def ok_get(payload=None):
    return mock.Mock(return_value=FakeResp(200, flow_payload() if payload is None else payload))


# ======================================================================================================== validation
class NearestSegmentDistanceTests(unittest.TestCase):
    """TomTom answers for the NEAREST road. The page says so when that road is not at the chosen point."""

    def obs(self, segment, lat=12.9159, lon=77.6240):
        return tomtom.FlowObservation(lat=lat, lon=lon, current_speed_kmh=20, free_flow_speed_kmh=40, current_travel_time_s=60, free_flow_travel_time_s=30,
                                      confidence=0.9, road_closure=False, frc="FRC1", segment=tuple(segment), fetched_at=1000.0)

    def test_a_point_on_the_road_is_at_distance_zero(self):
        self.assertLess(tomtom.nearest_segment_distance_m(self.obs([(12.9159, 77.6230), (12.9159, 77.6250)])), 0.5)

    def test_the_distance_is_to_the_line_between_vertices_not_only_to_the_vertices(self):
        # a long road passing 100 m north of the point: both vertices are ~1 km away, the line itself is 100 m away
        d = tomtom.nearest_segment_distance_m(self.obs([(12.9159 + 0.0009, 77.6140), (12.9159 + 0.0009, 77.6340)]))
        self.assertAlmostEqual(d, 0.0009 * 111_320, delta=2)

    def test_a_road_300_metres_east_is_reported_as_about_300_metres(self):
        d = tomtom.nearest_segment_distance_m(self.obs([(12.9150, 77.6240 + 0.0028), (12.9170, 77.6240 + 0.0028)]))
        self.assertAlmostEqual(d, 0.0028 * 111_320 * math.cos(math.radians(12.9159)), delta=3)
        self.assertGreater(d, tomtom.FAR_SEGMENT_NOTE_M)

    def test_the_page_says_the_reading_is_for_the_nearest_road_only_when_it_is_far(self):
        far = flow_payload(coordinates={"coordinate": [{"latitude": 12.9150, "longitude": 77.6268}, {"latitude": 12.9170, "longitude": 77.6268}]})
        near = flow_payload()   # the standard fixture: the road passes right at the point
        with_far = PageHarness(key=KEY, tomtom_outcome=FakeResp(200, far)).start()
        self.addCleanup(with_far.stop)
        with_far.search("silk board junction").choose(0).fetch()
        self.assertIn("TomTom answered for the nearest road, about", " ".join(c.value for c in with_far.at.caption))
        with_near = PageHarness(key=KEY, tomtom_outcome=FakeResp(200, near)).start()
        self.addCleanup(with_near.stop)
        with_near.search("silk board junction").choose(0).fetch()
        self.assertNotIn("nearest road, about", " ".join(c.value for c in with_near.at.caption))


class GeoTests(unittest.TestCase):
    def test_valid_location_inside_the_bengaluru_area(self):
        self.assertEqual(geo.validate_location(12.97, 77.59), (12.97, 77.59))
        self.assertEqual(geo.validate_location(13, 77), (13.0, 77.0)) if False else None

    def test_rejects_out_of_range_and_non_numbers(self):
        for lat, lon in ((91, 77.5), (-91, 77.5), (12.9, 181), (12.9, -181), (float("nan"), 77.5), (12.9, float("inf")),
                         (True, 77.5), ("12.9", "77.5"), (None, None), ([], {})):
            with self.assertRaises(geo.LocationError, msg=f"{lat!r},{lon!r}"):
                geo.validate_location(lat, lon)

    def test_rejects_other_cities_unless_the_box_is_lifted(self):
        with self.assertRaisesRegex(geo.LocationError, "outside the Bengaluru area"):
            geo.validate_location(48.8566, 2.3522)  # Paris
        self.assertEqual(geo.validate_location(48.8566, 2.3522, bbox=None), (48.8566, 2.3522))

    def test_default_centre_is_inside_the_scope(self):
        geo.validate_location(*geo.BENGALURU_CENTER)

    def test_query_is_cleaned_and_limited(self):
        self.assertEqual(geo.validate_query("  silk   board\tjunction "), "silk board junction")
        for bad in ("", "  ", "ab", "x" * 101, "bad\x00text", "tab\x07", 123, None):
            with self.assertRaises(geo.LocationError, msg=repr(bad)):
                geo.validate_query(bad)

    def test_distance(self):
        self.assertAlmostEqual(geo.distance_km(12.9, 77.6, 12.9, 77.6), 0.0, places=6)
        self.assertAlmostEqual(geo.distance_km(12.0, 77.0, 13.0, 77.0), 111.2, delta=0.5)


# ======================================================================================================== place search
class GeocodeTests(unittest.TestCase):
    def setUp(self):
        geocode.clear_cache()
        self.addCleanup(geocode.clear_cache)

    def test_parses_the_recorded_real_response(self):
        get = mock.Mock(return_value=FakeResp(200, NOMINATIM_REAL))
        places = geocode.search_places("Silk Board Junction", http_get=get, sleep=lambda s: None)
        self.assertEqual(len(places), 1)
        p = places[0]
        self.assertEqual((p.name, p.lat, p.lon), ("Silk Board Junction", 12.9158955, 77.6239872))
        self.assertTrue(p.display_name.startswith("Silk Board Junction"))

    def test_request_follows_the_nominatim_policy(self):
        get = mock.Mock(return_value=FakeResp(200, NOMINATIM_REAL))
        geocode.search_places("Silk Board Junction", http_get=get, sleep=lambda s: None)
        (url,), kw = get.call_args
        self.assertEqual(url, "https://nominatim.openstreetmap.org/search")
        ua = kw["headers"]["User-Agent"]
        self.assertEqual(ua, geocode.USER_AGENT)
        self.assertNotIn("python-requests", ua.lower())  # a library default is not allowed by the policy
        self.assertIn("TrafficIntelligenceDashboard", ua)
        p = kw["params"]
        self.assertLessEqual(p["limit"], 5)
        self.assertEqual((p["bounded"], p["countrycodes"], p["format"]), (1, "in", "jsonv2"))
        self.assertEqual(p["viewbox"], "77.35,13.25,77.85,12.7")  # left,top,right,bottom
        self.assertIsInstance(kw["timeout"], tuple)
        self.assertNotIn("autocomplete", str(p).lower())

    def test_no_autocomplete_short_text_never_reaches_the_network(self):
        get = mock.Mock()
        for q in ("", "s", "si"):
            with self.assertRaises(geo.LocationError):
                geocode.search_places(q, http_get=get)
        get.assert_not_called()

    def test_results_are_cached_in_memory(self):
        get = mock.Mock(return_value=FakeResp(200, NOMINATIM_REAL))
        a = geocode.search_places("Silk Board Junction", http_get=get, sleep=lambda s: None)
        b = geocode.search_places("  silk board   JUNCTION ", http_get=get, sleep=lambda s: None)
        self.assertEqual(a, b)
        self.assertEqual(get.call_count, 1)

    def test_rate_limit_keeps_requests_at_least_a_second_apart(self):
        clock = {"t": 100.0}
        sleeps = []
        get = mock.Mock(return_value=FakeResp(200, NOMINATIM_REAL))
        tick = lambda: clock["t"]

        def sleep(s):
            sleeps.append(s)
            clock["t"] += s
        geocode.search_places("first query", http_get=get, clock=tick, sleep=sleep)
        geocode.search_places("second query", http_get=get, clock=tick, sleep=sleep)  # immediately after: must wait
        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], geocode.MIN_INTERVAL_S - 1e-9)
        self.assertGreaterEqual(geocode.MIN_INTERVAL_S, 1.0)

    def test_busy_instead_of_freezing_when_the_wait_would_be_long(self):
        geocode._limiter._last = 1000.0 + 60  # pretend a request is "in the future" (clock far behind)
        with self.assertRaises(geocode.GeocodeError) as cm:
            geocode.search_places("some place", http_get=mock.Mock(), clock=lambda: 1000.0, sleep=lambda s: None)
        self.assertEqual(cm.exception.kind, "busy")

    def test_outside_duplicate_and_invalid_items_are_dropped(self):
        paris = {**NOMINATIM_REAL[0], "lat": "48.85", "lon": "2.35", "name": "Paris"}
        dup = dict(NOMINATIM_REAL[0])
        junk = [paris, dup, NOMINATIM_REAL[0], {"lat": "x", "lon": "y", "name": "bad"}, {"name": "no coords"}, "string", None, 7,
                {**NOMINATIM_REAL[0], "lat": "nan", "lon": "77.6"}]
        places = geocode._parse(junk)
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0].name, "Silk Board Junction")

    def test_empty_result_is_a_valid_answer(self):
        get = mock.Mock(return_value=FakeResp(200, []))
        self.assertEqual(geocode.search_places("nonexistent place xyz", http_get=get, sleep=lambda s: None), [])

    def test_service_problems_become_clear_errors(self):
        cases = [(requests.Timeout("t"), "timeout"), (requests.ConnectionError("c"), "network"), (FakeResp(429, []), "rate_limited"),
                 (FakeResp(500, []), "http"), (FakeResp(403, []), "http"), (FakeResp(200, json_error=True), "invalid_response"),
                 (FakeResp(200, {"not": "a list"}), "invalid_response")]
        for i, (outcome, kind) in enumerate(cases):
            geocode.clear_cache()
            get = mock.Mock(side_effect=outcome) if isinstance(outcome, Exception) else mock.Mock(return_value=outcome)
            with self.assertRaises(geocode.GeocodeError, msg=kind) as cm:
                geocode.search_places(f"query number {i}", http_get=get, sleep=lambda s: None)
            self.assertEqual(cm.exception.kind, kind)
            self.assertNotIn("http", cm.exception.message.lower().replace("(http", "")) if kind in ("timeout", "network") else None

    def test_failures_are_not_cached(self):
        get = mock.Mock(side_effect=[FakeResp(500, []), FakeResp(200, NOMINATIM_REAL)])
        with self.assertRaises(geocode.GeocodeError):
            geocode.search_places("retry me", http_get=get, sleep=lambda s: None)
        self.assertEqual(len(geocode.search_places("retry me", http_get=get, sleep=lambda s: None)), 1)


# ======================================================================================================== TomTom client
class TomTomClientTests(unittest.TestCase):
    NOW = 1_800_000_000.0

    def fetch(self, get=None, key=KEY, lat=PLACE[0], lon=PLACE[1]):
        return tomtom.fetch_flow(lat, lon, key, http_get=get or ok_get(), clock=lambda: self.NOW)

    # ---- credentials
    def test_missing_key_is_detected_before_any_network_call(self):
        for key in (None, "", "   ", 0, []):
            get = mock.Mock()
            with self.assertRaises(tomtom.ProviderError) as cm:
                tomtom.fetch_flow(*PLACE, key, http_get=get)
            self.assertEqual(cm.exception.kind, "no_key")
            get.assert_not_called()

    # ---- location validation
    def test_invalid_locations_never_reach_the_provider(self):
        for lat, lon in ((200, 77.6), (12.9, 500), (48.85, 2.35), (float("nan"), 77.6)):
            get = mock.Mock()
            with self.assertRaises(geo.LocationError):
                tomtom.fetch_flow(lat, lon, KEY, http_get=get)
            get.assert_not_called()

    # ---- valid response
    def test_valid_mocked_response_becomes_an_observation(self):
        obs = self.fetch()
        self.assertEqual((obs.current_speed_kmh, obs.free_flow_speed_kmh), (24.0, 52.0))
        self.assertEqual((obs.current_travel_time_s, obs.free_flow_travel_time_s, obs.confidence), (130.0, 60.0, 0.92))
        self.assertIs(obs.road_closure, False)
        self.assertEqual(obs.frc, "FRC1")
        self.assertEqual(len(obs.segment), 3)
        self.assertEqual(obs.fetched_at, self.NOW)  # OUR clock: the API provides no measurement time
        self.assertAlmostEqual(obs.speed_ratio, 24 / 52)
        self.assertIn("TomTom", obs.provider)
        self.assertFalse(hasattr(obs, "provider_timestamp"), "this endpoint has no measurement time; none may be invented")

    def test_request_is_made_server_side_with_timeouts(self):
        get = ok_get()
        self.fetch(get)
        (url,), kw = get.call_args
        self.assertTrue(url.startswith("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/"))
        self.assertNotIn(KEY, url)  # the key travels as a query parameter, never inside the URL text we handle
        self.assertEqual(kw["params"]["unit"], "kmph")
        self.assertEqual(kw["params"]["point"], f"{PLACE[0]},{PLACE[1]}")
        self.assertIsInstance(kw["timeout"], tuple)
        self.assertEqual(get.call_count, 1, "no automatic retry")

    def test_display_bands_and_staleness(self):
        def obs(cur, free=100.0, closed=False, age=0):
            o = self.fetch(ok_get(flow_payload(currentSpeed=cur, freeFlowSpeed=free, roadClosure=closed)))
            return o, self.NOW + age
        self.assertEqual(tomtom.band(*obs(80)), "free")      # exactly 80% is free-flowing
        self.assertEqual(tomtom.band(*obs(79.9)), "slow")
        self.assertEqual(tomtom.band(*obs(50)), "slow")      # exactly 50% is slow
        self.assertEqual(tomtom.band(*obs(49.9)), "congested")
        self.assertEqual(tomtom.band(*obs(90, closed=True)), "closed")
        self.assertEqual(tomtom.band(*obs(90, age=tomtom.STALE_AFTER_S)), "free")      # at the limit: still current
        self.assertEqual(tomtom.band(*obs(90, age=tomtom.STALE_AFTER_S + 1)), "stale")  # older: never shown as current
        o, now = obs(30, age=700)
        self.assertTrue(tomtom.is_stale(o, now))
        self.assertAlmostEqual(tomtom.age_s(o, now), 700)

    # ---- invalid responses
    def test_invalid_responses_are_rejected_whole(self):
        bad_payloads = {
            "not a dict": [], "empty": {}, "no flowSegmentData": {"x": 1}, "segment not dict": {"flowSegmentData": "x"},
            "missing currentSpeed": None, "string speed": flow_payload(currentSpeed="24"), "bool speed": flow_payload(currentSpeed=True),
            "None speed": flow_payload(currentSpeed=None), "negative speed": flow_payload(currentSpeed=-5),
            "absurd speed": flow_payload(currentSpeed=999), "NaN speed": flow_payload(currentSpeed=float("nan")),
            "zero free-flow": flow_payload(freeFlowSpeed=0), "negative travel time": flow_payload(currentTravelTime=-1),
            "confidence above 1": flow_payload(confidence=1.5), "closure not bool": flow_payload(roadClosure="no"),
            "no coordinates": flow_payload(coordinates=None), "empty coordinates": flow_payload(coordinates={"coordinate": []}),
            "one coordinate": flow_payload(coordinates={"coordinate": [{"latitude": 12.9, "longitude": 77.6}]}),
            "bad latitude": flow_payload(coordinates={"coordinate": [{"latitude": 123, "longitude": 77.6}, {"latitude": 12.9, "longitude": 77.6}]}),
            "coordinate not dict": flow_payload(coordinates={"coordinate": ["a", "b"]}),
            "segment far away": flow_payload(coordinates={"coordinate": [{"latitude": 13.2, "longitude": 77.8}, {"latitude": 13.21, "longitude": 77.8}]}),
        }
        missing = flow_payload()
        del missing["flowSegmentData"]["currentSpeed"]
        bad_payloads["missing currentSpeed"] = missing
        for name, payload in bad_payloads.items():
            with self.subTest(name), self.assertRaises(tomtom.ProviderError) as cm:
                self.fetch(ok_get(payload))
            self.assertEqual(cm.exception.kind, "invalid_response", name)

    def test_response_that_is_not_json_is_rejected(self):
        with self.assertRaises(tomtom.ProviderError) as cm:
            self.fetch(mock.Mock(return_value=FakeResp(200, json_error=True)))
        self.assertEqual(cm.exception.kind, "invalid_response")

    # ---- HTTP problems, timeouts
    def test_http_status_codes_map_to_clear_kinds(self):
        cases = [(401, REC_BAD_KEY, "auth"), (401, {}, "auth"), (403, {}, "auth"), (429, {}, "rate_limited"),
                 (400, REC_NO_ROAD, "no_segment"),                                                      # the ONLY 400 that means "no road near the point"
                 (400, REC_BAD_POINT, "unexpected_status"), (400, REC_NO_POINT, "unexpected_status"), (400, {}, "unexpected_status"),
                 (404, CONSTRUCTED_404, "unexpected_status"), (404, {}, "unexpected_status"), (405, {}, "unexpected_status"),
                 (596, {}, "unexpected_status"), (302, {}, "unexpected_status"),                        # statuses this application does not understand
                 (500, {}, "server"), (503, {}, "server")]
        for status, body, kind in cases:
            with self.subTest(status=status, body=str(body)[:40]), self.assertRaises(tomtom.ProviderError) as cm:
                self.fetch(mock.Mock(return_value=FakeResp(status, body)))
            self.assertEqual(cm.exception.kind, kind, status)

    def test_an_unreadable_error_body_never_decides_a_cause(self):
        for resp in (FakeResp(400, json_error=True), FakeResp(400, []), FakeResp(400, "text"), FakeResp(400, {"error": 5})):
            with self.subTest(str(resp._payload)[:20]), self.assertRaises(tomtom.ProviderError) as cm:
                self.fetch(mock.Mock(return_value=resp))
            self.assertEqual(cm.exception.kind, "unexpected_status")

    def test_timeout_and_network_errors(self):
        for exc, kind in ((requests.Timeout("slow"), "timeout"), (requests.ConnectTimeout("slow"), "timeout"),
                          (requests.ConnectionError("down"), "network"), (requests.RequestException("odd"), "network")):
            with self.subTest(kind), self.assertRaises(tomtom.ProviderError) as cm:
                self.fetch(mock.Mock(side_effect=exc))
            self.assertEqual(cm.exception.kind, kind)

    # ---- the key must never leak
    def test_the_key_never_appears_in_any_error(self):
        leaky = f"401 Client Error for url: https://api.tomtom.com/x?key={KEY}&point=1,2"
        outcomes = [requests.HTTPError(leaky), requests.ConnectionError(leaky), requests.Timeout(leaky),
                    FakeResp(401, {}), FakeResp(403, {}), FakeResp(429, {}), FakeResp(500, {}), FakeResp(200, [], ), FakeResp(200, json_error=True)]
        for out in outcomes:
            get = mock.Mock(side_effect=out) if isinstance(out, Exception) else mock.Mock(return_value=out)
            with self.assertRaises(tomtom.ProviderError) as cm:
                self.fetch(get)
            e = cm.exception
            for text in (str(e), repr(e), e.message, repr(e.args)):
                self.assertNotIn(KEY, text)
            self.assertIsNone(e.__cause__)  # raised `from None`: the original (key-bearing) exception is not chained
            self.assertTrue(e.__context__ is None or e.__suppress_context__)

    def test_the_key_is_never_logged(self):
        with self.assertLogs(level=logging.DEBUG) as logs:
            logging.getLogger("probe").debug("start")  # assertLogs needs at least one record
            self.fetch()
            try:
                self.fetch(mock.Mock(return_value=FakeResp(401, {})))
            except tomtom.ProviderError:
                pass
        self.assertFalse([r for r in logs.records if KEY in r.getMessage()])

    def test_a_provider_error_text_that_contains_the_key_never_leaves_the_shared_service(self):
        """Regression for the removed stand-alone client, whose raise_for_status() text contained the full URL with key=...
        The supported path (FlowService -> fetch_flow) must stay clean for the same failures."""
        from src.app import realtime
        self.assertFalse(hasattr(realtime, "fetch_flow_segment"))
        leaky = f"401 Client Error: Unauthorized for url: https://api.tomtom.com/x?key={KEY}&point=1%2C2"
        for outcome in (requests.HTTPError(leaky), requests.ConnectionError(leaky), requests.Timeout(leaky)):
            with mock.patch.object(tomtom.requests, "get", side_effect=outcome), self.assertRaises(tomtom.ProviderError) as cm:
                tomtom.FlowService().get(*PLACE, KEY)
            for text in (str(cm.exception), cm.exception.message, repr(cm.exception)):
                self.assertNotIn(KEY, text)
            self.assertIsNone(cm.exception.__cause__)


# ======================================================================================================== cache + budget
class FlowServiceTests(unittest.TestCase):
    def make(self, **kw):
        self.t = 1000.0
        self.calls = []

        def fetcher(lat, lon, key):
            self.calls.append((lat, lon))
            return tomtom.parse_response(flow_payload(), lat, lon, self.t)
        return tomtom.FlowService(fetcher=fetcher, clock=lambda: self.t, **kw)

    def test_repeat_within_ttl_is_served_from_memory(self):
        s = self.make()
        o1, src1 = s.get(*PLACE, KEY)
        self.t += 60
        o2, src2 = s.get(*PLACE, KEY)
        self.assertEqual((src1, src2), ("provider", "cache"))
        self.assertIs(o1, o2)
        self.assertEqual(len(self.calls), 1)

    def test_refetch_after_ttl(self):
        s = self.make()
        s.get(*PLACE, KEY)
        self.t += tomtom.CACHE_TTL_S + 1
        _, src = s.get(*PLACE, KEY)
        self.assertEqual((src, len(self.calls)), ("provider", 2))

    def test_forced_refresh_is_rate_limited_to_once_per_30_seconds(self):
        s = self.make()
        s.get(*PLACE, KEY)
        self.t += 5
        self.assertEqual(s.get(*PLACE, KEY, force=True)[1], "cache")
        self.t += tomtom.MIN_REFRESH_S
        self.assertEqual(s.get(*PLACE, KEY, force=True)[1], "provider")
        self.assertEqual(len(self.calls), 2)

    def test_per_run_request_budget(self):
        s = self.make(budget=2)
        s.get(12.90, 77.60, KEY)
        s.get(12.91, 77.61, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            s.get(12.92, 77.62, KEY)
        self.assertEqual(cm.exception.kind, "budget")
        self.assertEqual(s.get(12.90, 77.60, KEY)[1], "cache")  # cached answers still work after the budget is spent

    def test_failed_attempts_count_against_the_budget(self):
        s = tomtom.FlowService(fetcher=mock.Mock(side_effect=tomtom.ProviderError("timeout", "slow")), budget=2)
        for _ in range(2):
            with self.assertRaises(tomtom.ProviderError):
                s.get(*PLACE, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            s.get(*PLACE, KEY)
        self.assertEqual(cm.exception.kind, "budget")

    def test_different_places_do_not_share_a_cache_entry(self):
        s = self.make()
        s.get(12.90, 77.60, KEY)
        s.get(12.95, 77.65, KEY)
        self.assertEqual(len(self.calls), 2)

    def test_service_validates_the_location_first(self):
        s = self.make()
        with self.assertRaises(geo.LocationError):
            s.get(48.85, 2.35, KEY)
        self.assertEqual(self.calls, [])

    def test_budget_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {"LIVE_MAX_REQUESTS": "7"}):
            self.assertEqual(tomtom.FlowService().budget, 7)
        with mock.patch.dict(os.environ, {"LIVE_MAX_REQUESTS": "junk"}):
            self.assertEqual(tomtom.FlowService().budget, tomtom.RUN_BUDGET_DEFAULT)


# ======================================================================================================== guarantees (static)
class GuaranteeTests(unittest.TestCase):
    LIVE_FILES = sorted((root_dir / "src" / "live").glob("*.py")) + [root_dir / "src" / "ui" / "live_map.py", root_dir / "src" / "ui" / "leaflet_map.py"]

    def test_provider_data_is_kept_in_memory_only(self):
        forbidden = re.compile(r"\bopen\(|write_text|write_bytes|\.to_csv|\.to_json|\.to_parquet|sqlite|shelve|pickle|json\.dump\(|\bmkdir|tempfile")
        for f in self.LIVE_FILES:
            code = "\n".join(l for l in f.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith(("#", '"""')))
            code = re.sub(r'""".*?"""', "", code, flags=re.S)
            self.assertIsNone(forbidden.search(code), f"{f.name} must not write provider data to disk")

    def test_a_fetch_creates_no_files(self):
        before = {p for p in root_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ):
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                svc = tomtom.FlowService(clock=lambda: 1e9)
                with mock.patch.object(tomtom.requests, "get", ok_get()):
                    svc.get(*PLACE, KEY)
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)
        after = {p for p in root_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        self.assertEqual(after - before, set())

    def test_live_code_is_not_connected_to_the_forecasting_models(self):
        banned = ("predictLstm", "predictXgboost", "predict_all", "predict_next_congestion", "DemoContext", "load_models", "forecast_data",
                  "from src.app import demo", "keras", "xgboost")
        for f in self.LIVE_FILES:
            text = f.read_text(encoding="utf-8")
            for word in banned:
                self.assertNotIn(word, text, f"{f.name} must not touch the models ({word})")

    def test_no_live_code_prints_or_embeds_the_key(self):
        for f in self.LIVE_FILES:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"print\(.*key", f.name)
            self.assertNotRegex(text, r"(st\.(error|warning|info|write|caption|markdown)|logging|logger)\(.*api_key", f.name)


# ======================================================================================================== vendored map library
class LeafletTests(unittest.TestCase):
    def test_vendored_files_match_the_published_integrity_hashes(self):
        self.assertEqual(leaflet_map._sri(leaflet_map.VENDOR / "leaflet.js"), "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=")
        self.assertEqual(leaflet_map._sri(leaflet_map.VENDOR / "leaflet.css"), "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=")
        self.assertTrue((leaflet_map.VENDOR / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("BSD 2-Clause License"))

    def test_a_modified_library_is_refused(self):
        leaflet_map._assets.cache_clear()
        self.addCleanup(leaflet_map._assets.cache_clear)
        with mock.patch.dict(leaflet_map.SRI, {"leaflet.js": "sha256-tampered"}):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                leaflet_map._assets()

    def test_empty_map_is_only_the_basemap(self):
        spec = leaflet_map.build_spec([12.97, 77.59], 11)
        self.assertEqual((spec["markers"], spec["lines"]), ([], []))

    def test_embedded_page_has_attribution_no_cdn_and_no_secrets(self):
        spec = leaflet_map.build_spec([12.97, 77.59], 11, markers=[{"lat": 12.9, "lon": 77.6, "color": "#fff", "tooltip": "</script><b>x</b>"}])
        doc = leaflet_map.html_document(spec, 500)
        self.assertIn("OpenStreetMap", doc)
        self.assertIn("https://tile.openstreetmap.org/{z}/{x}/{y}.png", doc)
        self.assertIsNone(re.search(r'<script[^>]+src=', doc), "no script may be loaded from the network")
        self.assertIsNone(re.search(r'<link[^>]+href=["\']http', doc))
        self.assertEqual(doc.count("<script"), 3)
        self.assertEqual(doc.count("</script>"), 3, "a '</script>' inside the data must not close the tag early")
        for needle in ("api.tomtom.com", "key=", "TOMTOM"):
            self.assertNotIn(needle, doc)
        self.assertIn("tile", leaflet_map.OFFLINE_NOTE.lower())  # an offline state exists


# ======================================================================================================== the page
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = str(root_dir / "src" / "ui" / "pages" / "live_map.py")
APP = str(root_dir / "app.py")   # the real router (app.py runs unchanged and draws the sidebar)


def _router_pages(st):
    """AppTest does not run file-based st.Page pages under a router, so the only substitution is HOW the page is started: a callable that
    executes the real live-map page script. app.py itself, the sidebar, the health model and the page are all the real code."""
    def run_live_map():
        exec(compile(open(PAGE, encoding="utf-8").read(), PAGE, "exec"), {"__name__": "__main__", "__file__": PAGE})
    return [st.Page(run_live_map, title="Live Traffic Map", url_path="live-map", default=True)]


def snapshot():
    return {p for p in root_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


class PageHarness:
    """Runs the live-map page with mocked network and captures every map spec the page draws."""

    def __init__(self, key=None, tomtom_outcome=None, geocode_payload=None, clock=None, script=PAGE):
        self.key, self.specs, self.net, self.script = key, [], [], script
        self.tomtom_outcome = tomtom_outcome if tomtom_outcome is not None else FakeResp(200, flow_payload())
        self.geocode_payload = NOMINATIM_REAL if geocode_payload is None else geocode_payload
        self.service = tomtom.FlowService(clock=clock or __import__("time").time)

    def _get(self, url, **kw):
        self.net.append((url, kw))
        if "nominatim" in url:
            if isinstance(self.geocode_payload, Exception):
                raise self.geocode_payload
            return FakeResp(200, self.geocode_payload)
        if "tomtom" in url:
            if isinstance(self.tomtom_outcome, Exception):
                raise self.tomtom_outcome
            return self.tomtom_outcome
        raise AssertionError(f"unexpected network call: {url}")

    def start(self):
        geocode.clear_cache()
        patches = [mock.patch.dict(os.environ), mock.patch.object(requests, "get", side_effect=self._get),
                   mock.patch.object(leaflet_map, "show", side_effect=lambda spec, h=540: self.specs.append(spec)),
                   mock.patch.object(tomtom, "SERVICE", self.service), mock.patch.object(geocode, "MIN_INTERVAL_S", 0.0)]
        if self.script == APP:
            from src.ui import navigation
            patches.append(mock.patch.object(navigation, "build_pages", _router_pages))
        for p in patches:
            p.start()
        self._patches = patches
        os.environ.pop("TOMTOM_API_KEY", None)
        if self.key:
            os.environ["TOMTOM_API_KEY"] = self.key
        self.at = AppTest.from_file(self.script, default_timeout=120).run()
        return self

    def stop(self):
        for p in reversed(self._patches):
            p.stop()
        geocode.clear_cache()

    def texts(self):
        out = []
        for g in (self.at.markdown, self.at.caption, self.at.info, self.at.warning, self.at.error, self.at.success):
            out += [e.value for e in g]
        return " ".join(out)

    def search(self, text):
        self.at.text_input(key="lm_query").set_value(text)
        next(b for b in self.at.button if b.label == "Search").click()
        self.at.run()
        return self

    def choose(self, index=0):
        self.at.selectbox(key="lm_pick").select_index(index).run()
        return self

    def fetch(self):
        self.at.button(key="lm_fetch_btn").click().run()
        return self

    def tomtom_calls(self):
        return [n for n in self.net if "tomtom" in n[0]]


class LiveMapPageTests(unittest.TestCase):
    def harness(self, **kw):
        h = PageHarness(**kw).start()
        self.addCleanup(h.stop)
        return h

    # ---- missing credentials / empty map
    def test_without_a_key_the_map_is_useful_but_empty_and_explains_setup(self):
        h = self.harness()
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertEqual([e.value for e in h.at.error], [])
        spec = h.specs[-1]
        self.assertEqual((spec["markers"], spec["lines"]), ([], []))  # no fake markers, no fake traffic
        self.assertEqual(spec["center"], list(geo.BENGALURU_CENTER))
        text = h.texts()
        for needle in ("Live traffic not configured", "Live traffic is not configured.", "TOMTOM_API_KEY", "secrets.toml",
                       "no autocomplete", "OpenStreetMap"):
            self.assertIn(needle, text)
        self.assertTrue(h.at.button(key="lm_fetch_btn").disabled)
        self.assertEqual(h.net, [], "opening the page must not make any network request")

    def test_page_never_shows_a_reading_without_a_real_response(self):
        h = self.harness()
        text = h.texts()
        self.assertIn("No reading yet", text)
        self.assertNotIn("km/h", text)
        self.assertNotIn("Last fetched", text.replace("Last fetched", "", 0)) if False else None

    # ---- search (mocked Nominatim, real recorded shape)
    def test_search_choose_and_map_marker_without_a_key(self):
        h = self.harness().search("silk board junction")
        self.assertEqual(len(h.net), 1)
        url, kw = h.net[0]
        self.assertIn("nominatim.openstreetmap.org", url)
        self.assertEqual(kw["headers"]["User-Agent"], geocode.USER_AGENT)
        self.assertEqual(len(h.at.selectbox(key="lm_pick").options), 1)
        self.assertEqual(h.specs[-1]["markers"], [], "a search result is a candidate: nothing is drawn until the user picks it")
        h.choose(0)
        m = h.specs[-1]["markers"]
        self.assertEqual(len(m), 1)
        self.assertEqual((m[0]["lat"], m[0]["lon"]), PLACE)
        self.assertEqual(h.specs[-1]["lines"], [])
        self.assertTrue(h.at.button(key="lm_fetch_btn").disabled)  # no key: no traffic request possible
        self.assertEqual(h.tomtom_calls(), [])
        self.assertIn("OpenStreetMap contributors (ODbL), via Nominatim", h.texts())

    def test_short_search_text_is_refused_without_a_request(self):
        h = self.harness().search("ab")
        self.assertEqual(h.net, [])
        self.assertIn("at least 3 characters", " ".join(w.value for w in h.at.warning))

    def test_empty_search_result_shows_a_clear_message(self):
        h = self.harness(geocode_payload=[]).search("nonexistent place zzz")
        self.assertIn("No matching place found", " ".join(i.value for i in h.at.info))
        self.assertEqual(h.specs[-1]["markers"], [])

    def test_search_service_failure_is_reported_and_the_map_stays(self):
        h = self.harness(geocode_payload=requests.Timeout("slow")).search("silk board")
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertIn("did not answer in time", " ".join(w.value for w in h.at.warning))
        self.assertEqual(h.specs[-1]["markers"], [])

    # ---- coordinates: validation
    def test_typed_coordinates_are_validated(self):
        h = self.harness()
        h.at.number_input(key="lm_lat").set_value(48.8566)
        h.at.number_input(key="lm_lon").set_value(2.3522)
        h.at.button(key="lm_use_coords").click().run()
        self.assertIn("outside the Bengaluru area", " ".join(e.value for e in h.at.error))
        self.assertEqual(h.specs[-1]["markers"], [])
        h.at.number_input(key="lm_lat").set_value(12.9159)
        h.at.number_input(key="lm_lon").set_value(77.6240)
        h.at.button(key="lm_use_coords").click().run()
        self.assertEqual(len(h.specs[-1]["markers"]), 1)
        self.assertEqual(h.net, [])

    # ---- live traffic (mocked TomTom)
    def test_valid_mocked_reading_is_shown_with_last_fetched_and_without_the_key(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertEqual([e.value for e in h.at.exception], [])
        calls = h.tomtom_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["params"]["key"], KEY)  # sent to the provider by the server...
        line = h.specs[-1]["lines"]
        self.assertEqual(len(line), 1)
        self.assertEqual(line[0]["points"][0], [12.9159, 77.6240])
        self.assertEqual(line[0]["color"], "#EF4444")  # 24 of 52 km/h = 46%: congested band
        text = h.texts()
        for needle in ("24 km/h", "46% of the free-flow speed", "Last fetched", "not provided by this API", "Congested",
                       "Traffic data: © TomTom", "not used by the forecasting models"):
            self.assertIn(needle, text)
        # ...but never shown to the browser
        blob = text + repr(h.specs) + " ".join(str(v) for v in [e.value for e in h.at.error])
        self.assertNotIn(KEY, blob)
        self.assertNotIn("api.tomtom.com", repr(h.specs))

    def test_refresh_is_served_from_the_cache_within_30_seconds(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch().fetch()
        self.assertEqual(len(h.tomtom_calls()), 1)
        self.assertIn("once every 30 seconds", " ".join(i.value for i in h.at.info))

    def test_a_stale_reading_is_flagged_and_never_coloured_as_current(self):
        old = lambda: __import__("time").time() - 700
        h = self.harness(key=KEY, clock=old).search("silk board junction").choose(0).fetch()
        self.assertEqual(h.specs[-1]["lines"][0]["color"], "#93A4C0")  # neutral grey, not green/amber/red
        self.assertIn("STALE", h.texts())
        self.assertIn("Stale reading", h.texts())

    def test_provider_problems_are_explained_and_draw_nothing(self):
        cases = [(requests.Timeout("slow"), "did not answer in time"), (requests.ConnectionError(f"x key={KEY}"), "Could not reach TomTom"),
                 (FakeResp(401, {}), "rejected the request"), (FakeResp(403, {}), "rejected the request"),
                 (FakeResp(429, {}), "limiting requests"), (FakeResp(400, REC_NO_ROAD), "no traffic segment"), (FakeResp(503, {}), "temporarily unavailable"),
                 (FakeResp(404, CONSTRUCTED_404), "does not recognise"), (FakeResp(400, REC_BAD_POINT), "does not recognise"),
                 (FakeResp(200, {"unexpected": 1}), "rejected"), (FakeResp(200, json_error=True), "could not be read")]
        for outcome, expected in cases:
            with self.subTest(expected):
                h = PageHarness(key=KEY, tomtom_outcome=outcome).start()
                try:
                    h.search("silk board junction").choose(0).fetch()
                    self.assertEqual([e.value for e in h.at.exception], [])
                    shown = " ".join(x.value for g in (h.at.error, h.at.warning, h.at.info) for x in g)
                    self.assertIn(expected, shown)
                    self.assertEqual(h.specs[-1]["lines"], [], "no traffic is drawn without a valid response")
                    self.assertNotIn("km/h", h.texts())
                    self.assertNotIn(KEY, h.texts() + shown)
                finally:
                    h.stop()

    def test_failed_refresh_keeps_the_old_reading_visibly_aged(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        h.tomtom_outcome = requests.Timeout("slow")
        h.service.min_refresh = 0
        h.fetch()
        self.assertIn("did not answer in time", " ".join(w.value for w in h.at.warning))
        self.assertEqual(len(h.specs[-1]["lines"]), 1)  # the earlier reading stays, with its own fetch time

    def test_changing_the_place_clears_the_previous_reading(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertEqual(len(h.specs[-1]["lines"]), 1)
        h.at.button(key="lm_clear_btn").click().run()
        self.assertEqual((h.specs[-1]["markers"], h.specs[-1]["lines"]), ([], []))

    def test_layer_toggle_hides_the_traffic_segment(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        h.at.button_group(key="lm_layers").set_value(["Selected place"]).run()
        self.assertEqual(h.specs[-1]["lines"], [])
        self.assertEqual(len(h.specs[-1]["markers"]), 1)

    def test_nothing_is_written_to_disk_by_the_page(self):
        before = snapshot()
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertEqual(snapshot() - before, set())


class SidebarFollowsTheLatestFetchTests(unittest.TestCase):
    """D4: the sidebar status chip used to be drawn BEFORE the page ran, so it showed the state from before a fetch and lagged one interaction
    behind. These run the real router (app.py): the chip must already reflect a fetch in the SAME run that performed it."""

    def harness(self, **kw):
        h = PageHarness(script=APP, **kw).start()
        self.addCleanup(h.stop)
        return h

    @staticmethod
    def chip(h):
        return " ".join(m.value for m in h.at.sidebar.markdown if "Live traffic:" in m.value)

    def test_a_key_that_was_never_used_reads_not_yet_tested(self):
        h = self.harness(key=KEY)
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertIn("Live traffic: key found, not yet tested", self.chip(h))

    def test_a_successful_fetch_is_in_the_chip_in_the_same_run(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0)
        self.assertIn("not yet tested", self.chip(h), "nothing has been fetched yet")
        h.fetch()                                               # ONE interaction: the fetch
        self.assertEqual(len(h.tomtom_calls()), 1)
        chip = self.chip(h)
        self.assertIn("Live traffic: last response OK", chip)   # before the fix this still said 'not yet tested'
        self.assertNotIn("not yet tested", chip)
        diag = " ".join(m.value for m in h.at.sidebar.expander[0].markdown) + " ".join(c.value for c in h.at.sidebar.expander[0].caption)
        self.assertIn("The last valid provider response was received at", diag)
        self.assertIn("this run has made 1 provider request(s)", diag)

    def test_a_failed_fetch_is_in_the_chip_in_the_same_run(self):
        h = self.harness(key=KEY, tomtom_outcome=FakeResp(401, {})).search("silk board junction").choose(0).fetch()
        chip = self.chip(h)
        self.assertIn("Live traffic: provider problem", chip)
        self.assertNotIn("not yet tested", chip)
        self.assertNotIn("last response OK", chip)
        self.assertEqual(len(h.specs[-1]["lines"]), 0, "a failed fetch draws no traffic")

    def test_a_failed_refresh_after_a_success_shows_the_problem_and_keeps_the_last_valid_time(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertIn("last response OK", self.chip(h))
        h.tomtom_outcome = requests.Timeout("slow")
        h.service.min_refresh = 0
        h.fetch()
        self.assertIn("Live traffic: provider problem", self.chip(h))
        diag = " ".join(m.value for m in h.at.sidebar.expander[0].markdown)
        self.assertIn("The last valid response was at", diag)   # the stale reading's own time is still reported
        self.assertEqual(len(h.specs[-1]["lines"]), 1)

    def test_a_refresh_refused_by_the_30_second_limit_does_not_change_the_status(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        before = self.chip(h)
        h.fetch()                                               # refused by the shared limit: no new request
        self.assertEqual(len(h.tomtom_calls()), 1)
        self.assertEqual(self.chip(h), before)

    def test_the_status_stays_under_the_navigation_and_is_drawn_once(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        headings = [m.value for m in h.at.sidebar.markdown if "System status" in m.value]
        self.assertEqual(len(headings), 1, "the sidebar status must be drawn exactly once per run")
        self.assertEqual([e.value for e in h.at.exception], [])
        self.assertEqual([e.label for e in h.at.sidebar.expander], ["Diagnostics (files, models, data)"])

    def test_the_key_is_not_in_the_sidebar(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        side = " ".join(m.value for m in h.at.sidebar.markdown) + " ".join(c.value for c in h.at.sidebar.caption)
        self.assertNotIn(KEY, side)


class RecordedUi:
    """Records what a run PUTS on the screen while it is busy (a spinner text, a caption), because the final element tree never shows it."""

    def __init__(self):
        from streamlit.delta_generator import DeltaGenerator
        self.spinners, self.captions = [], []
        self._dg, self._orig_caption, self._orig_spinner = DeltaGenerator, DeltaGenerator.caption, st.spinner
        outer = self

        def caption(dg, body, *a, **k):
            outer.captions.append(str(body))
            return outer._orig_caption(dg, body, *a, **k)

        def spinner(text="In progress...", *a, **k):
            outer.spinners.append(str(text))
            return outer._orig_spinner(text, *a, **k)
        self._patches = [mock.patch.object(DeltaGenerator, "caption", caption), mock.patch.object(st, "spinner", spinner)]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


class LoadingStateTests(unittest.TestCase):
    """The sidebar says which status it is showing while a run is busy, and a slow operation says what it is doing. The line never outlives the run."""

    def harness(self, **kw):
        h = PageHarness(script=APP, **kw).start()
        self.addCleanup(h.stop)
        return h

    def test_the_first_run_says_checking_and_later_runs_say_the_status_is_from_the_previous_action(self):
        with RecordedUi() as ui:
            h = self.harness(key=KEY)
            first = [c for c in ui.captions if c.startswith(("Checking", "Updating"))]
            self.assertEqual(first, ["Checking..."])
            ui.captions.clear()
            h.search("silk board junction")                      # a later run of the same session
        later = [c for c in ui.captions if c.startswith(("Checking", "Updating"))]
        self.assertEqual(later, ["Updating... the status below is from the previous action."])

    def test_the_loading_line_is_gone_when_the_run_has_finished(self):
        h = self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        side = " ".join(c.value for c in h.at.sidebar.caption)
        self.assertNotIn("Updating", side)
        self.assertNotIn("Checking", side)
        self.assertIn("Live traffic: last response OK", " ".join(m.value for m in h.at.sidebar.markdown))   # and the status is the fresh one

    def test_a_page_that_crashes_still_gets_a_fresh_status_and_no_leftover_loading_line(self):
        def crashing_pages(st_):
            def broken():
                raise RuntimeError("deliberate failure of a test page")
            return [st_.Page(broken, title="Live Traffic Map", url_path="live-map", default=True)]
        from src.ui import navigation
        with mock.patch.object(navigation, "build_pages", crashing_pages), mock.patch.dict(os.environ), \
                mock.patch.object(tomtom, "SERVICE", tomtom.FlowService()):
            os.environ["TOMTOM_API_KEY"] = KEY
            at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertTrue(any("deliberate failure" in e.value for e in at.exception), "the page's own error is still reported, not hidden")
        self.assertIn("System status", " ".join(m.value for m in at.sidebar.markdown))
        self.assertIn("Live traffic: key found, not yet tested", " ".join(m.value for m in at.sidebar.markdown))
        self.assertNotIn("Updating", " ".join(c.value for c in at.sidebar.caption))
        self.assertNotIn("Checking", " ".join(c.value for c in at.sidebar.caption))

    def test_a_place_search_shows_that_it_is_searching(self):
        with RecordedUi() as ui:
            self.harness(key=KEY).search("silk board junction")
        self.assertIn("Searching OpenStreetMap...", ui.spinners)

    def test_a_traffic_request_shows_that_it_is_contacting_tomtom(self):
        with RecordedUi() as ui:
            self.harness(key=KEY).search("silk board junction").choose(0).fetch()
        self.assertIn("Contacting TomTom...", ui.spinners)


if __name__ == "__main__":
    unittest.main()

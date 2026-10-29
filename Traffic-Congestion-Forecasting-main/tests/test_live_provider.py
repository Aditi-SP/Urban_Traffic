"""
Stage 2 tests: the safeguards around the TomTom provider (in-flight sharing, 429 cooldown, bounded and expiring in-memory cache,
request budget, no key leak). The basic client/validation/cache tests are in test_live_map.py and are not repeated here.

All provider traffic here is MOCKED. Passing these tests says nothing about whether the real TomTom endpoint works with a valid key.

Run:   python -m unittest tests.test_live_provider -v
"""
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
os.environ.pop("LIVE_CACHE_TTL_S", None)  # tests must not depend on a local runtime setting; tests that need it set it themselves

from src.live import tomtom  # noqa: E402
from tests.test_live_map import KEY, FakeResp, flow_payload  # noqa: E402

PLACE = (12.9158955, 77.6239872)


def observation(lat, lon, at):
    return tomtom.parse_response(flow_payload(), lat, lon, at)


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t


def service(clock, fetcher, **kw):
    return tomtom.FlowService(fetcher=fetcher, clock=clock, **kw)


class InFlightSharingTests(unittest.TestCase):
    def run_threads(self, svc, n=6):
        results, errors = [], []

        def worker():
            try:
                results.append(svc.get(*PLACE, KEY))
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        return threads, results, errors

    def test_simultaneous_requests_for_one_point_make_one_provider_call(self):
        clock, started, release, calls = Clock(), threading.Event(), threading.Event(), []

        def fetcher(lat, lon, key):
            calls.append(1)
            started.set()
            release.wait(10)
            return observation(lat, lon, clock())
        svc = service(clock, fetcher)
        threads, results, errors = self.run_threads(svc)
        self.assertTrue(started.wait(10))
        time.sleep(0.3)  # let the other workers reach the lock while the first request is still in progress
        release.set()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(svc.requests_made, 1, "one request must cost one unit of the budget, however many callers waited")
        self.assertEqual(len(results), 6)
        self.assertEqual(len({id(o) for o, _ in results}), 1)
        self.assertEqual(sorted(s for _, s in results), ["cache"] * 5 + ["provider"])

    def test_a_failed_shared_request_fails_for_everyone_and_is_not_cached(self):
        clock, started, release, calls = Clock(), threading.Event(), threading.Event(), []

        def fetcher(lat, lon, key):
            calls.append(1)
            started.set()
            release.wait(10)
            raise tomtom.ProviderError("timeout", "TomTom did not answer in time. Please try again.")
        svc = service(clock, fetcher)
        threads, results, errors = self.run_threads(svc, n=4)
        self.assertTrue(started.wait(10))
        time.sleep(0.3)
        release.set()
        for t in threads:
            t.join(10)
        self.assertEqual((len(calls), len(results), len(errors)), (1, 0, 4))
        self.assertTrue(all(isinstance(e, tomtom.ProviderError) and e.kind == "timeout" for e in errors))
        with self.assertRaises(tomtom.ProviderError):  # nothing was cached; a later attempt really asks again (and costs budget)
            svc.get(*PLACE, KEY)
        self.assertEqual((len(calls), svc.requests_made), (2, 2))

    def test_a_crashing_fetcher_does_not_leave_later_callers_waiting(self):
        clock = Clock()
        svc = service(clock, mock.Mock(side_effect=[ValueError("boom"), observation(*PLACE, 1000.0)]))
        with self.assertRaises(ValueError):
            svc.get(*PLACE, KEY)
        self.assertEqual(svc._inflight, {})
        self.assertEqual(svc.get(*PLACE, KEY)[1], "provider")


class RateLimitCooldownTests(unittest.TestCase):
    def test_429_pauses_provider_requests_but_still_serves_cached_readings(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=[observation(12.90, 77.60, clock()),
                                         tomtom.ProviderError("rate_limited", "TomTom is limiting requests. Please wait before trying again."),
                                         observation(12.95, 77.65, clock())])
        svc = service(clock, fetcher)
        svc.get(12.90, 77.60, KEY)
        clock.t += 1
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(12.91, 77.61, KEY)
        self.assertEqual(cm.exception.kind, "rate_limited")
        self.assertEqual(fetcher.call_count, 2)

        clock.t += 5  # inside the cooldown: no provider request, and it is not charged to the budget
        used = svc.requests_made
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(12.92, 77.62, KEY)
        self.assertEqual(cm.exception.kind, "rate_limited")
        self.assertIn("wait about", cm.exception.message)
        self.assertEqual((fetcher.call_count, svc.requests_made), (2, used))
        self.assertEqual(svc.get(12.90, 77.60, KEY)[1], "cache")  # a reading already held is still shown

        clock.t += tomtom.RATE_LIMIT_COOLDOWN_S  # after the cooldown a request is allowed again
        self.assertEqual(svc.get(12.95, 77.65, KEY)[1], "provider")

    def test_other_errors_do_not_start_a_cooldown(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=[tomtom.ProviderError("timeout", "slow"), tomtom.ProviderError("server", "down"),
                                         observation(*PLACE, clock())])
        svc = service(clock, fetcher)
        for _ in range(2):
            with self.assertRaises(tomtom.ProviderError):
                svc.get(*PLACE, KEY)
        self.assertEqual(svc.get(*PLACE, KEY)[1], "provider")
        self.assertEqual(fetcher.call_count, 3)

    def test_http_429_from_the_real_client_starts_the_cooldown_without_leaking_the_key(self):
        clock = Clock(time.time())
        svc = tomtom.FlowService(clock=clock)
        get = mock.Mock(return_value=FakeResp(429, {}))
        with mock.patch.object(tomtom.requests, "get", get):
            with self.assertRaises(tomtom.ProviderError) as first:
                svc.get(*PLACE, KEY)
            with self.assertRaises(tomtom.ProviderError) as second:
                svc.get(*PLACE, KEY)
        self.assertEqual((first.exception.kind, second.exception.kind), ("rate_limited", "rate_limited"))
        self.assertEqual(get.call_count, 1, "the second attempt must not reach the provider")
        for e in (first.exception, second.exception):
            self.assertNotIn(KEY, e.message)


class CacheTests(unittest.TestCase):
    def test_an_expired_reading_is_never_returned_as_a_fresh_one(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=[observation(*PLACE, clock()), tomtom.ProviderError("timeout", "slow")])
        svc = service(clock, fetcher)
        first, _ = svc.get(*PLACE, KEY)
        clock.t += tomtom.CACHE_TTL_S + 1
        with self.assertRaises(tomtom.ProviderError):  # the refresh failed: the old reading is NOT handed back as current
            svc.get(*PLACE, KEY)
        self.assertEqual(tomtom.band(first, clock()), "congested")  # 301 s old: still inside the 10 min "current" window, so a real band
        clock.t += tomtom.STALE_AFTER_S
        self.assertEqual(tomtom.band(first, clock()), "stale")
        self.assertGreater(tomtom.age_s(first, clock()), tomtom.STALE_AFTER_S)

    def test_a_refetch_after_expiry_carries_a_new_fetch_time(self):
        clock = Clock()
        svc = service(clock, lambda lat, lon, key: observation(lat, lon, clock()))
        a, _ = svc.get(*PLACE, KEY)
        clock.t += tomtom.CACHE_TTL_S + 1
        b, source = svc.get(*PLACE, KEY)
        self.assertEqual(source, "provider")
        self.assertEqual(b.fetched_at - a.fetched_at, tomtom.CACHE_TTL_S + 1)

    def test_cache_is_bounded(self):
        clock = Clock()
        svc = service(clock, lambda lat, lon, key: observation(lat, lon, clock()))
        with mock.patch.object(tomtom, "MAX_CACHE_ENTRIES", 3):
            for i in range(5):
                clock.t += 1
                svc.get(12.90 + i * 0.01, 77.60, KEY)
        self.assertEqual(len(svc._cache), 3)
        self.assertNotIn((12.90, 77.60), svc._cache)  # the oldest were dropped
        self.assertIn((12.94, 77.60), svc._cache)

    def test_ttl_can_be_changed_from_the_environment_and_defaults_to_five_minutes(self):
        self.assertEqual(tomtom.CACHE_TTL_S, 300)
        with mock.patch.dict(os.environ, {"LIVE_CACHE_TTL_S": "0"}):
            clock = Clock()
            fetcher = mock.Mock(side_effect=lambda lat, lon, key: observation(lat, lon, clock()))
            svc = service(clock, fetcher)
            self.assertEqual(svc.ttl, 0)
            svc.get(*PLACE, KEY)
            svc.get(*PLACE, KEY)
            self.assertEqual(fetcher.call_count, 2, "ttl 0 disables reuse between requests")
        for value, expected in (("junk", 300), ("-5", 0), ("120", 120)):
            with mock.patch.dict(os.environ, {"LIVE_CACHE_TTL_S": value}):
                self.assertEqual(tomtom.FlowService().ttl, expected, value)
        with mock.patch.dict(os.environ):
            os.environ.pop("LIVE_CACHE_TTL_S", None)
            self.assertEqual(tomtom.FlowService().ttl, 300)

    def test_cache_hits_make_no_http_request_through_the_real_client(self):
        clock = Clock(time.time())
        svc = tomtom.FlowService(clock=clock)
        get = mock.Mock(return_value=FakeResp(200, flow_payload()))
        with mock.patch.object(tomtom.requests, "get", get):
            svc.get(*PLACE, KEY)
            clock.t += 20
            svc.get(*PLACE, KEY)                 # within the TTL
            self.assertEqual(svc.get(*PLACE, KEY, force=True)[1], "cache")  # a refresh within 30 s is also answered from memory
            clock.t += tomtom.MIN_REFRESH_S
            self.assertEqual(svc.get(*PLACE, KEY, force=True)[1], "provider")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(svc.requests_made, 2)

    def test_request_cap_counts_provider_attempts_only(self):
        clock = Clock()
        svc = service(clock, lambda lat, lon, key: observation(lat, lon, clock()), budget=2)
        for _ in range(5):
            svc.get(12.90, 77.60, KEY)  # cache hits
        self.assertEqual(svc.requests_made, 1)
        svc.get(12.91, 77.61, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(12.92, 77.62, KEY)
        self.assertEqual(cm.exception.kind, "budget")
        self.assertNotIn(KEY, cm.exception.message)


class TtlZeroTests(unittest.TestCase):
    """LIVE_CACHE_TTL_S=0: no provider reading is kept or reused between requests, yet the quota safeguards still work."""

    def zero(self, clock, fetcher, **kw):
        with mock.patch.dict(os.environ, {"LIVE_CACHE_TTL_S": "0"}):
            svc = service(clock, fetcher, **kw)
        self.assertEqual(svc.ttl, 0)
        return svc

    def test_nothing_is_kept_and_ordinary_requests_are_never_answered_from_memory(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=lambda lat, lon, key: observation(lat, lon, clock()))
        svc = self.zero(clock, fetcher)
        for _ in range(3):
            self.assertEqual(svc.get(*PLACE, KEY)[1], "provider")
            clock.t += 1
        self.assertEqual(fetcher.call_count, 3)
        self.assertEqual(svc._cache, {}, "no provider reading may be retained")
        self.assertTrue(all(isinstance(v, float) for v in svc._last_request.values()), "only request times are remembered")

    def test_an_early_refresh_is_refused_not_answered_from_a_stored_copy(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=lambda lat, lon, key: observation(lat, lon, clock()))
        svc = self.zero(clock, fetcher)
        svc.get(*PLACE, KEY)
        clock.t += 5
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(*PLACE, KEY, force=True)
        self.assertEqual(cm.exception.kind, "throttled")
        self.assertIn("once every 30 seconds", cm.exception.message)
        self.assertEqual((fetcher.call_count, svc.requests_made), (1, 1), "a refused refresh costs neither a request nor budget")
        clock.t += tomtom.MIN_REFRESH_S  # the 30 second limit is measured from the last provider request
        self.assertEqual(svc.get(*PLACE, KEY, force=True)[1], "provider")
        self.assertEqual((fetcher.call_count, svc.requests_made, svc._cache), (2, 2, {}))

    def test_the_refresh_limit_is_per_point(self):
        clock = Clock()
        svc = self.zero(clock, lambda lat, lon, key: observation(lat, lon, clock()))
        svc.get(12.90, 77.60, KEY)
        clock.t += 1
        self.assertEqual(svc.get(12.95, 77.65, KEY, force=True)[1], "provider")

    def test_simultaneous_requests_still_share_one_provider_call_but_nothing_is_kept(self):
        clock, started, release, calls = Clock(), threading.Event(), threading.Event(), []

        def fetcher(lat, lon, key):
            calls.append(1)
            started.set()
            release.wait(10)
            return observation(lat, lon, clock())
        svc = self.zero(clock, fetcher)
        out = []
        threads = [threading.Thread(target=lambda: out.append(svc.get(*PLACE, KEY))) for _ in range(4)]
        for t in threads:
            t.start()
        self.assertTrue(started.wait(10))
        time.sleep(0.3)
        release.set()
        for t in threads:
            t.join(10)
        self.assertEqual((len(calls), len(out), svc.requests_made), (1, 4, 1))
        self.assertEqual((svc._cache, svc._inflight), ({}, {}))
        svc.get(*PLACE, KEY)  # a LATER request is a new provider request, not a reuse
        self.assertEqual(len(calls), 2)

    def test_the_429_cooldown_and_the_budget_still_apply(self):
        clock = Clock()
        fetcher = mock.Mock(side_effect=[tomtom.ProviderError("rate_limited", "TomTom is limiting requests. Please wait before trying again.")])
        svc = self.zero(clock, fetcher, budget=5)
        with self.assertRaises(tomtom.ProviderError):
            svc.get(*PLACE, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            svc.get(12.95, 77.65, KEY)
        self.assertEqual((cm.exception.kind, fetcher.call_count), ("rate_limited", 1))
        capped = self.zero(Clock(), lambda lat, lon, key: observation(lat, lon, 1000.0), budget=1)
        capped.get(12.90, 77.60, KEY)
        with self.assertRaises(tomtom.ProviderError) as cm:
            capped.get(12.91, 77.61, KEY)
        self.assertEqual(cm.exception.kind, "budget")

    def test_staleness_is_still_judged_from_the_fetch_time_of_the_reading_on_screen(self):
        clock = Clock()
        svc = self.zero(clock, lambda lat, lon, key: observation(lat, lon, clock()))
        shown, _ = svc.get(*PLACE, KEY)
        clock.t += tomtom.STALE_AFTER_S + 1
        self.assertEqual(tomtom.band(shown, clock()), "stale")


class TtlZeroPageTests(unittest.TestCase):
    def test_an_early_refresh_keeps_the_reading_on_screen_and_says_why(self):
        from tests.test_live_map import PageHarness
        with mock.patch.dict(os.environ, {"LIVE_CACHE_TTL_S": "0"}):
            h = PageHarness(key=KEY)  # the service reads the setting when it is created
        self.assertEqual(h.service.ttl, 0)
        h.start()  # started outside the patch: start() snapshots and restores the environment itself
        self.addCleanup(h.stop)
        h.search("silk board junction").choose(0).fetch()
        self.assertEqual(len(h.specs[-1]["lines"]), 1)
        h.fetch()  # explicit refresh straight away
        self.assertEqual(len(h.tomtom_calls()), 1, "the refresh must not reach the provider")
        self.assertIn("once every 30 seconds", " ".join(i.value for i in h.at.info))
        self.assertEqual(len(h.specs[-1]["lines"]), 1, "the earlier reading stays, with its own Last fetched time")
        self.assertNotIn(KEY, h.texts())


if __name__ == "__main__":
    unittest.main()

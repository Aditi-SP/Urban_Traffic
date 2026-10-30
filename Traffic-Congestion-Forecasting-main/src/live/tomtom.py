"""
Live traffic from the TomTom Traffic Flow "Flow Segment Data" API.

Documentation read: https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/traffic-flow/flow-segment-data
  GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json?key=...&point=lat,lon&unit=kmph
  response: {"flowSegmentData": {currentSpeed, freeFlowSpeed, currentTravelTime, freeFlowTravelTime, confidence,
                                  roadClosure, frc, coordinates: {coordinate: [{latitude, longitude}, ...]}}}

STATUS: this module has been tested with MOCKED responses only. It has NOT been run against the live service with a valid key,
so no claim is made that the real integration works until that has been done.

Rules built in
  * The key is read by the caller (environment / local secrets) and used server-side only. It is never put in a message,
    an exception, a log line or the page: provider errors are translated into our own messages WITHOUT the request URL
    (requests' own error text contains the URL and therefore the key).
  * Strict validation: a response with a missing, mistyped or implausible field is rejected as a whole; nothing is filled in.
  * "Last fetched" is OUR clock. This endpoint returns no measurement time, so the provider's own timestamp is unknown.
  * Provider data is kept in memory only (a small TTL cache); nothing is written to disk until the licence terms are verified.
  * Requests have timeouts, there is no automatic retry, and a per-run budget protects the free monthly quota.
  * The result is NEVER fed to the forecasting models: they need inputs this provider does not supply.
"""
import math
import os
import threading
import time
from dataclasses import dataclass

import requests

from src.live import geo

PROVIDER = "TomTom Traffic Flow (Flow Segment Data)"
ATTRIBUTION = "Traffic data: © TomTom"
URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
ZOOM = 10
TIMEOUT = (3.05, 8.0)           # (connect, read) seconds
CACHE_TTL_S = 300               # a repeated request for the same point within 5 min is answered from memory
MIN_REFRESH_S = 30              # an explicit refresh is not sent to the provider more often than this
STALE_AFTER_S = 600             # a reading older than 10 min is shown as stale, never as current
RUN_BUDGET_DEFAULT = 300        # provider requests per application run (free tier: 20,000 per month, per TomTom's pricing page)
RATE_LIMIT_COOLDOWN_S = 60      # after an HTTP 429, no provider request is sent for this long
MAX_CACHE_ENTRIES = 200         # bound on the in-memory cache
MAX_SEGMENT_OFFSET_KM = 5.0    # the returned segment must lie near the point that was asked for

# Display bands: this application's convention, based on current speed as a share of free-flow speed.
# They are NOT provider categories and are NOT validated against the dataset's Congestion Level.
FREE_MIN_RATIO = 0.80
SLOW_MIN_RATIO = 0.50


class ProviderError(RuntimeError):
    """A provider problem. `kind` is machine-readable; `message` is safe to show (it never contains the key or the URL)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


def _err(kind, message):
    return ProviderError(kind, message)


@dataclass(frozen=True)
class FlowObservation:
    lat: float                       # the point that was asked for
    lon: float
    current_speed_kmh: float
    free_flow_speed_kmh: float
    current_travel_time_s: float
    free_flow_travel_time_s: float
    confidence: float
    road_closure: object             # bool, or None when the provider did not say
    frc: object                      # functional road class text, or None
    segment: tuple                   # ((lat, lon), ...) geometry returned by the provider
    fetched_at: float                # epoch seconds on OUR clock when the response was received
    provider: str = PROVIDER

    @property
    def speed_ratio(self):
        return self.current_speed_kmh / self.free_flow_speed_kmh


def age_s(obs: FlowObservation, now=None) -> float:
    return max(0.0, (time.time() if now is None else now) - obs.fetched_at)


def is_stale(obs: FlowObservation, now=None) -> bool:
    return age_s(obs, now) > STALE_AFTER_S


def band(obs: FlowObservation, now=None) -> str:
    """'closed' | 'stale' | 'free' | 'slow' | 'congested'. A stale reading is never classified as current."""
    if obs.road_closure is True:
        return "closed"
    if is_stale(obs, now):
        return "stale"
    r = obs.speed_ratio
    return "free" if r >= FREE_MIN_RATIO else "slow" if r >= SLOW_MIN_RATIO else "congested"


def _number(seg, name, lo, hi):
    v = seg.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise _err("invalid_response", f"The provider response has an invalid '{name}' value, so it was rejected.")
    return float(v)


def parse_response(payload, lat: float, lon: float, fetched_at: float) -> FlowObservation:
    """Validate a decoded JSON response and build an observation, or raise ProviderError('invalid_response')."""
    seg = payload.get("flowSegmentData") if isinstance(payload, dict) else None
    if not isinstance(seg, dict):
        raise _err("invalid_response", "The provider response does not contain traffic flow data, so it was rejected.")
    current = _number(seg, "currentSpeed", 0, 300)
    free = _number(seg, "freeFlowSpeed", 0, 300)
    if free <= 0:
        raise _err("invalid_response", "The provider reported a free-flow speed of zero, so it was rejected.")
    cur_t = _number(seg, "currentTravelTime", 0, 86400)
    free_t = _number(seg, "freeFlowTravelTime", 0, 86400)
    conf = _number(seg, "confidence", 0, 1)
    closure = seg.get("roadClosure")
    if closure is not None and not isinstance(closure, bool):
        raise _err("invalid_response", "The provider response has an invalid 'roadClosure' value, so it was rejected.")
    frc = seg.get("frc")
    frc = frc if isinstance(frc, str) else None

    coords = seg.get("coordinates")
    pts = coords.get("coordinate") if isinstance(coords, dict) else None
    if not isinstance(pts, list) or not 2 <= len(pts) <= 1000:
        raise _err("invalid_response", "The provider response has no usable road geometry, so it was rejected.")
    segment = []
    for p in pts:
        if not isinstance(p, dict):
            raise _err("invalid_response", "The provider response has invalid road geometry, so it was rejected.")
        try:
            la, lo = geo.validate_location(p.get("latitude"), p.get("longitude"), bbox=None)
        except geo.LocationError:
            raise _err("invalid_response", "The provider response has invalid coordinates, so it was rejected.") from None
        segment.append((la, lo))
    if min(geo.distance_km(lat, lon, la, lo) for la, lo in segment) > MAX_SEGMENT_OFFSET_KM:
        raise _err("invalid_response", "The returned road segment is not near the requested location, so it was rejected.")
    return FlowObservation(lat=lat, lon=lon, current_speed_kmh=current, free_flow_speed_kmh=free, current_travel_time_s=cur_t,
                           free_flow_travel_time_s=free_t, confidence=conf, road_closure=closure, frc=frc,
                           segment=tuple(segment), fetched_at=fetched_at)


# WHAT IS VERIFIED about refused keys: a real, deliberately invalid key gets HTTP 401 (checked against the live service). HOW TomTom words an
# HTTP 403 (no access to the Traffic API, quota, or something else) has NEVER been observed, and TomTom's terms and quotas are unread (README:
# "Verification status" and "TomTom terms"). The three word lists below are therefore GUESSES, not TomTom's documented wording. They are used
# only to choose between messages that say what the answer SUGGESTS and what to check; when no list matches, the neutral message is used, which
# names the possible causes and says TomTom did not say which. The body is used only to choose a message: it is never shown or logged, and the
# key is removed from it first. Replace the guesses with TomTom's real wording once a real 403 has been seen; do not add new messages before that.
_QUOTA_WORDS = ("over the limit", "over limit", "quota", "exceeded", "limit reached")
_PERMISSION_WORDS = ("not authorized", "not authorised", "no access", "not enabled", "not allowed", "permission", "subscription", "product", "not entitled")
_INVALID_WORDS = ("invalid", "not valid", "expired", "unknown key", "no such key", "missing key", "wrong key")


def _body_text(resp, key) -> str:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001  an unreadable body just means "unclear"
        return ""
    parts = []
    if isinstance(body, dict):
        parts += [v for k, v in body.items() if k in ("error", "message", "errorText") and isinstance(v, str)]
        detail = body.get("detailedError")
        if isinstance(detail, dict):
            parts += [v for k, v in detail.items() if k in ("code", "message") and isinstance(v, str)]
    return " ".join(parts).lower().replace(key.lower(), "")[:500]


def _denied(status, resp, key) -> ProviderError:
    """HTTP 401/403. 401 is the verified "key not accepted" answer. For a 403 the body is only a hint (see the note above): the message says what
    the answer SUGGESTS, and the neutral message is used when the hint is not recognised."""
    text = _body_text(resp, key)
    if status == 401:
        return _err("auth", "TomTom rejected the request: the key was not accepted (it is invalid or has expired). Check that you copied it completely.")
    if any(w in text for w in _QUOTA_WORDS):
        return _err("quota", "TomTom rejected the request (HTTP 403) and its answer suggests the key's request quota may be used up. Check your usage and plan "
                             "in your TomTom account; TomTom's wording for this case has not been verified here.")
    if any(w in text for w in _PERMISSION_WORDS):
        return _err("forbidden", "TomTom rejected the request (HTTP 403) and its answer suggests this key may not be enabled for the Traffic Flow API. If so, "
                                 "enable the Traffic API for this key in your TomTom developer account; TomTom's wording for this case has not been verified here.")
    if any(w in text for w in _INVALID_WORDS):
        return _err("auth", "TomTom rejected the request (HTTP 403) and its answer suggests the key is invalid or has expired. Check that you copied it completely.")
    return _err("auth", "TomTom rejected the request (HTTP 403): the key may be invalid, expired, not enabled for the Traffic API, or over its quota. "
                        "TomTom's answer did not say which; check the key in your TomTom developer account.")


def fetch_flow(lat, lon, api_key, *, http_get=None, clock=time.time) -> FlowObservation:
    """One request to the provider. Raises geo.LocationError (bad location) or ProviderError (everything else)."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise _err("no_key", "No TomTom API key is configured.")  # checked BEFORE any network call
    lat, lon = geo.validate_location(lat, lon)
    try:
        resp = (http_get or requests.get)(URL.format(zoom=ZOOM), params={"key": api_key.strip(), "point": f"{lat},{lon}", "unit": "kmph"},
                                          timeout=TIMEOUT)
    except requests.Timeout:
        raise _err("timeout", "TomTom did not answer in time. Please try again.") from None
    except requests.RequestException:  # message deliberately dropped: requests' text contains the URL and so the key
        raise _err("network", "Could not reach TomTom. Check your internet connection and try again.") from None

    status = resp.status_code
    if status in (401, 403):
        raise _denied(status, resp, api_key.strip())
    if status == 429:
        raise _err("rate_limited", "TomTom is limiting requests. Please wait before trying again.")
    if status in (400, 404):
        raise _err("no_segment", "TomTom has no traffic segment near this location. Try a point on a main road.")
    if status != 200:
        raise _err("server", f"TomTom is temporarily unavailable (HTTP {status}). Please try again later.")
    try:
        payload = resp.json()
    except ValueError:
        raise _err("invalid_response", "The provider response could not be read, so it was rejected.") from None
    return parse_response(payload, lat, lon, clock())


class _Flight:
    """One provider request that is in progress; callers asking for the same point meanwhile wait for its result."""

    def __init__(self):
        self.event = threading.Event()
        self.obs = None
        self.error = None


class FlowService:
    """In-memory cache + request budget around fetch_flow. Nothing is written to disk.

    Safeguards (all enforced in get(), under one lock):
      * cache: a reading younger than `ttl` is reused; an explicit refresh reuses one younger than `min_refresh`.
        With ttl == 0 NO reading is kept or reused between requests; the refresh limit is then enforced from the time of the
        last provider request alone (an explicit refresh inside `min_refresh` is refused with kind 'throttled');
      * in-flight sharing: simultaneous requests for the same point cause ONE provider request;
      * budget: at most `budget` provider requests per application run (an attempt that fails still counts);
      * cooldown: after HTTP 429 no provider request is sent for RATE_LIMIT_COOLDOWN_S (cached readings are still served);
      * the cache holds at most MAX_CACHE_ENTRIES readings.
    """

    def __init__(self, fetcher=None, clock=time.time, ttl=None, min_refresh=MIN_REFRESH_S, budget=None):
        self._fetcher, self._clock, self.min_refresh = fetcher, clock, min_refresh
        self.ttl = ttl if ttl is not None else _env_number("LIVE_CACHE_TTL_S", CACHE_TTL_S, minimum=0)
        self.budget = budget if budget is not None else _env_budget()
        self.requests_made = 0
        self._cooldown_until = 0.0
        self._last_success = None   # when (our clock) the provider last gave a valid response: a time only, never the data
        self._last_error = None     # (kind, when) of the provider's last failure
        self._cache, self._last_request, self._inflight, self._lock = {}, {}, {}, threading.Lock()

    @staticmethod
    def _key(lat, lon):
        return round(float(lat), 4), round(float(lon), 4)  # about 11 m

    def get(self, lat, lon, api_key, *, force=False):
        """Returns (observation, source) with source 'cache' or 'provider'."""
        lat, lon = geo.validate_location(lat, lon)
        k = self._key(lat, lon)
        with self._lock:
            now = self._clock()
            hit = self._cache.get(k)
            if hit is not None:
                age = age_s(hit, now)
                if (not force and age < self.ttl) or (force and age < self.min_refresh):
                    return hit, "cache"
            flight = self._inflight.get(k)
            leader = flight is None
            if leader:
                last = self._last_request.get(k)
                if force and self.ttl <= 0 and last is not None and now - last < self.min_refresh:
                    raise _err("throttled", f"Refreshes are limited to once every {self.min_refresh} seconds. "
                                            "The reading on screen is the latest one fetched.")
                if now < self._cooldown_until:
                    raise _err("rate_limited", "TomTom recently reported too many requests, so no new request is sent yet. "
                                               f"Please wait about {int(self._cooldown_until - now) + 1} seconds.")
                if self.requests_made >= self.budget:
                    raise _err("budget", f"The limit of {self.budget} provider requests for this run was reached (it protects the free monthly quota). "
                                         "Restart the app or raise LIVE_MAX_REQUESTS to continue.")
                self.requests_made += 1
                self._last_request[k] = now  # only a time, never provider data
                while len(self._last_request) > MAX_CACHE_ENTRIES:
                    del self._last_request[min(self._last_request, key=self._last_request.get)]
                flight = self._inflight[k] = _Flight()

        if not leader:  # the same point is already being requested: share that one request instead of sending another
            if not flight.event.wait(sum(TIMEOUT) + 2):
                raise _err("timeout", "TomTom did not answer in time. Please try again.")
            if flight.error is not None:
                raise flight.error
            return flight.obs, "cache"

        try:
            if self._fetcher is None:
                obs = fetch_flow(lat, lon, api_key, clock=self._clock)
            else:
                obs = self._fetcher(lat, lon, api_key)
            flight.obs = obs
            with self._lock:
                self._last_success = obs.fetched_at
                if self.ttl > 0:  # with ttl 0 nothing is kept after the request (callers that waited on it already have the result)
                    self._cache[k] = obs
                while len(self._cache) > MAX_CACHE_ENTRIES:
                    del self._cache[min(self._cache, key=lambda c: self._cache[c].fetched_at)]
            return obs, "provider"
        except BaseException as e:
            flight.error = e
            if isinstance(e, ProviderError):
                with self._lock:
                    self._last_error = (e.kind, self._clock())
            if isinstance(e, ProviderError) and e.kind == "rate_limited":
                with self._lock:
                    self._cooldown_until = self._clock() + RATE_LIMIT_COOLDOWN_S
            raise
        finally:
            with self._lock:
                self._inflight.pop(k, None)
            flight.event.set()

    def status(self) -> dict:
        """What this running service has actually seen (for status panels). No provider request is made and no data is returned."""
        with self._lock:
            now = self._clock()
            return {"requests_made": self.requests_made, "budget": self.budget, "ttl_s": self.ttl,
                    "cooldown_remaining_s": max(0.0, self._cooldown_until - now),
                    "last_success_at": self._last_success, "last_error": self._last_error}

    def clear(self):
        with self._lock:
            self._last_success = self._last_error = None
            self._cache.clear()
            self._last_request.clear()
            self.requests_made = 0
            self._cooldown_until = 0.0


def _env_number(name, default, minimum):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _env_budget() -> int:
    return _env_number("LIVE_MAX_REQUESTS", RUN_BUDGET_DEFAULT, minimum=1)


SERVICE = FlowService()  # the instance the dashboard uses

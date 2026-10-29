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
MAX_SEGMENT_OFFSET_KM = 5.0     # the returned segment must lie near the point that was asked for

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
        raise _err("auth", "TomTom rejected the request: the key may be invalid, expired, not enabled for this API, or over its quota.")
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


class FlowService:
    """In-memory cache + request budget around fetch_flow. Nothing is written to disk."""

    def __init__(self, fetcher=None, clock=time.time, ttl=CACHE_TTL_S, min_refresh=MIN_REFRESH_S, budget=None):
        self._fetcher, self._clock, self.ttl, self.min_refresh = fetcher, clock, ttl, min_refresh
        self.budget = budget if budget is not None else _env_budget()
        self.requests_made = 0
        self._cache, self._lock = {}, threading.Lock()

    @staticmethod
    def _key(lat, lon):
        return round(float(lat), 4), round(float(lon), 4)  # about 11 m

    def get(self, lat, lon, api_key, *, force=False):
        """Returns (observation, source) with source 'cache' or 'provider'."""
        lat, lon = geo.validate_location(lat, lon)
        k = self._key(lat, lon)
        with self._lock:
            hit = self._cache.get(k)
            if hit is not None:
                age = age_s(hit, self._clock())
                if (not force and age < self.ttl) or (force and age < self.min_refresh):
                    return hit, "cache"
            if self.requests_made >= self.budget:
                raise _err("budget", f"The limit of {self.budget} provider requests for this run was reached (it protects the free monthly quota). "
                                     "Restart the app or raise LIVE_MAX_REQUESTS to continue.")
            self.requests_made += 1
        if self._fetcher is None:
            obs = fetch_flow(lat, lon, api_key, clock=self._clock)
        else:
            obs = self._fetcher(lat, lon, api_key)
        with self._lock:
            self._cache[k] = obs
        return obs, "provider"

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.requests_made = 0


def _env_budget() -> int:
    try:
        return max(1, int(os.environ.get("LIVE_MAX_REQUESTS", RUN_BUDGET_DEFAULT)))
    except ValueError:
        return RUN_BUDGET_DEFAULT


SERVICE = FlowService()  # the instance the dashboard uses

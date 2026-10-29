"""
Place search through the public OpenStreetMap Nominatim service, used within its usage policy
(https://operations.osmfoundation.org/policies/nominatim/, read 2026-10-28):

  * at most 1 request per second for the whole application  -> process-wide rate limiter (MIN_INTERVAL_S)
  * a valid, identifying User-Agent (not a library default)  -> USER_AGENT
  * results must be cached on our side                      -> in-memory cache (never written to disk)
  * no autocomplete                                          -> a search runs only when the user submits; short texts are rejected
  * attribution: OpenStreetMap contributors (ODbL)          -> ATTRIBUTION, shown next to the map

Only the first few results are requested, restricted to the Bengaluru area. A result is just a candidate: the user has
to pick it, and nothing here claims that it is where any dataset road is.
"""
import threading
import time
from dataclasses import dataclass

import requests

from src.live import geo

URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "TrafficIntelligenceDashboard/1.0 (local academic project)"
ATTRIBUTION = "Search results: © OpenStreetMap contributors (ODbL), via Nominatim"
TIMEOUT = (3.05, 8.0)        # (connect, read) seconds
MAX_RESULTS = 5
MIN_INTERVAL_S = 1.1         # policy maximum is 1 request per second; stay safely under it
MAX_WAIT_S = 3.0             # if the limiter would make us wait longer than this, say "busy" instead of freezing the page
CACHE_TTL_S = 24 * 3600
EMPTY_TTL_S = 3600
CACHE_MAX = 200


class GeocodeError(RuntimeError):
    """The place search failed; `kind` is one of: timeout, network, rate_limited, http, invalid_response, busy."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


@dataclass(frozen=True)
class Place:
    name: str
    display_name: str
    lat: float
    lon: float
    category: str = ""


class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval, self._last, self._lock = min_interval, None, threading.Lock()

    def wait(self, clock=time.monotonic, sleep=time.sleep):
        with self._lock:
            now = clock()
            if self._last is not None:
                wait = self._last + self.min_interval - now
                if wait > MAX_WAIT_S:
                    raise GeocodeError("busy", "The search service is busy; please try again in a few seconds.")
                if wait > 0:
                    sleep(wait)
            self._last = clock()


_limiter = _RateLimiter(MIN_INTERVAL_S)
_cache: dict = {}
_cache_lock = threading.Lock()


def clear_cache():
    """For tests."""
    with _cache_lock:
        _cache.clear()
    _limiter._last = None


def _parse(payload) -> list:
    if not isinstance(payload, list):
        raise GeocodeError("invalid_response", "The search service returned an unexpected response.")
    places, seen = [], set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            lat, lon = geo.validate_location(float(item["lat"]), float(item["lon"]))  # also drops results outside the map area
        except (KeyError, TypeError, ValueError, geo.LocationError):
            continue
        display = str(item.get("display_name") or "").strip()[:240]
        name = str(item.get("name") or "").strip()[:120] or (display.split(",")[0].strip() if display else "")
        if not name:
            continue
        key = (round(lat, 5), round(lon, 5))
        if key in seen:
            continue
        seen.add(key)
        places.append(Place(name=name, display_name=display or name, lat=lat, lon=lon, category=str(item.get("category") or "")[:40]))
    return places


def search_places(query, *, limit: int = MAX_RESULTS, http_get=None, clock=time.monotonic, sleep=time.sleep) -> list:
    """Search places in the Bengaluru area. Raises geo.LocationError (bad text) or GeocodeError (service problem).
    An empty list is a valid answer ("nothing found")."""
    text = geo.validate_query(query)
    limit = max(1, min(int(limit), MAX_RESULTS))
    key = (text.lower(), limit)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > clock():
            return list(hit[1])

    _limiter.wait(clock, sleep)
    min_lat, min_lon, max_lat, max_lon = geo.BENGALURU_BBOX
    params = {"q": text, "format": "jsonv2", "limit": limit, "countrycodes": "in", "accept-language": "en",
              "viewbox": f"{min_lon},{max_lat},{max_lon},{min_lat}", "bounded": 1}
    try:
        resp = (http_get or requests.get)(URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.Timeout:
        raise GeocodeError("timeout", "The search service did not answer in time. Please try again.") from None
    except requests.RequestException:
        raise GeocodeError("network", "Could not reach the search service. Check your internet connection and try again.") from None
    status = resp.status_code
    if status == 429:
        raise GeocodeError("rate_limited", "The search service asked us to slow down. Please wait a minute and try again.")
    if status != 200:
        raise GeocodeError("http", f"The search service returned an error (HTTP {status}).")
    try:
        payload = resp.json()
    except ValueError:
        raise GeocodeError("invalid_response", "The search service returned a response that could not be read.") from None
    places = _parse(payload)

    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (clock() + (CACHE_TTL_S if places else EMPTY_TTL_S), tuple(places))
    return places

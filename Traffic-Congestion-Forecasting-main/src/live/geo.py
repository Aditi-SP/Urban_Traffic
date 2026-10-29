"""
Location helpers for the live map: validation only. No coordinates of roads are stored or invented here.

BENGALURU_CENTER is the approximate city centre and is used ONLY to centre the initial map view.
BENGALURU_BBOX is an approximate rectangle around the Bengaluru area; it is the scope of the map and of searches.
Neither says anything about where any dataset road is.
"""
import math
import re

BENGALURU_CENTER = (12.9716, 77.5946)          # (lat, lon), approximate city centre: map view only
BENGALURU_BBOX = (12.70, 77.35, 13.25, 77.85)  # (min_lat, min_lon, max_lat, max_lon), approximate scope of this map

MIN_QUERY_LEN, MAX_QUERY_LEN = 3, 100


class LocationError(ValueError):
    """A location or search text that cannot be used; the message says why."""


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_location(lat, lon, bbox=BENGALURU_BBOX):
    """Return (lat, lon) as floats, or raise LocationError. Accepts real numbers only (no strings, no bool, no NaN)."""
    if not _is_number(lat) or not _is_number(lon):
        raise LocationError("Latitude and longitude must be real numbers.")
    if not -90 <= lat <= 90:
        raise LocationError(f"Latitude {lat} is outside -90 to 90.")
    if not -180 <= lon <= 180:
        raise LocationError(f"Longitude {lon} is outside -180 to 180.")
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            raise LocationError("This location is outside the Bengaluru area covered by this map "
                                f"(latitude {min_lat}-{max_lat}, longitude {min_lon}-{max_lon}).")
    return float(lat), float(lon)


def validate_query(text) -> str:
    """Clean a search text (strip, collapse spaces) or raise LocationError. Searches are explicit: no autocomplete."""
    if not isinstance(text, str):
        raise LocationError("Search text must be text.")
    clean = re.sub(r"\s+", " ", text).strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in clean):
        raise LocationError("Search text contains control characters.")
    if len(clean) < MIN_QUERY_LEN:
        raise LocationError(f"Type at least {MIN_QUERY_LEN} characters, then press Search.")
    if len(clean) > MAX_QUERY_LEN:
        raise LocationError(f"Search text is too long (maximum {MAX_QUERY_LEN} characters).")
    return clean


def distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))

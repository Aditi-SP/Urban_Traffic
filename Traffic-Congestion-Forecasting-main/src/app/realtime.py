"""
Real-time provider integration interface (TomTom Traffic Flow "Flow Segment Data" API).

Provider documentation checked: https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/traffic-flow/flow-segment-data
  GET https://api.tomtom.com/traffic/services/4/flowSegmentData/{style}/{zoom}/json?key=...&point=lat,lon&unit=kmph
  Response flowSegmentData: frc, currentSpeed, freeFlowSpeed, currentTravelTime (s), freeFlowTravelTime (s),
  confidence (0-1), roadClosure, coordinates.

IMPORTANT - compatibility with the trained models:
  The models need, for the last 7+ observations of a road: Congestion Level, Traffic Volume, Average Speed,
  Travel Time Index, Road Capacity Utilization, Incident Reports and Weather. The provider returns only a CURRENT
  speed / travel-time snapshot for one point: no traffic volume, no capacity utilisation, no incident counts, no
  weather, no history and no dataset-style Congestion Level. Therefore provider data can NOT be turned into a valid
  model input, and this module never does so; `prediction_blockers()` lists exactly what is missing.

Status: the request code below has NOT been run against the live service (no API key was available).
The API key is read from the TOMTOM_API_KEY environment variable or Streamlit secrets and is never hard-coded.
"""
import os
from datetime import datetime, timezone

import requests

ENV_KEY = "TOMTOM_API_KEY"
BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"

# How each model input relates to what the provider offers.
FEATURE_COVERAGE = [
    ("Average Speed", "currentSpeed (km/h)", "Partial: provider speed at one point; the dataset's definition of "
                                             "Average Speed is undocumented, so equivalence is unverified."),
    ("Travel Time Index", "currentTravelTime / freeFlowTravelTime", "Partial: derivable, but the dataset's index "
                                                                  "(range 1.0-1.5) is not documented, so equivalence is unverified."),
    ("Congestion Level", "-", "Not provided (the dataset's 0-100 definition is unknown)."),
    ("Traffic Volume", "-", "Not provided by the Flow Segment API."),
    ("Road Capacity Utilization", "-", "Not provided."),
    ("Incident Reports", "-", "Not in this endpoint (a separate Incidents API exists; counts are not comparable)."),
    ("Weather Conditions", "-", "Not provided (needs a separate weather source)."),
    ("7+ days of history per road", "-", "Not provided: the endpoint returns the current state only."),
    ("Road identity (16 dataset roads)", "-", "The dataset has no coordinates, so a dataset road cannot be mapped "
                                              "to a provider location without an external, verified source."),
]


class RealtimeConfigError(RuntimeError):
    """No API key configured."""


def get_api_key(secrets=None):
    """Environment variable first, then (optionally) a Streamlit-secrets-like mapping."""
    key = os.environ.get(ENV_KEY)
    if not key and secrets is not None:
        try:
            key = secrets.get(ENV_KEY)
        except Exception:  # st.secrets raises if no secrets file exists; that simply means 'not configured'
            key = None
    return key or None


def fetch_flow_segment(lat: float, lon: float, api_key: str, zoom: int = 10, timeout: float = 10.0) -> dict:
    """Fetch one provider observation. Raises on HTTP/format problems (no silent fallback)."""
    if not api_key:
        raise RealtimeConfigError(f"Set the {ENV_KEY} environment variable or Streamlit secret.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude must be in [-90, 90] and longitude in [-180, 180].")
    # requests' own error text contains the full URL, which includes `key=...`; never let it reach the UI or a log.
    try:
        resp = requests.get(BASE_URL.format(zoom=zoom), params={"key": api_key, "point": f"{lat},{lon}", "unit": "kmph"},
                            timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Provider returned HTTP {getattr(e.response, 'status_code', 'error')}. "
                           "Details are withheld because the raw error text contains the API key.") from None
    except requests.RequestException as e:
        raise RuntimeError(f"Provider request failed ({type(e).__name__}). "
                           "Details are withheld because the raw error text contains the API key.") from None
    seg = resp.json().get("flowSegmentData")
    required = ["currentSpeed", "freeFlowSpeed", "currentTravelTime", "freeFlowTravelTime", "confidence"]
    missing = [k for k in required if seg is None or k not in seg]
    if missing:
        raise ValueError(f"Provider response lacks expected fields {missing}.")
    return {
        "provider": "TomTom Traffic Flow (Flow Segment Data)",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observation_timestamp": None,  # the API returns no observation time, so freshness cannot be verified
        **{k: seg[k] for k in required},
        "roadClosure": seg.get("roadClosure"),
        "frc": seg.get("frc"),
    }


def prediction_blockers(observation: dict = None) -> list:
    """Why provider data cannot be fed to the trained models (never empty for this provider)."""
    return [f"{name}: {note}" for name, provider_field, note in FEATURE_COVERAGE if provider_field == "-"]

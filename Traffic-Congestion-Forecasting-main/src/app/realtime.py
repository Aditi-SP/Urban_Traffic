"""
What the TomTom provider can and cannot supply to the trained models (no network code lives here).

The ONLY TomTom client in this application is src/live/tomtom.py. Every page reaches it through ui/live_map.request_reading()
and the shared tomtom.SERVICE, which provides the request cap, refresh limit, 429 cooldown, cache setting and key handling.
An earlier stand-alone request function in this module had none of those and has been removed; do not add another client here.

IMPORTANT - compatibility with the trained models:
  The models need, for the last 7+ observations of a road: Congestion Level, Traffic Volume, Average Speed,
  Travel Time Index, Road Capacity Utilization, Incident Reports and Weather. The provider returns only a CURRENT
  speed / travel-time snapshot for one point: no traffic volume, no capacity utilisation, no incident counts, no
  weather, no history and no dataset-style Congestion Level. Therefore provider data can NOT be turned into a valid
  model input, and nothing here ever does so; `prediction_blockers()` lists exactly what is missing.

The API key is read from the TOMTOM_API_KEY environment variable or Streamlit secrets and is never hard-coded.
"""
import os

ENV_KEY = "TOMTOM_API_KEY"

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


def get_api_key(secrets=None):
    """Environment variable first, then (optionally) a Streamlit-secrets-like mapping."""
    key = os.environ.get(ENV_KEY)
    if not key and secrets is not None:
        try:
            key = secrets.get(ENV_KEY)
        except Exception:  # st.secrets raises if no secrets file exists; that simply means 'not configured'
            key = None
    return key or None


def prediction_blockers(observation: dict = None) -> list:
    """Why provider data cannot be fed to the trained models (never empty for this provider)."""
    return [f"{name}: {note}" for name, provider_field, note in FEATURE_COVERAGE if provider_field == "-"]

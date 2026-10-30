"""
Real status of the dashboard's parts, replacing the earlier hard-coded chips. Used by the sidebar, the homepage strip and Diagnostics.

Every item comes from something that was actually checked or observed, and says how:
  * traffic provider (EXTERNAL): whether a key is configured, and what the shared provider service has really seen in this run
    (last valid response, last failure, cooldown). No request is made to find out: a key that was never used is "not yet tested",
    never "working". Provider health is kept apart from the health of this application.
  * historical data: the latest observation date in the loaded dataset and its age. There is no live observation feed in this project.
  * models: whether the trained files exist (loading is only tested on the Traffic Forecasts page and in Diagnostics, because it is slow).
  * community reports: whether the local report database is usable.
A failed check is never shown as green, and messages carry no secrets, absolute paths or stack traces.
"""
import datetime as dt
import time
from dataclasses import dataclass

from src.live import credentials, tomtom

DATA_STALE_DAYS = 30
ROOT_HINT = "<project>"


@dataclass(frozen=True)
class Item:
    key: str
    label: str
    scope: str          # external | data | models | storage
    state: str          # ok | warn | bad | unknown
    chip: str           # short text for a chip
    summary: str        # one or two plain sentences
    detail: str = ""    # what exactly was checked
    value: str = ""     # a few words for a compact card


KIND = {"ok": "ok", "warn": "warn", "bad": "bad", "unknown": "mute"}
_ERROR_TEXT = {
    "auth": ("bad", "TomTom did not accept the key (invalid, expired, or not enabled for the Traffic API)"),
    "forbidden": ("bad", "TomTom's answer suggests this key may not be enabled for the Traffic Flow API (if so, enable the Traffic API for it in your TomTom account)"),
    "quota": ("warn", "TomTom's answer suggests the key's request quota may be used up"),
    "rate_limited": ("warn", "TomTom reported too many requests"),
    "timeout": ("warn", "TomTom did not answer in time"),
    "network": ("warn", "TomTom could not be reached from this computer"),
    "server": ("warn", "TomTom reported a temporary problem"),
    "invalid_response": ("bad", "TomTom sent a response this application could not accept"),
    "no_segment": ("warn", "TomTom has no traffic segment near the last point that was tried"),
}


def _safe(text, root=None) -> str:
    """One line, without the project path."""
    t = " ".join(str(text).split())
    if root:
        t = t.replace(str(root), ROOT_HINT)
    return t[:300]


def _clock(ts) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _age(seconds) -> str:
    return f"{int(seconds)} s ago" if seconds < 90 else f"{seconds / 60:.0f} min ago" if seconds < 5400 else f"{seconds / 3600:.1f} h ago"


def provider_item(has_key: bool, status: dict, now: float, hint: str = "") -> Item:
    if not has_key:
        return Item("provider", "Live traffic (TomTom)", "external", "bad", "Live traffic: not configured",
                    "No TomTom key is configured, so live traffic cannot be requested. Set TOMTOM_API_KEY (see the README); the map and place search still work."
                    + (" " + hint if hint else ""),
                    "Checked: the TOMTOM_API_KEY environment variable and .streamlit/secrets.toml. The value is never read into this panel.", "Not configured")
    ok_at, err = status.get("last_success_at"), status.get("last_error")
    cooldown = status.get("cooldown_remaining_s", 0)
    extra = f" Requests are paused for about {int(cooldown) + 1} s after a rate-limit response." if cooldown > 0 else ""
    detail = f"Checked: key present; this run has made {status.get('requests_made', 0)} provider request(s) (cap {status.get('budget')}). No test request is sent by this panel."
    if err and (ok_at is None or err[1] >= ok_at):
        state, text = _ERROR_TEXT.get(err[0], ("warn", "TomTom did not give a usable response"))
        last_ok = f" The last valid response was at {_clock(ok_at)}." if ok_at else " No valid response has been received yet."
        return Item("provider", "Live traffic (TomTom)", "external", state, "Live traffic: provider problem",
                    f"The last request failed: {text} ({_age(now - err[1])}).{last_ok}{extra}", detail, "Provider problem")
    if ok_at:
        return Item("provider", "Live traffic (TomTom)", "external", "ok", f"Live traffic: last response OK {_clock(ok_at)}",
                    f"The last valid provider response was received at {_clock(ok_at)} ({_age(now - ok_at)}, this computer's clock). "
                    f"TomTom gives no measurement time, so the traffic itself may be older.{extra}", detail, f"Last response {_clock(ok_at)}")
    return Item("provider", "Live traffic (TomTom)", "external", "unknown", "Live traffic: key found, not yet tested",
                "A key is configured, but no request has been made in this run, so it is not yet known whether the key works or whether TomTom has data "
                "for your point.", detail, "Key found, not tested")


def dataset_span(df):
    """(first, last) observation dates of the loaded dataset, as Timestamps."""
    from src.dataProcessing import forecast_data as fd
    return df[fd.DATE_COL].min(), df[fd.DATE_COL].max()


def data_item(df, error, now: float, root=None) -> Item:
    if error or df is None:
        return Item("data", "Historical dataset", "data", "bad", "Data: unavailable",
                    f"The historical dataset could not be used: {_safe(error or 'not loaded', root)}",
                    "Checked: that the dataset file loads and passes the project's validation.", "Unavailable")
    from src.dataProcessing import forecast_data as fd
    first, last = df[fd.DATE_COL].min(), df[fd.DATE_COL].max()
    age = (dt.date.fromtimestamp(now) - last.date()).days
    stale = age > DATA_STALE_DAYS
    return Item("data", "Historical dataset", "data", "warn" if stale else "ok",
                f"Data: historical, {first.year}-{last.year} (latest {last:%Y-%m-%d})",
                f"{len(df):,} rows, {first:%Y-%m-%d} to {last:%Y-%m-%d}. The latest observation is {age} days old"
                + ("; no newer observations are loaded, so replays and forecasts cannot reach recent dates." if stale else "."),
                "Checked: the dataset's own dates, not an external feed. This project has no live observation feed.", f"Latest {last:%Y-%m-%d}")


def models_item(missing_names, problems=(), root=None) -> Item:
    if missing_names:
        return Item("models", "Trained models", "models", "bad", "Models: unavailable",
                    "Trained model files are missing: " + ", ".join(sorted(missing_names)) + ". Forecasts are disabled; the rest of the dashboard works.",
                    "Checked: that each required model file exists.", "Unavailable")
    if problems:
        return Item("models", "Trained models", "models", "bad", "Models: could not load",
                    "A model file exists but could not be loaded: " + _safe(problems[0], root), "Checked: loading the models.", "Could not load")
    return Item("models", "Trained models", "models", "warn", "Models: not validated for 2026",
                "The trained model files are present (loading is tested on the Traffic Forecasts page). They are experimental: on held-out test data "
                "neither model beat the per-road historical mean, and they were never validated for 2026.",
                "Checked: file presence only. See Model Insights for the evaluation.", "Experimental")


def community_item(h: dict) -> Item:
    return Item("community", "Community reports", "storage", h["state"], "Reports: working" if h["state"] == "ok" else "Reports: unavailable",
                h["summary"][:300], "Checked: that the local report database can be opened.",
                ("Unavailable" if h["count"] is None else f"{h['count']} saved"))


def build_items(*, has_key, provider_status, df, data_error, models_missing, models_problems=(), community, now=None, root=None, key_hint="") -> list:
    now = time.time() if now is None else now
    return [provider_item(has_key, provider_status, now, key_hint), data_item(df, data_error, now, root),
            models_item(models_missing, models_problems, root), community_item(community)]


def collect(now=None) -> list:
    """The live status of this running application (cheap: nothing slow is loaded and no network request is made)."""
    from src.app import artifacts
    from src.community import store
    from src.ui import common
    df, err = common.dataset_status()
    cred = credentials.load()
    return build_items(has_key=cred.configured, provider_status=tomtom.SERVICE.status(), df=df, data_error=err, key_hint=cred.hint_text(),
                       models_missing=[n for _, n, _ in artifacts.missing()], community=store.health(), now=now, root=common.PROJECT_ROOT)

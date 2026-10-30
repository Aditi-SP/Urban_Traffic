"""
Section 1: Live Traffic Map (Stage 1).

What is on the map
  * a real open basemap (OpenStreetMap tiles drawn by an embedded Leaflet map, attribution shown), centred on Bengaluru
  * the place the user searched for and chose (a search result from OpenStreetMap's Nominatim, or coordinates the user typed)
  * the traffic segment returned by TomTom for that place, ONLY after a real, valid provider response
Nothing else is ever drawn: no sample traffic, no markers for dataset roads (the dataset has no coordinates), no simulated values.

What this page does not do
  * it does not store provider data (in-memory cache only, until the provider's terms are verified)
  * it does not send the API key to the browser (all provider calls are made by the server)
  * it does not feed the readings to the forecasting models: those models need inputs this provider does not supply
"""
import html
import time
from dataclasses import asdict

import streamlit as st

from src.live import geo, geocode, tomtom
from src.live import credentials
from src.ui import health, leaflet_map, theme
from src.ui.common import METRICS_JSON, get_data, read_json

LAYERS = ["Traffic segment", "Selected place"]
MAP_HEIGHT = 500
SEVERITY = {"no_key": "error", "auth": "error", "forbidden": "error", "quota": "warning", "invalid_response": "error", "no_segment": "info",
            "timeout": "warning", "network": "warning", "server": "warning", "rate_limited": "warning", "budget": "warning", "throttled": "info"}
BAND_TEXT = {"free": ("Free-flowing", "ok", theme.GREEN), "slow": ("Slow", "warn", theme.AMBER),
             "congested": ("Congested", "bad", theme.RED), "closed": ("Road closed", "bad", theme.RED),
             "stale": ("Stale reading", "mute", theme.MUTED)}
S = st.session_state


# ------------------------------------------------------------------------------------------------ map figure
def clock_text(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def age_text(seconds: float) -> str:
    return f"{int(seconds)} s ago" if seconds < 90 else f"{seconds / 60:.0f} min ago"


def build_map_spec(selected, obs, layers, now=None) -> dict:
    """What the map shows. With nothing selected there are no markers and no lines: it is just the basemap."""
    now = time.time() if now is None else now
    markers, lines, bounds = [], [], None
    center, zoom = list(geo.BENGALURU_CENTER), 11

    if selected and "Selected place" in layers:
        center, zoom = [selected["lat"], selected["lon"]], 15
        markers.append({"lat": selected["lat"], "lon": selected["lon"], "color": theme.BLUE,
                        "tooltip": html.escape(str(selected["name"]))[:120]})
    if obs is not None and "Traffic segment" in layers:
        b = tomtom.band(obs, now)
        pts = [[la, lo] for la, lo in obs.segment]
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
        tooltip = (f"Current speed: {obs.current_speed_kmh:.0f} km/h<br>Free-flow speed: {obs.free_flow_speed_kmh:.0f} km/h<br>"
                   f"{obs.speed_ratio:.0%} of free-flow ({BAND_TEXT[b][0]})<br>Last fetched: {clock_text(obs.fetched_at)} "
                   "(this computer's clock)<br>Provider measurement time: not provided")
        lines.append({"points": pts, "color": BAND_TEXT[b][2], "tooltip": tooltip})
    return leaflet_map.build_spec(center, zoom, markers, lines, bounds)


# ------------------------------------------------------------------------------------------------ actions
def _search(text):
    S["lm_geo_msg"] = None
    try:
        places = geocode.search_places(text)
    except geo.LocationError as e:
        S["lm_geo_msg"] = ("warning", str(e))
        return
    except geocode.GeocodeError as e:
        S["lm_geo_msg"] = ("error" if e.kind in ("http", "invalid_response") else "warning", e.message)
        return
    S["lm_results"] = [asdict(p) for p in places]
    S.pop("lm_pick", None)  # a new search resets the choice (this runs before the select box is created)
    S["lm_last_pick"], S["lm_selected"], S["lm_obs"], S["lm_msg"] = None, None, None, None
    if not places:
        S["lm_geo_msg"] = ("info", "No matching place found in the Bengaluru area. Try a different or more specific name.")


def _use_coordinates():
    """Callback of the 'Use these' button: reads the two number inputs and validates them."""
    try:
        la, lo = geo.validate_location(S.get("lm_lat"), S.get("lm_lon"))
    except geo.LocationError as e:
        S["lm_geo_msg"] = ("error", str(e))
        return
    S["lm_geo_msg"] = None
    S["lm_selected"] = {"name": f"Coordinates {la:.5f}, {lo:.5f}", "display_name": f"Coordinates entered by you: {la:.5f}, {lo:.5f}",
                        "lat": la, "lon": lo, "category": "manual"}
    S["lm_obs"], S["lm_msg"] = None, None


def _request_fetch():
    S["lm_fetch_req"] = True  # handled at the top of the next run so a spinner and the updated map appear together


def _clear():
    for k in ("lm_selected", "lm_obs", "lm_msg", "lm_results", "lm_geo_msg", "lm_last_pick", "lm_pick"):
        S.pop(k, None)


def request_reading(lat, lon, api_key, force=False):
    """The ONE way pages ask for a live reading: a call into the shared tomtom.SERVICE (cache setting, refresh limit, request
    cap, 429 cooldown) and one translation of its errors into messages.

    Used by the Live Traffic Map and by Mode 2 of the Traffic Forecasts page, so both are under the same safeguards.
    Returns (observation, None-or-message) on success and (None, (severity, text)) on any failure; a failure never produces data.
    """
    try:
        with st.spinner("Contacting TomTom..."):
            obs, source = tomtom.SERVICE.get(lat, lon, api_key, force=force)
    except geo.LocationError as e:
        return None, ("error", str(e))
    except tomtom.ProviderError as e:
        return None, (SEVERITY.get(e.kind, "error"), e.message)
    except Exception as e:  # noqa: BLE001  never crash the page; only the type is shown (raw text could contain the request URL)
        return None, ("error", f"Unexpected problem while requesting traffic ({type(e).__name__}). No reading was produced.")
    note = ("info", "Refreshes are limited to once every 30 seconds; this is the reading fetched a moment ago.") \
        if (source == "cache" and force) else None
    return obs, note


def _do_fetch(api_key):
    sel = S.get("lm_selected")
    if not sel:
        return
    force = S.get("lm_obs") is not None
    obs, msg = request_reading(sel["lat"], sel["lon"], api_key, force)
    if obs is not None:  # on failure an earlier reading (if any) stays on screen, aged, so it cannot look current
        S["lm_obs"] = obs
    S["lm_msg"] = msg


def _show(msg):
    if msg:
        getattr(st, msg[0])(msg[1])


# ------------------------------------------------------------------------------------------------ page
def _controls():
    results = S.get("lm_results") or []
    with st.form("lm_search", border=False):
        a, b = st.columns([6, 1])
        text = a.text_input("Search a road, junction or place in Bengaluru", key="lm_query", label_visibility="collapsed",
                            placeholder="Search a road, junction or place in Bengaluru, then press Search", max_chars=geo.MAX_QUERY_LEN)
        submitted = b.form_submit_button("Search", use_container_width=True)
    if submitted:
        _search(text)
        results = S.get("lm_results") or []

    r1, r2 = st.columns([5, 3])
    with r1:
        if results:
            pick = st.selectbox("Search results", list(range(len(results))), index=None, key="lm_pick",
                                placeholder="Choose the place you meant", label_visibility="collapsed",
                                format_func=lambda i: results[i]["display_name"][:110] if isinstance(i, int) else str(i))
            if pick is not None and pick != S.get("lm_last_pick"):
                S["lm_last_pick"], S["lm_selected"], S["lm_obs"], S["lm_msg"] = pick, results[pick], None, None
        else:
            st.caption("Search results will appear here. Searches run only when you press Search (no autocomplete).")
    with r2:
        layers = st.pills("Layers", LAYERS, selection_mode="multi", default=LAYERS, key="lm_layers", label_visibility="collapsed")
    _show(S.get("lm_geo_msg"))
    return layers or []


def _card(*a):
    return theme.card_html(*a, compact=True)


def _panel(has_key, sel, obs, now):
    """Right-hand column: provider status, chosen place, the reading (with its age) and the actions."""
    theme_cards = []
    prov = health.provider_item(has_key, tomtom.SERVICE.status(), now)
    theme_cards.append(_card("Traffic provider", prov.value, ("Add a key to see traffic (see the message above). The map and search work without it." if not has_key
                                                              else prov.summary), health.KIND[prov.state] if prov.state != "ok" else "ok"))
    theme_cards.append(_card("Selected place", (sel["name"][:38] + "...") if sel and len(sel["name"]) > 38 else (sel["name"] if sel else "None"),
                                       f"{sel['lat']:.5f}, {sel['lon']:.5f}  |  " + ("typed by you" if sel["category"] == "manual" else "OpenStreetMap search, chosen by you")
                                       if sel else "Search for a place and choose a result.", "info" if sel else "mute"))
    if obs is not None:
        b = tomtom.band(obs, now)
        label, kind, _ = BAND_TEXT[b]
        age = tomtom.age_s(obs, now)
        theme_cards.append(_card("Current speed", f"{obs.current_speed_kmh:.0f} km/h",
                                           f"{obs.speed_ratio:.0%} of the free-flow speed ({obs.free_flow_speed_kmh:.0f} km/h)  |  {label}", kind))
        theme_cards.append(_card("Travel time on this segment", f"{obs.current_travel_time_s / 60:.1f} min",
                                           f"free-flow {obs.free_flow_travel_time_s / 60:.1f} min  |  provider confidence {obs.confidence:.0%}"
                                           + ("  |  ROAD CLOSED" if obs.road_closure else ""), "info"))
        theme_cards.append(_card("Last fetched", clock_text(obs.fetched_at),
                                           f"{age_text(age)} on this computer's clock. The provider's own measurement time is not "
                                           "provided by this API, so the data may be older than this." + ("  STALE: fetch again." if tomtom.is_stale(obs, now) else ""),
                                           "bad" if tomtom.is_stale(obs, now) else "ok"))
    else:
        note = ("Press the button to request it." if (has_key and sel) else "Choose a place first." if has_key
                else "Add a TomTom key to request traffic for a place.")
        theme_cards.append(_card("Traffic reading", "None", "No reading yet. " + note, "mute"))
    st.markdown("<div style='display:flex;flex-direction:column;gap:.6rem'>" + "".join(theme_cards) + "</div>", unsafe_allow_html=True)
    st.write("")
    st.button("Refresh traffic" if obs is not None else "Get traffic for this place", key="lm_fetch_btn", type="primary",
              disabled=not (has_key and sel), on_click=_request_fetch, use_container_width=True)
    st.button("Clear selection", key="lm_clear_btn", on_click=_clear, use_container_width=True, disabled=sel is None)
    _show(S.get("lm_msg"))
    with st.expander("Enter coordinates instead"):
        st.number_input("Latitude", value=geo.BENGALURU_CENTER[0], format="%.6f", step=0.001, key="lm_lat")
        st.number_input("Longitude", value=geo.BENGALURU_CENTER[1], format="%.6f", step=0.001, key="lm_lon")
        st.button("Use these coordinates", key="lm_use_coords", on_click=_use_coordinates, use_container_width=True)
        st.caption("Used exactly as entered; the starting values are only the approximate city centre.")
    st.caption("Readings are shown on their own. They are not used by the forecasting models, which need inputs (traffic volume, "
               "capacity, incidents, weather, history) that this provider does not supply.")


def _link(path, label):
    """A link to another page; where the page router is not running (a page tested on its own) it says where to find it instead."""
    try:
        st.page_link(path, label=label, icon=":material/arrow_forward:")
    except Exception:  # noqa: BLE001
        st.caption(f"Open \"{label}\" from the sidebar.")


def _model_note() -> str:
    """One verified sentence from the saved evaluation, or an honest statement that it is missing."""
    try:
        info = read_json(METRICS_JSON)
        m = info["metrics"]
        base, lstm, xgb = m["Road-mean baseline"]["RMSE"], m["LSTM"]["RMSE"], m["XGBoost"]["RMSE"]
    except Exception:  # noqa: BLE001  a missing or unreadable file must not break the homepage
        return "The saved evaluation (model/metrics.json) was not found, so no model figures are shown."
    verdict = "Neither model beat the road mean." if min(lstm, xgb) >= base else "See Model Insights for the comparison."
    return (f"Held-out test RMSE (congestion points; lower is better): road mean {base:.2f}, LSTM {lstm:.2f}, XGBoost {xgb:.2f}. "
            f"{verdict} Not validated for 2026.")


def _overview(now):
    """Homepage strip: only verified status, with links to the detail pages. Nothing here is a live measurement or a forecast."""
    items = {i.key: i for i in health.collect(now)}
    theme.section("At a glance", "Verified status only. Open a page for the details.")
    c1, c2, c3, c4 = st.columns(4)
    kind = lambda i: health.KIND[i.state] if i.state != "ok" else "ok"
    c1.markdown(_card("Live traffic", items["provider"].value, items["provider"].summary[:170], kind(items["provider"])), unsafe_allow_html=True)
    c2.markdown(_card("Historical data", items["data"].value, items["data"].summary[:170], kind(items["data"])), unsafe_allow_html=True)
    with c2:
        _link("src/ui/pages/forecasts.py", "Traffic Forecasts")
    c3.markdown(_card("Models", items["models"].value, _model_note(), "warn" if items["models"].state != "bad" else "bad"), unsafe_allow_html=True)
    with c3:
        _link("src/ui/pages/performance.py", "Model Insights")
    c4.markdown(_card("Community reports", items["community"].value, items["community"].summary[:170], kind(items["community"])), unsafe_allow_html=True)
    with c4:
        _link("src/ui/pages/community.py", "Community Reports")


SETUP_COMMANDS = """$secure = Read-Host "Paste your TomTom key (input is hidden)" -AsSecureString
$plain  = [System.Net.NetworkCredential]::new('', $secure).Password
[Environment]::SetEnvironmentVariable("TOMTOM_API_KEY", $plain, "User")
Remove-Variable plain, secure"""


def _setup_notice(cred):
    """A SHORT warning (so the map stays on screen, also in a narrow window) and, collapsed below it, the full setup steps and the reasons."""
    short = cred.short_hint()
    st.warning("**Live traffic is not configured.** It needs a TomTom key (`TOMTOM_API_KEY`). " + (short + " " if short else "") + "See the README.")
    with st.expander("Set up the key / why it was not found"):
        st.markdown("**1. Save the key once** (PowerShell; the prompt hides what you type, and the key is never shown or sent to your browser):")
        st.code(SETUP_COMMANDS, language="powershell")
        st.markdown("**2. Restart the dashboard with the launcher** (it loads the saved key even from an older terminal): stop the running one with "
                    "Ctrl+C, then run `.\\run_dashboard.ps1`. Or put `TOMTOM_API_KEY = \"your-key\"` in `.streamlit/secrets.toml` (excluded from Git). "
                    "The key stays on the server. Checks and error meanings: README, \"Setting up TOMTOM_API_KEY\".")
        if cred.hint_text():
            st.markdown("**Why it was not found:** " + cred.hint_text())


def render():
    cred = credentials.load()
    has_key = cred.configured
    obs, sel = S.get("lm_obs"), S.get("lm_selected")
    chip = (("Traffic provider key found", "warn") if has_key else ("Live traffic not configured", "bad"))
    theme.page_header("Live Traffic Map", "Search a place in Bengaluru; traffic appears only from a real provider response",
                      chips=[chip, ("Map: OpenStreetMap", "mute")], compact=True)

    if not has_key:
        _setup_notice(cred)
    layers = _controls()
    if S.pop("lm_fetch_req", False) and has_key:
        _do_fetch(cred.key)
    obs, sel, now = S.get("lm_obs"), S.get("lm_selected"), time.time()

    map_col, panel_col = st.columns([7, 3], gap="medium")
    with map_col:
        leaflet_map.show(build_map_spec(sel, obs, layers, now), MAP_HEIGHT)
        parts = ["Map: © OpenStreetMap contributors (ODbL)"]
        if S.get("lm_results"):
            parts.append(geocode.ATTRIBUTION)
        if obs is not None:
            parts.append(tomtom.ATTRIBUTION)
        st.caption("  |  ".join(parts))
        theme.legend_bands("Colour bands are this app's display convention (current speed as a share of free-flow speed: free-flowing "
                           "80% or more, slow 50-80%, congested below 50%). They are not provider categories and not validated.")
    with panel_col:
        _panel(has_key, sel, obs, now)

    _overview(now)
    df = get_data()
    if df is not None:
        first, last = health.dataset_span(df)
        st.caption(f"The historical dataset ({first.year}-{last.year}) has no coordinates, so none of its roads are placed on this map.")

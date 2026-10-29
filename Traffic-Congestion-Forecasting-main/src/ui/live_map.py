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
from src.ui import leaflet_map, theme
from src.ui.common import get_data, tomtom_key

LAYERS = ["Traffic segment", "Selected place"]
MAP_HEIGHT = 500
SEVERITY = {"no_key": "error", "auth": "error", "invalid_response": "error", "no_segment": "info",
            "timeout": "warning", "network": "warning", "server": "warning", "rate_limited": "warning", "budget": "warning"}
BAND_TEXT = {"free": ("Free-flowing", "ok", theme.GREEN), "slow": ("Slow", "warn", theme.AMBER),
             "congested": ("Congested", "bad", theme.RED), "closed": ("Road closed", "bad", theme.RED),
             "stale": ("Stale reading", "mute", theme.MUTED)}
S = st.session_state


# ------------------------------------------------------------------------------------------------ map figure
def _clock_text(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _age_text(seconds: float) -> str:
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
                   f"{obs.speed_ratio:.0%} of free-flow ({BAND_TEXT[b][0]})<br>Last fetched: {_clock_text(obs.fetched_at)} "
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


def _do_fetch(api_key):
    sel = S.get("lm_selected")
    if not sel:
        return
    force = S.get("lm_obs") is not None
    try:
        with st.spinner("Contacting TomTom..."):
            obs, source = tomtom.SERVICE.get(sel["lat"], sel["lon"], api_key, force=force)
    except geo.LocationError as e:
        S["lm_msg"] = ("error", str(e))
    except tomtom.ProviderError as e:  # an earlier successful reading (if any) stays on screen, aged, so it cannot look current
        S["lm_msg"] = (SEVERITY.get(e.kind, "error"), e.message)
    else:
        S["lm_obs"] = obs
        S["lm_msg"] = ("info", "Refreshes are limited to once every 30 seconds; this is the reading fetched a moment ago.") \
            if (source == "cache" and force) else None


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
    theme_cards.append(_card("Traffic provider", "TomTom" if has_key else "Not configured",
                                       "Key found on the server; it is never sent to your browser." if has_key
                                       else "The map and search work without it.", "warn" if has_key else "bad"))
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
        theme_cards.append(_card("Last fetched", _clock_text(obs.fetched_at),
                                           f"{_age_text(age)} on this computer's clock. The provider's own measurement time is not "
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
    if not has_key:
        theme.callout("<b>Live traffic is not configured.</b> Add a free TomTom key: set the <code>TOMTOM_API_KEY</code> environment "
                      "variable, or put it in <code>.streamlit/secrets.toml</code> (excluded from Git). It stays on the server.")
    st.caption("Readings are shown on their own. They are not used by the forecasting models, which need inputs (traffic volume, "
               "capacity, incidents, weather, history) that this provider does not supply.")


def render():
    has_key = bool(tomtom_key())
    obs, sel = S.get("lm_obs"), S.get("lm_selected")
    chip = (("Traffic provider key found", "warn") if has_key else ("Live traffic not configured", "bad"))
    theme.page_header("Live Traffic Map", "Search a place in Bengaluru; traffic appears only from a real provider response",
                      chips=[chip, ("Map: OpenStreetMap", "mute")], compact=True)

    layers = _controls()
    if S.pop("lm_fetch_req", False) and has_key:
        _do_fetch(tomtom_key())
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

    df = get_data()
    if df is not None:
        st.caption("The historical dataset (2022-2024) has no coordinates, so none of its roads are placed on this map.")

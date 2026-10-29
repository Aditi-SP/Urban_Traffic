"""
An embedded interactive map: Leaflet 1.9.4 (vendored, BSD 2-Clause) drawing OpenStreetMap tiles.

Why not Plotly: the pinned Streamlit 1.40 cannot draw Plotly's current map traces, and its older mapbox traces were removed
from plotly.py 7. Leaflet is a small, standard open-source library; it is vendored in src/ui/vendor/leaflet and verified
against the integrity hashes published on leafletjs.com each time, so the browser loads no script from a CDN.

Basemap: OpenStreetMap raster tiles from tile.openstreetmap.org. Policy (https://operations.osmfoundation.org/policies/tiles/,
read 2026-10-28): attribution must be clearly visible (Leaflet's attribution control is always shown, bottom right), tiles are
only fetched for what the user is looking at (no prefetching or offline use), the browser sends its own User-Agent/Referer,
and heavy use is not allowed (this is a local demo; use a commercial tile provider before any wider deployment).

Only plain data goes to the browser (positions, colours, text). No API key or provider URL is ever included.
"""
import base64
import functools
import hashlib
import json
from pathlib import Path

import streamlit.components.v1 as components

VENDOR = Path(__file__).resolve().parent / "vendor" / "leaflet"
SRI = {"leaflet.js": "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
       "leaflet.css": "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="}
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors'

OFFLINE_NOTE = "Map tiles could not be loaded. Check your internet connection. Anything drawn on the map is still real data."


def _sri(path: Path) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()


@functools.lru_cache(maxsize=1)
def _assets():
    """(css, js) text of the vendored Leaflet, after checking the published integrity hashes."""
    for name, expected in SRI.items():
        if _sri(VENDOR / name) != expected:
            raise RuntimeError(f"Vendored {name} does not match the published Leaflet 1.9.4 hash; refusing to use it.")
    return (VENDOR / "leaflet.css").read_text(encoding="utf-8"), (VENDOR / "leaflet.js").read_text(encoding="utf-8")


def build_spec(center, zoom, markers=(), lines=(), bounds=None) -> dict:
    """Plain, JSON-serialisable description of the map (also what the tests inspect). Empty markers/lines = just the basemap."""
    return {"center": [float(center[0]), float(center[1])], "zoom": zoom, "bounds": bounds,
            "markers": list(markers), "lines": list(lines),
            "tiles": {"url": TILE_URL, "attribution": TILE_ATTRIBUTION}, "offline_note": OFFLINE_NOTE}


_APP_JS = """
const spec = JSON.parse(document.getElementById('spec').textContent);
const map = L.map('map', {scrollWheelZoom: true}).setView(spec.center, spec.zoom);
const tiles = L.tileLayer(spec.tiles.url, {maxZoom: 19, attribution: spec.tiles.attribution});
let loaded = 0, failed = 0;
tiles.on('tileload', () => { loaded++; document.getElementById('offline').style.display = 'none'; });
tiles.on('tileerror', () => { failed++; if (loaded === 0 && failed >= 3) { document.getElementById('offline').style.display = 'block'; } });
tiles.addTo(map);
spec.lines.forEach(l => {
  const p = L.polyline(l.points, {color: l.color, weight: 8, opacity: 0.95}).addTo(map);
  p.bindTooltip(l.tooltip, {sticky: true});
});
spec.markers.forEach(m => {
  const c = L.circleMarker([m.lat, m.lon], {radius: 9, color: '#ffffff', weight: 2, fillColor: m.color, fillOpacity: 1}).addTo(map);
  c.bindTooltip(m.tooltip);
});
if (spec.bounds) { map.fitBounds(spec.bounds, {padding: [60, 60], maxZoom: 17}); }
document.getElementById('offline').textContent = spec.offline_note;
"""


def html_document(spec: dict, height: int = 540) -> str:
    css, js = _assets()
    data = json.dumps(spec).replace("</", "<\\/")  # a "</script>" inside the data can never close the script tag
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="strict-origin-when-cross-origin"><style>{css}
html, body {{ margin: 0; height: 100%; background: #0B1426; }}
#map {{ height: {int(height)}px; width: 100%; border-radius: 12px; }}
#offline {{ display: none; position: absolute; z-index: 1000; left: 12px; right: 12px; bottom: 34px; padding: 8px 12px; border-radius: 8px;
           background: rgba(11,20,38,.92); color: #E6EDF7; font: 13px/1.4 sans-serif; border: 1px solid #F59E0B; }}
.leaflet-tooltip {{ font: 13px/1.4 sans-serif; }}
</style></head><body><div id="map"></div><div id="offline"></div>
<script type="application/json" id="spec">{data}</script>
<script>{js}</script><script>{_APP_JS}</script></body></html>"""


def show(spec: dict, height: int = 540):
    """Render the map in the page (an iframe). Tests replace this function to capture the spec."""
    components.html(html_document(spec, height), height=height, scrolling=False)

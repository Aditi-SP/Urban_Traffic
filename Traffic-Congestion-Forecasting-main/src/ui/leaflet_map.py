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

Community photo markers (kind "community") carry a small JPEG thumbnail as a data URL that this application produced itself, and the reporter's
description. The popup is built from DOM text nodes (textContent), never from HTML, so a description can never inject markup. They are drawn as
purple camera markers, clearly different from the provider's coloured traffic line.
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


def build_spec(center, zoom, markers=(), lines=(), bounds=None, circles=()) -> dict:
    """Plain, JSON-serialisable description of the map (also what the tests inspect). Empty markers/lines = just the basemap."""
    return {"center": [float(center[0]), float(center[1])], "zoom": zoom, "bounds": bounds,
            "markers": list(markers), "lines": list(lines), "circles": list(circles),
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
(spec.circles || []).forEach(c => {
  L.circle([c.lat, c.lon], {radius: c.radius_m, color: c.color, weight: 1.5, dashArray: '6 6', fillColor: c.color, fillOpacity: 0.05, interactive: false}).addTo(map);
});
function line(parent, text, cls) {
  if (!text) { return; }
  const d = document.createElement('div'); if (cls) { d.className = cls; } d.textContent = text; parent.appendChild(d);
}
function popupFor(p) {
  const root = document.createElement('div'); root.className = 'tc-popup';
  line(root, p.source, 'src');
  if (typeof p.image === 'string' && p.image.indexOf('data:image/jpeg;base64,') === 0) {
    const img = document.createElement('img'); img.src = p.image; img.alt = 'Community traffic photo'; root.appendChild(img);
  }
  line(root, p.description, 'desc'); line(root, p.uploaded, 'when'); line(root, p.state, p.fresh ? 'fresh' : 'expired');
  line(root, p.distance); line(root, p.location);
  return root;
}
spec.markers.forEach(m => {
  if (m.kind === 'community') {
    const icon = L.divIcon({className: 'tc-photo-marker' + (m.fresh ? '' : ' expired'), html: '<span>&#128247;</span>', iconSize: [34, 34], iconAnchor: [17, 17], popupAnchor: [0, -16]});
    const mk = L.marker([m.lat, m.lon], {icon: icon, title: m.title || 'Community photo', keyboard: true}).addTo(map);
    mk.bindPopup(popupFor(m.popup || {}), {maxWidth: 290, minWidth: 230});
    return;
  }
  const c = L.circleMarker([m.lat, m.lon], {radius: 9, color: '#ffffff', weight: 2, fillColor: m.color, fillOpacity: 1}).addTo(map);
  c.bindTooltip(m.tooltip);
});
if (spec.bounds) { map.fitBounds(spec.bounds, {padding: [60, 60], maxZoom: 17}); }
document.getElementById('offline').textContent = spec.offline_note;
"""


def html_document(spec: dict, height: int = 540) -> str:
    css, js = _assets()
    # Every <, > and & in the data becomes a JSON \u escape (still the same JSON): neither "</script>" nor "<!--" nor "<script" in a description can ever
    # change how the HTML parser reads the page.
    data = json.dumps(spec).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="strict-origin-when-cross-origin"><style>{css}
html, body {{ margin: 0; height: 100%; background: #0B1426; }}
#map {{ height: {int(height)}px; width: 100%; border-radius: 12px; }}
#offline {{ display: none; position: absolute; z-index: 1000; left: 12px; right: 12px; bottom: 34px; padding: 8px 12px; border-radius: 8px;
           background: rgba(11,20,38,.92); color: #E6EDF7; font: 13px/1.4 sans-serif; border: 1px solid #F59E0B; }}
.leaflet-tooltip {{ font: 13px/1.4 sans-serif; }}
.tc-photo-marker span {{ display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 9px; background: #8B5CF6;
                         border: 2px solid #fff; box-shadow: 0 1px 5px rgba(0,0,0,.5); font-size: 18px; }}
.tc-photo-marker.expired span {{ background: #64748B; opacity: .8; }}
.tc-popup {{ font: 13px/1.4 sans-serif; color: #111827; }}
.tc-popup img {{ display: block; width: 100%; border-radius: 6px; margin: 4px 0; }}
.tc-popup .src {{ font-size: 11px; color: #6D28D9; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }}
.tc-popup .desc {{ font-weight: 600; }} .tc-popup .when {{ font-weight: 700; }}
.tc-popup .fresh {{ color: #15803D; font-weight: 700; }} .tc-popup .expired {{ color: #6B7280; font-weight: 700; }}
</style></head><body><div id="map"></div><div id="offline"></div>
<script type="application/json" id="spec">{data}</script>
<script>{js}</script><script>{_APP_JS}</script></body></html>"""


def show(spec: dict, height: int = 540):
    """Render the map in the page (an iframe). Tests replace this function to capture the spec."""
    components.html(html_document(spec, height), height=height, scrolling=False)

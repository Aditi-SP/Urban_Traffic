"""
The browser-side capture of a photo report: the person's GPS position (from the browser's own geolocation), a photo from the camera or the
gallery, in ONE small Streamlit component (src/ui/components/photo_capture/index.html, plain HTML and JavaScript, no build step).

The component only collects. Everything it reports is checked again on the server (src/community/photos.py), which cannot trust a browser:
the position is "reported by the device", never "verified". The tests replace `capture` with a function that returns a prepared value.
"""
from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).resolve().parent / "components" / "photo_capture"
_COMPONENT = components.declare_component("tc_photo_capture", path=str(_DIR))


def capture(key: str):
    """Show the component and return its latest value (a dict, see photos.explain_capture) or None before the first message."""
    return _COMPONENT(key=key, default=None)

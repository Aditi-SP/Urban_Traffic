"""
How a community photo report is shown (used by the Community Reports page and by the Live Traffic Map).

Everything user-written is escaped plain text; the photo is shown from the application's own re-encoded JPEG; the reporter is never shown; the position
is labelled as reported by the device. Fresh and expired reports look different and say so in words, not only in colour.
"""
import html

import streamlit as st

from src.community import photos
from src.ui import theme

SOURCE_LABEL = "Community photo (sent by a road user, not provider traffic data)"


def photo_card(report, *, now, distance_km=None, show_image=True):
    """One photo report as an image next to its text block."""
    esc = lambda t: html.escape(str(t), quote=True)
    fresh = report.is_fresh(now)
    img = photos.read_image(report) if show_image else None
    left, right = st.columns([2, 3], gap="medium")
    with left:
        if img:
            st.image(img, use_container_width=True)
        else:
            st.caption("The photo was removed after the retention time." if not report.image_file else "The photo file is not available.")
    with right:
        badge = theme.chip_html("Fresh" if fresh else "Expired", "ok" if fresh else "mute")
        lines = [f'<div class="tc-chips" style="margin-bottom:.3rem">{badge}{theme.chip_html("Unverified", "warn")}</div>',
                 f'<div style="font-size:1.05rem;font-weight:600">{esc(report.description) if report.description else "<em>No description</em>"}</div>',
                 f'<div class="note"><b>{esc(photos.uploaded_text(report, now))}</b> ({esc(report.created_utc.replace("T", " ").replace("+00:00", " UTC"))})</div>',
                 f'<div class="note">{esc(photos.state_text(report, now))}</div>']
        if distance_km is not None:
            lines.append(f'<div class="note">{esc(photos.distance_text(distance_km))}</div>')
        lines.append(f'<div class="note">Location: {report.latitude:.4f}, {report.longitude:.4f} (reported by the device, about {report.accuracy_m:.0f} m accuracy; not verified)</div>')
        st.markdown("".join(lines), unsafe_allow_html=True)


def marker_for(item, now, *, path=None):
    """The map marker (plain data) for one NearbyPhoto; the popup text is built from text nodes in the browser."""
    r = item.report
    return {"kind": "community", "lat": r.latitude, "lon": r.longitude, "fresh": r.is_fresh(now), "title": "Community photo",
            "popup": {"source": SOURCE_LABEL, "image": photos.thumbnail_data_url(r, path=path), "description": r.description or "No description",
                      "uploaded": photos.uploaded_text(r, now), "state": photos.state_text(r, now), "fresh": r.is_fresh(now),
                      "distance": photos.distance_text(item.distance_km), "location": photos.PROVENANCE}}

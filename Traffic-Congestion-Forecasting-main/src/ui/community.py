"""
Section 3: Community Reports (Stage 0 = navigation page and empty state only).

No upload, database or storage exists yet, and no sample reports are shown because none exist.
"""
import streamlit as st

from src.ui import theme


def render():
    theme.page_header(
        "Community Reports",
        "Photos and short reports from other road users, for the same area",
        chips=[("Not implemented yet", "warn"), ("Nothing is stored", "mute")])

    theme.stat_cards([
        {"label": "Reports", "value": "Not available", "kind": "mute", "note": "No database exists yet, so there is nothing to count."},
        {"label": "Photo upload", "value": "Disabled", "kind": "mute", "note": "This page cannot upload or save anything yet."},
        {"label": "Sharing scope", "value": "This computer", "kind": "info", "note": "Planned: reports shared only with sessions of the same running server."},
        {"label": "Identity", "value": "Anonymous", "kind": "info", "note": "Planned: no accounts; safeguards instead of verified users."},
    ])

    theme.section("Recent reports")
    theme.empty_state(
        "Community photo reports are not available yet",
        "Nothing can be uploaded or saved from this page. No sample reports are shown, because none exist.",
        bullets=["Planned: share a traffic photo with a category and an optional short description for a chosen area.",
                 "Planned: browse recent reports with visible age and stale-report warnings.",
                 "Planned: flag misleading reports, and mark reports as resolved or outdated."],
        min_height=300)

    with st.expander("Planned safeguards (not implemented in this stage)"):
        st.markdown(
            "- Accept only JPEG, PNG or WebP images; limit file size; re-encode them and strip location metadata; use random file names.\n"
            "- Rate limits and duplicate detection; a visible age on every report and a stale warning for old ones.\n"
            "- A photo will never be presented as verification of an accident, and will never be used to train models without a "
            "separate, explicit consent and data-governance plan.\n"
            "- No precise location tracking: a named place or a coarse map area only.")

"""
Traffic Intelligence Dashboard: the page router.

Run with:  streamlit run app.py        (or .\\run_dashboard.ps1)

This file only sets up the page, the theme and the navigation. Each of the five sections lives in src/ui/ and runs as a
page from src/ui/pages/. Everything the former single-page app offered is still here, under these sections:
    Traffic Forecasts                   historical replay / what-if forecasts, provider interface (formerly the Live Prediction Demo tab)
    Model Performance & Data Sources    overview, model results, data & wrangling, historical analysis, limitations (formerly five tabs)
    Live Traffic Map, Community Reports, Road & Incident Details    new sections (the details page shows the historical road profile only; no incidents)
"""
import streamlit as st

from src.ui import diagnostics, health, navigation, theme

st.set_page_config(page_title="Traffic Intelligence Dashboard", page_icon=":material/traffic:", layout="wide",
                   initial_sidebar_state="expanded")
theme.inject_css()

page = st.navigation(navigation.build_pages(st), position="sidebar")


def chips_html(items):
    return ('<div class="tc-chips" style="flex-direction:column;align-items:flex-start;gap:.35rem">'
            + "".join(theme.chip_html(i.chip, health.KIND[i.state]) for i in items) + "</div>")


# The sidebar is laid out BEFORE the page runs, exactly as it always was: the heading, the Diagnostics panel and its buttons (a button click is
# then handled at the start of the run, and the open panel survives a rerun). Only the status TEXT (the chips and the provider/data/model/report
# lines in Diagnostics) is filled in AFTER the page has run, into containers reserved here. A page can fetch live traffic while it runs, and the
# status must show the result of THAT fetch; drawn first, it lagged one interaction behind (D4). A container keeps what it showed until the new
# text replaces it, so while a page is still running the previous status stays on screen (Streamlit greys it as out of date) under a line that says
# so ("Updating..."); the first run of a session shows "Checking..." instead. The final draw, which also clears that line, is in a finally so it
# happens when a page stops early (st.stop) or fails.
with st.sidebar:
    st.markdown('<div class="tc-section"><h2>System status</h2></div>', unsafe_allow_html=True)
    chips_slot = st.container()
    with chips_slot:
        checking = st.empty()
        # Shown from the very start of the run, so it is also visible while the Diagnostics buttons below or the page itself are busy; a run that
        # finishes at once never lets it be seen. It says that the chips under it are the PREVIOUS status until this run has finished, so a status
        # from before the latest action is never mistaken for the current one. The first run of a session has no previous status.
        checking.caption("Updating... the status below is from the previous action." if "tc_status_drawn" in st.session_state else "Checking...")
    st.write("")
    with st.expander("Diagnostics (files, models, data)"):
        status_slot = st.container()
        diagnostics.render_tools()
try:
    page.run()
finally:
    items = health.collect()  # real checks, no network request; see src/ui/health.py
    checking.empty()
    with chips_slot:
        st.markdown(chips_html(items), unsafe_allow_html=True)
    with status_slot:
        diagnostics.render_status(items)
    st.session_state["tc_status_drawn"] = True

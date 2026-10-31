"""
Traffic Intelligence Dashboard: the page router.

Run with:  streamlit run app.py        (or .\run_dashboard.ps1)

This file only sets up the page, the theme and the navigation. Each of the five sections lives in src/ui/ and runs as a
page from src/ui/pages/:
    Live Traffic Map                    search a place; TomTom traffic when a key is configured; fresh community photos near the place
    Traffic Forecasts                   historical replay / what-if forecasts, provider interface (formerly the Live Prediction Demo tab)
    Community Reports                   report traffic with a photo taken at your location, and short text reports
    Road History                        the 2022-2024 profile of each road in the dataset (no live data, no incidents)
    Model Performance & Data Sources    overview, model results, data & wrangling, historical analysis, limitations

There is no status panel in the sidebar. The checks behind it are still available: src/ui/health.py and src/ui/diagnostics.py (used by the tests),
`.\run_dashboard.ps1 -CheckOnly` (the preflight) and `python -m src.live.check_key` (the TomTom setup), none of which needs the dashboard to be open.
"""
import streamlit as st

from src.ui import navigation, theme

st.set_page_config(page_title="Traffic Intelligence Dashboard", page_icon=":material/traffic:", layout="wide",
                   initial_sidebar_state="expanded")
theme.inject_css()

page = st.navigation(navigation.build_pages(st), position="sidebar")
page.run()

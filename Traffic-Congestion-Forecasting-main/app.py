"""
Traffic Intelligence Dashboard: the page router.

Run with:  streamlit run app.py        (or .\\run_dashboard.ps1)

This file only sets up the page, the theme and the navigation. Each of the five sections lives in src/ui/ and runs as a
page from src/ui/pages/. Everything the former single-page app offered is still here, under these sections:
    Traffic Forecasts                   historical replay / what-if forecasts, provider interface (formerly the Live Prediction Demo tab)
    Model Performance & Data Sources    overview, model results, data & wrangling, historical analysis, limitations (formerly five tabs)
    Live Traffic Map, Community Reports, Road & Incident Details    new sections (Stage 0: shells and empty states only)
"""
import os

import streamlit as st

from src.ui import common, navigation, theme

st.set_page_config(page_title="Traffic Intelligence Dashboard", page_icon=":material/traffic:", layout="wide",
                   initial_sidebar_state="expanded")
theme.inject_css()

page = st.navigation(navigation.build_pages(st), position="sidebar")

with st.sidebar:
    live = ("Provider key found, not yet verified", "warn") if common.tomtom_key() else ("Live traffic: not configured", "bad")
    st.markdown('<div class="tc-section"><h2>System status</h2></div>'
                '<div class="tc-chips" style="flex-direction:column;align-items:flex-start;gap:.35rem">'
                + theme.chip_html(*live)
                + theme.chip_html("Data: historical, 2022-2024", "warn")
                + theme.chip_html("Models: not validated for 2026", "warn") + "</div>", unsafe_allow_html=True)
    st.write("")
    with st.expander("Files / Status"):
        st.write("XGBoost predictions:", os.path.exists(common.XGB_PRED_PATH))
        st.write("LSTM predictions:", os.path.exists(common.LSTM_PRED_PATH))
        st.write("Baseline predictions:", os.path.exists(common.BASELINE_PRED_PATH))
        st.write("Training History CSV:", os.path.exists(common.XGB_TRAIN_HISTORY_CSV))
        st.write("Dataset:", os.path.exists(common.fd.DATA_PATH))
        st.write("Spark results (optional):", os.path.exists(os.path.join(common.SPARK_DIR, "summary.json")))
        st.write("TomTom key (optional, unverified):", bool(common.tomtom_key()))

page.run()

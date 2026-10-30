"""
The "Diagnostics" panel in the sidebar (it replaces the old "Files / Status" list, whose entries are all still here).

It shows the same real items as the status chips, with what was checked, and the files the dashboard needs. Two slower checks
run only when asked: loading the trained models, and the full preflight. Nothing here makes a network request, and the text
carries no secrets, no absolute paths and no stack traces.
"""
import os

import streamlit as st

import time

from src.live import credentials, tomtom
from src.ui import common, health

ICON = {"ok": "\u2705", "warn": "\u26a0\ufe0f", "bad": "\u274c", "unknown": "\u2754"}
FILES = [("Dataset", lambda: common.fd.DATA_PATH, True), ("XGBoost predictions", lambda: common.XGB_PRED_PATH, True),
         ("LSTM predictions", lambda: common.LSTM_PRED_PATH, True), ("Baseline predictions", lambda: common.BASELINE_PRED_PATH, True),
         ("Saved metrics", lambda: common.METRICS_JSON, True), ("XGBoost training history", lambda: common.XGB_TRAIN_HISTORY_CSV, True),
         ("Spark results", lambda: os.path.join(common.SPARK_DIR, "summary.json"), False)]


def file_rows():
    """[(name, present, required)] from the real file system."""
    return [(n, os.path.exists(f()), req) for n, f, req in FILES]


def render_status(items=None):
    """The status lines (provider, data, models, reports). The router draws these AFTER the page has run, so they show the latest fetch."""
    items = items or health.collect()
    for i in items:
        st.markdown(f"{ICON[i.state]} **{i.label}**: {i.summary}")
        if i.detail:
            st.caption(i.detail)


def render(items=None):
    render_status(items)
    render_tools()


def render_tools():
    """Key configuration, process start, settings, files and the two buttons. The router draws these BEFORE the page runs (as buttons must be:
    a click is then handled at the start of the run and survives a rerun), while the status lines above are filled in afterwards."""
    st.markdown("**TomTom key configuration** (only whether a key is configured is shown, never its value)")
    for line in credentials.load().report_lines():
        st.caption(line)
    st.caption(f"This dashboard process {'started' if credentials.STARTED_AT_EXACT else 'loaded this code'} at "
               f"{time.strftime('%H:%M:%S', time.localtime(credentials.STARTED_AT))} and serves port "
               f"{st.get_option('server.port')}. A variable set after that time, or in another terminal window, is not visible to it: stop it and start it again.")
    for line in credentials.settings_lines({"LIVE_CACHE_TTL_S": tomtom.SERVICE.ttl, "LIVE_MAX_REQUESTS": tomtom.SERVICE.budget}):
        st.caption(line)
    st.markdown("**Files the dashboard reads**")
    for name, present, required in file_rows():
        st.markdown(f"{ICON['ok'] if present else (ICON['bad'] if required else ICON['unknown'])} {name}: "
                    + ("present" if present else ("MISSING" if required else "not present (optional)")))
    c1, c2 = st.columns(2)
    if c1.button("Check models load", key="diag_models", use_container_width=True):
        _, lstm_ready, problems = common.load_models()
        st.session_state["diag_models_result"] = problems or ["Both trained models loaded." if lstm_ready else "No problem reported."]
        st.session_state["diag_models_ok"] = not problems
    if c2.button("Run preflight", key="diag_preflight", use_container_width=True):
        from src import preflight
        bad = [c for c in preflight.run_checks(fast=True) if c.status in ("FAIL", "WARN")]
        st.session_state["diag_preflight_result"] = [f"{c.status}: {c.name}" + (f" ({health._safe(c.detail, common.PROJECT_ROOT)})" if c.detail else "") for c in bad] or ["All required checks passed."]
        st.session_state["diag_preflight_ok"] = not any(c.status == "FAIL" for c in bad)
    for key, title in (("diag_models", "Models"), ("diag_preflight", "Preflight")):
        res = st.session_state.get(key + "_result")
        if res:
            (st.success if st.session_state.get(key + "_ok") else st.error)(f"**{title}:** " + " | ".join(health._safe(r, common.PROJECT_ROOT) for r in res))

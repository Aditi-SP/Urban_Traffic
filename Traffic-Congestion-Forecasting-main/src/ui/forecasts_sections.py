"""Section: Traffic Forecasts (the former Live Prediction Demo tab: historical replay / what-if, and the provider interface)."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.dataProcessing import forecast_data as fd
from src.app import artifacts, demo, realtime
from src.ui import theme
from src.ui.common import *  # noqa: F401,F403  (shared loaders/helpers, see common.py)
from src.ui.common import split_table_cached


def reset_demo():
    for k in [k for k in st.session_state if str(k).startswith("demo_")]:
        del st.session_state[k]


def demo_chart(ctx, res, hist, source_label):
    road = res["road"]
    h = hist.tail(14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h[fd.DATE_COL], y=h[fd.TARGET_COL], mode="lines+markers",
                             name="Historical actual (dataset)", line=dict(color="#C9D6EA")))
    fig.add_trace(go.Scatter(x=[res["target_date"]], y=[res["XGBoost"]], mode="markers", name="XGBoost forecast",
                             marker=dict(symbol="diamond", size=13, color="orange")))
    fig.add_trace(go.Scatter(x=[res["target_date"]], y=[res["LSTM"]], mode="markers", name="LSTM forecast",
                             marker=dict(symbol="diamond", size=13, color="royalblue")))
    fig.add_trace(go.Scatter(x=[res["target_date"]], y=[res["Persistence baseline"]], mode="markers",
                             name="Persistence baseline", marker=dict(symbol="x", size=11, color="gray")))
    fig.add_trace(go.Scatter(x=[res["target_date"]], y=[res["Road-mean baseline"]], mode="markers",
                             name="Road-mean baseline", marker=dict(symbol="cross", size=11, color="green")))
    actual = ctx.actual_next_day(road, res["last_observation_date"])
    if actual is not None:
        fig.add_trace(go.Scatter(x=[res["target_date"]], y=[actual], mode="markers", name="Actual next day (held in dataset)",
                                 marker=dict(symbol="circle-open", size=15, color="black", line=dict(width=2))))
    fig.add_vline(x=res["last_observation_date"], line_dash="dot", line_color="gray")
    fig.update_layout(title=f"{road}: recent observations and next-day forecast  [{source_label}]",
                      yaxis_title="Congestion Level (0-100)", yaxis_range=[0, 105], template=plotly_template(),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def show_table(df, **kw):
    # exact height = no empty trailing row, no inner scrollbar
    st.dataframe(df, hide_index=True, height=35 * (len(df) + 1) + 3, use_container_width=True, **kw)


observation_table = demo.observation_table


def render_demo_replay(ctx, xgb_model):
    st.markdown("**Input source: historical data from the dataset.** This is *not* live traffic.")
    c1, c2 = st.columns(2)
    with c1:
        road = st.selectbox("Road / intersection", ctx.roads, key="demo_road")
        include_train = st.checkbox("Include dates from the training period (not a true held-out forecast)",
                                    key="demo_include_train")
    dates = ctx.replay_dates(road, test_only=not include_train)
    if not dates:
        st.error("No valid history windows for this road; choose another road.")
        return
    with c2:
        t = st.selectbox("Last observed day (t): forecast is for t + 1", dates, key=f"demo_date_{road}_{include_train}",
                         format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%A)"))
        source = st.radio("Input", ["Historical replay (unchanged real data)",
                                    "What-if scenario (edit day-t conditions)"], key="demo_source")

    hist = ctx.history(road, t)
    base = hist.iloc[-1]
    overrides, preset = {}, "Keep real values"

    if source.startswith("What-if"):
        with st.expander("Edit the day-t conditions", expanded=True):
            st.caption("The edited values replace the real day-t observation; the earlier 6+ observations stay real. "
                       "Presets use this road's 10th/90th-percentile values from the TRAINING period. "
                       "Note: the models learned little day-to-day signal, so their response to edits is weak and "
                       "not necessarily monotonic (e.g. 'heavy' may not give a higher forecast than 'light').")
            preset = st.selectbox("Scenario", ["Keep real values", *demo.PRESETS, "Rainy weather (weather only)"],
                                  key="demo_preset")
            vals = {c: float(base[c]) for c in demo.EDITABLE_NUMERIC}
            weather = base["Weather Conditions"]
            if preset in demo.PRESETS:
                vals = ctx.preset_values(road, preset)
            elif preset.startswith("Rainy"):
                weather = "Rain"
            kp = f"demo_{road}_{t}_{preset}"
            grid = st.columns(3)
            for i, col in enumerate(demo.EDITABLE_NUMERIC):
                lo, hi = float(ctx.ranges.loc["min", col]), float(ctx.ranges.loc["max", col])
                v = float(np.clip(vals[col], lo, hi))
                with grid[i % 3]:
                    if col == "Incident Reports":
                        overrides[col] = st.number_input(col, int(lo), int(hi), int(round(v)), 1, key=f"{kp}_{col}")
                    else:
                        overrides[col] = st.slider(col, lo, hi, v, key=f"{kp}_{col}")
            with grid[len(demo.EDITABLE_NUMERIC) % 3]:
                overrides["Weather Conditions"] = st.selectbox(
                    "Weather Conditions", fd.WEATHER_CATEGORIES, index=fd.WEATHER_CATEGORIES.index(weather),
                    key=f"{kp}_weather")
    else:
        with st.expander("Day-t observation used as input", expanded=False):
            show_table(observation_table(base))

    b1, b2, _ = st.columns([1, 1, 4])
    predict_clicked = b1.button("Predict", type="primary", key="demo_predict")
    b2.button("Reset demo", key="demo_reset", on_click=reset_demo)  # callback runs before the rerun

    if predict_clicked:
        used = demo.apply_scenario(hist, overrides) if overrides else hist
        label = "What-if scenario" if overrides else "Historical replay"
        try:
            res = demo.predict_all(used, ctx)
        except (demo.DemoInputError, artifacts.ArtifactError, fd.DatasetError) as e:
            st.error(f"Cannot predict: {e}")
        else:
            res.update(source=label, scenario=preset, used=used, replay_t=t)
            st.session_state["demo_result"] = res
            st.session_state.setdefault("demo_log", []).append({
                "Run": len(st.session_state.get("demo_log", [])) + 1, "Road": road,
                "Last observed day": pd.Timestamp(t).date(), "Input": f"{label}: {preset}",
                "XGBoost": round(res["XGBoost"], 2), "LSTM": round(res["LSTM"], 2),
                "Persistence": round(res["Persistence baseline"], 2), "Road mean": round(res["Road-mean baseline"], 2)})

    res = st.session_state.get("demo_result")
    if res is None:
        st.info("Choose a road and day, then press **Predict**.")
        return

    st.divider()
    st.subheader(f"Forecast for {res['road']} on {res['target_date'].date()}")
    st.caption(f"Input source: **{res['source']}** ({res['scenario']}) | forecast made from data up to "
               f"{res['last_observation_date'].date()} | horizon: {res['horizon']} | target: Congestion Level (0-100)")
    split = ctx.split_of(res["target_date"])
    if split != "test":
        st.warning(f"The forecast day falls in the {split} period, which the models saw during training or tuning: "
                   "this is not a held-out forecast.")
    for w in res["warnings"]:
        st.warning(w)
    actual = ctx.actual_next_day(res["road"], res["last_observation_date"])
    cols = st.columns(5)
    cols[0].metric("XGBoost", f"{res['XGBoost']:.1f}")
    cols[1].metric("LSTM", f"{res['LSTM']:.1f}")
    cols[2].metric("Persistence baseline", f"{res['Persistence baseline']:.1f}")
    cols[3].metric("Road-mean baseline", f"{res['Road-mean baseline']:.1f}")
    cols[4].metric("Actual next day", "n/a" if actual is None else f"{actual:.1f}")

    # the chart comes right after the numbers so it is visible without scrolling on a laptop
    fig = demo_chart(ctx, res, res["used"], res["source"])
    fig.update_layout(height=350, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if res["source"] == "What-if scenario" and actual is not None:
        st.caption("The 'actual' value belongs to the unedited history, so it is not a valid target for a what-if input.")
    if ctx.test_metrics:
        tm = ctx.test_metrics
        st.caption("For scale: on the held-out test set the typical error (RMSE) was "
                   f"XGBoost {tm['XGBoost']['RMSE']:.1f}, LSTM {tm['LSTM']['RMSE']:.1f}, "
                   f"road-mean baseline {tm['Road-mean baseline']['RMSE']:.1f} congestion points. "
                   "These are empirical errors, not confidence intervals; no uncertainty interval is produced.")
    st.caption("No congestion category (e.g. 'high') is shown: the dataset defines no validated thresholds.")

    with st.expander("Input factors and model details"):
        f1, f2 = st.columns(2)
        with f1:
            st.write("**Input factors (day-t observation fed to the models)**")
            show_table(observation_table(res["used"].iloc[-1]))
        with f2:
            st.write("**XGBoost: globally most important features** (training-time gain, not specific to this prediction)")
            if xgb_model is not None:
                fi = get_xgb_importances(xgb_model).head(6).sort_values("gain")
                st.plotly_chart(px.bar(fi, x="gain", y="feature", orientation="h", template=plotly_template()),
                                use_container_width=True)
        st.caption("Model versions (sha256 prefix): " + ", ".join(f"{k} {v}" for k, v in res["model_version"].items()))

    log = st.session_state.get("demo_log")
    if log:
        st.write("**Prediction log (this session)**")
        st.dataframe(pd.DataFrame(log), hide_index=True, height=min(35 * (len(log) + 1) + 3, 250), use_container_width=True)


def render_demo_realtime():
    st.markdown("**Input source: real-time traffic provider (TomTom Traffic Flow API).**")
    st.warning(
        "Provider data cannot be used to make a valid prediction with these models. The API returns only a current "
        "speed / travel-time snapshot for one point; the models need traffic volume, capacity utilisation, incident "
        "counts, weather and 7+ observations of history, which the provider does not supply. Rather than filling the "
        "gaps with made-up values, this mode shows the provider observation and the compatibility gap.")
    st.write("**Feature coverage: model inputs vs provider fields**")
    st.dataframe(pd.DataFrame(realtime.FEATURE_COVERAGE, columns=["Model input", "Provider field", "Status"]),
                 height=360)

    key = tomtom_key()
    if not key:
        st.info(f"No API key configured. Set the `{realtime.ENV_KEY}` environment variable "
                f"(PowerShell: `$env:{realtime.ENV_KEY}='your-key'; streamlit run app.py`) or add "
                f"`{realtime.ENV_KEY} = \"your-key\"` to `.streamlit/secrets.toml`. "
                "Status of this integration: **not verified against the live service**.")
        return
    st.success("API key found (not displayed).")
    st.caption("The dataset has no road coordinates, so enter a location yourself; nothing is looked up or invented.")
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 12.9716, format="%.5f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 77.5946, format="%.5f")
    if st.button("Fetch provider observation"):
        try:
            obs = realtime.fetch_flow_segment(lat, lon, key)
        except Exception as e:  # network/HTTP/format problems are shown verbatim, never replaced by fake data
            st.error(f"Provider request failed: {type(e).__name__}: {e}")
            return
        st.json(obs)
        st.error("Prediction blocked. Missing model inputs: " + "; ".join(realtime.prediction_blockers(obs)))


def render_demo(xgb_model, lstm_ready):
    st.subheader("Historical replay and what-if forecasts")
    mode = st.radio("Mode", ["Mode 1: Interactive Demo (historical replay / what-if)",
                             "Mode 2: Real-Time Traffic API"], horizontal=True, key="demo_mode")
    if not mode.startswith("Mode 1"):
        render_demo_realtime()  # needs neither the models nor the dataset
        return
    if artifacts.missing() or xgb_model is None or not lstm_ready:
        st.warning("**Prediction is disabled** because the trained models could not be loaded (see the setup message at the "
                   "top of the page). The other tabs still work.")
        return
    if get_data() is None:
        st.warning("**Prediction is disabled** because the dataset could not be loaded (see the setup message at the top of the page).")
        return
    try:
        ctx = load_demo_context(file_stamp(LSTM_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH, XGB_CFG_PATH, LSTM_CFG_PATH, fd.DATA_PATH))
    except (fd.DatasetError, artifacts.ArtifactError) as e:
        st.error(f"**Prediction is disabled:** {e}")
        return
    render_demo_replay(ctx, xgb_model)



def render():
    theme.page_header(
        "Traffic Forecasts",
        "Next-day congestion from the historical models and baselines",
        chips=[("Historical data 2022-2024", "warn"), ("Not validated for 2026", "bad"), ("Baselines shown beside the models", "info")])
    theme.callout("<b>How to read this page.</b> Everything here replays the 2022\u20132024 dataset: these are historical model results, "
                  "not live traffic and not validated 2026 forecasts. The models were evaluated on 2024-03-21 to 2024-08-09, and live "
                  "provider data cannot be fed to them because the inputs they need are not available from a provider.")
    xgb_model, lstm_ready = ensure_ready()
    render_demo(xgb_model, lstm_ready)

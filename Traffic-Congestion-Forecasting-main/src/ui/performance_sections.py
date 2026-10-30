"""Section: Model Performance & Data Sources (the former Overview / Data / EDA / Model Results / Limitations tabs)."""
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
from src.ui.common import comparison_notes, comparison_table
from src.ui.common import *  # noqa: F401,F403  (shared loaders/helpers, see common.py)
from src.ui.common import split_table_cached


def render_provenance(info):
    """Where the headline numbers come from, and what they can and cannot show. Everything is read from the saved evaluation and the dataset."""
    df = get_data()
    n, period = info.get("n_test_samples"), info.get("test_period")
    rows = [f"**Dataset:** Kaggle `preethamgouda/banglore-city-traffic-dataset`"
            + (f", {len(df):,} rows, {df[fd.DATE_COL].min():%Y-%m-%d} to {df[fd.DATE_COL].max():%Y-%m-%d}, daily, {df[fd.ROAD_COL].nunique()} roads" if df is not None else "")
            + ". Its provenance and licence are unverified.",
            "**Evaluation:** one chronological split by target date. The figures above are from the **held-out test set**"
            + (f" ({period[0]} to {period[1]}, {n} samples, the same rows for every model)." if n and period else "."),
            "**Compared with:** the road-mean baseline (each road's training-period average) and persistence (tomorrow = today).",
            "**Training versus held-out:** errors measured during training or validation (for example the XGBoost training curve) only decide when "
            "training stops. They are not performance; only held-out test figures count.",
            "**What this shows:** how the models did on 2022-2024 data. It does **not** show accuracy on live traffic or on 2025-2026."]
    st.markdown("**How these numbers were produced**\n\n" + "\n".join(f"- {r}" for r in rows))
    m = info.get("metrics", {})
    with st.expander("What do MAE, RMSE and R² mean?"):
        mae = m.get("Road-mean baseline", {}).get("MAE")
        st.markdown(
            "- **MAE** (mean absolute error): the average size of the miss, in congestion points on the 0-100 scale. "
            + (f"For the road-mean baseline it is {mae:.1f}: on average its prediction is about {mae:.0f} points away from what happened. " if mae else "")
            + "Lower is better.\n"
            "- **RMSE** (root mean squared error): like MAE but large misses count more, so it is the larger of the two. Lower is better.\n"
            "- **R²**: the share of the variation in congestion that the predictions explain. 1 is perfect, 0 is no better than always guessing the overall "
            "average, and below 0 is worse than that.\n"
            "- A model is only useful if it beats the simple baselines on the **same** held-out rows. Here neither learned model does.")


def render_overview():
    st.subheader("Project objective")
    st.write(
        "Forecast the **next-day Congestion Level (0-100) of each of 16 Bangalore roads/intersections** and compare "
        "two machine-learning models, **XGBoost** and an **LSTM**, against simple baselines under a fair, "
        "chronological evaluation."
    )
    st.markdown(
        "- **Data:** public Kaggle *Banglore City Traffic Dataset* (daily records, 2022-2024).\n"
        "- **Models:** XGBoost (gradient-boosted trees) and an LSTM, trained on identical samples.\n"
        "- **Evaluation:** chronological train / validation / test split; every model is scored on the same test rows.\n"
        "- **Baselines:** *persistence* (tomorrow = today) and *road mean* (each road's training-period average).\n"
        "- **Demo:** the **Traffic Forecasts** page runs the trained models on real historical windows or what-if edits (2022-2024 data, not live traffic)."
    )
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        m = read_json(metrics_path)["metrics"]
        st.subheader("Headline result (held-out test set)")
        lowest = min(m, key=lambda k: m[k]["RMSE"])
        theme.stat_cards([{"label": name, "value": f"RMSE {v['RMSE']:.2f}", "note": f"R² {v['R2']:.3f}  |  MAE {v['MAE']:.2f}",
                           "kind": "ok" if name == lowest else "info"} for name, v in m.items()])
        st.caption("Green marks the lowest test RMSE. The differences between the road mean, LSTM and XGBoost are within statistical noise (see Model Results).")
        st.table(comparison_table(m))
        info = read_json(metrics_path)
        st.caption(comparison_notes(info))
        render_provenance(info)
        best_model = min(("LSTM", "XGBoost"), key=lambda k: m[k]["RMSE"])
        base = m["Road-mean baseline"]["RMSE"]
        if m[best_model]["RMSE"] >= base:
            st.warning("Experimental models. Honest finding: neither learned model beats the simple per-road historical-mean baseline "
                       "on the held-out test set. They learn which roads are busy, but this evaluation does not show that they predict "
                       "next-day changes. Their performance on recent real-world conditions has not been established.")
    else:
        st.info("No metrics yet: run `python -m src.modelTraining.comparePerformancePlots`.")


def render_data():
    df = get_data()
    if df is None:
        dataset_unavailable()
        return
    st.subheader("Dataset")
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Roads", df[fd.ROAD_COL].nunique())
    c[2].metric("From", str(df[fd.DATE_COL].min().date()))
    c[3].metric("To", str(df[fd.DATE_COL].max().date()))
    st.caption("Source: Kaggle `preethamgouda/banglore-city-traffic-dataset`. Third-party descriptions say it derives "
               "from Government of Karnataka open data; that and the licence are **not verified**. "
               "Granularity is **daily** (every timestamp is midnight).")

    st.subheader("Wrangling and validation steps")
    st.markdown(
        "1. Parse dates; coerce numeric columns with errors raised (not silently nulled).\n"
        "2. Reject missing values, duplicate (date, road) rows and unknown weather categories "
        "(none were found, so **nothing was imputed or dropped**).\n"
        "3. Sort by road then date; build per-road windows of the last 7 observations (windows never cross roads).\n"
        "4. Keep a sample only if the road was also observed on the next day (the target exists).\n"
        "5. Split chronologically by target date; fit the LSTM scaler on training windows only."
    )

    per_road = df.groupby(fd.ROAD_COL)[fd.DATE_COL].agg(["count", "min", "max"]).reset_index()
    span = (df[fd.DATE_COL].max() - df[fd.DATE_COL].min()).days + 1
    per_road["coverage_%"] = (per_road["count"] / span * 100).round(1)
    gaps = df.sort_values([fd.ROAD_COL, fd.DATE_COL]).groupby(fd.ROAD_COL)[fd.DATE_COL].diff().dt.days.dropna()
    g1, g2 = st.columns(2)
    g1.write("**Observations per road** (days with data / total days)")
    g1.dataframe(per_road.rename(columns={"count": "observations"}), height=320)
    fig = px.histogram(gaps[gaps <= 15], nbins=15, title="Days between consecutive observations of a road",
                       labels={"value": "days"}, template=plotly_template())
    fig.update_layout(showlegend=False)
    g2.plotly_chart(fig, use_container_width=True)
    st.caption("Roads are observed on only about 30-90% of days, so many windows cannot be built; the models use "
               "'the last 7 observations' plus the gap in days between them.")

    split = split_table_cached(file_stamp(fd.DATA_PATH))
    st.subheader("Chronological split (by target date)")
    st.dataframe(split.rename(columns={"count": "samples", "min": "first target date", "max": "last target date"}))


def render_eda():
    df = get_data()
    if df is None:
        dataset_unavailable()
        return
    st.info("The dataset is daily, so there is **no hour-of-day information**: a true rush-hour (time-of-day) analysis "
            "is not possible. The time analysis below is by weekday and month.")
    d = df.copy()
    d["weekday"] = d[fd.DATE_COL].dt.day_name()
    d["month"] = d[fd.DATE_COL].dt.to_period("M").dt.to_timestamp()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    c1, c2 = st.columns(2)
    by_road = d.groupby([fd.ROAD_COL, "Area Name"])[fd.TARGET_COL].mean().reset_index().sort_values(fd.TARGET_COL)
    c1.plotly_chart(px.bar(by_road, x=fd.TARGET_COL, y=fd.ROAD_COL, color="Area Name", orientation="h",
                           title="Mean congestion by road (the main source of predictability)",
                           template=plotly_template()), use_container_width=True)
    by_wd = d.groupby("weekday")[fd.TARGET_COL].mean().reindex(order).reset_index()
    c2.plotly_chart(px.bar(by_wd, x="weekday", y=fd.TARGET_COL, title="Mean congestion by weekday",
                           template=plotly_template()), use_container_width=True)
    c2.caption("Weekday differences are small (about 1 point), so weekday is a weak predictor.")

    c3, c4 = st.columns(2)
    by_month = d.groupby("month")[fd.TARGET_COL].mean().reset_index()
    c3.plotly_chart(px.line(by_month, x="month", y=fd.TARGET_COL, markers=True,
                            title="Monthly mean congestion (all roads)", template=plotly_template()),
                    use_container_width=True)
    by_w = d.groupby("Weather Conditions")[fd.TARGET_COL].agg(["mean", "count"]).reset_index()
    c4.plotly_chart(px.bar(by_w, x="Weather Conditions", y="mean", hover_data=["count"],
                           title="Mean congestion by weather", template=plotly_template()),
                    use_container_width=True)

    corr = d[fd.NUMERIC_COLS].corr()
    st.plotly_chart(px.imshow(corr.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation of same-day measurements",
                              template=plotly_template()), use_container_width=True)
    st.caption("Congestion correlates strongly with same-day volume and capacity utilisation, but those are NOT known "
               "the day before, which is why next-day forecasting is hard.")
    share100 = (d[fd.TARGET_COL] >= 100).mean() * 100
    st.write(f"**{share100:.0f}%** of all observations have Congestion Level exactly 100 (a cap), "
             "which limits how much a regression model can learn from the extremes.")

    st.subheader("Big-data component (PySpark)")
    summary_path = os.path.join(SPARK_DIR, "summary.json")
    if not os.path.exists(summary_path):
        st.info("Spark results not generated yet. Run `python -m src.bigdata.spark_analysis` (needs Java).")
        return
    summary = read_json(summary_path)
    st.caption(f"Engine: {summary['engine']}. Input rows read: {summary['rows_read']}, after cleaning: "
               f"{summary['rows_after_cleaning']} (null keys {summary['rows_null_keys']}, duplicates "
               f"{summary['rows_duplicate_date_road']}, out of range {summary['rows_out_of_range_congestion']}). "
               "It is a local single-machine run on a small file: it demonstrates the workflow, not distributed scale.")
    t1, t2 = st.tabs(["By road (Spark)", "By weather (Spark)"])
    for tab, name in ((t1, "by_road"), (t2, "by_weather")):
        p = os.path.join(SPARK_DIR, f"{name}.csv")
        with tab:
            if os.path.exists(p):
                st.dataframe(pd.read_csv(p))


def render_results(xgb_model, num_plot_samples):
    xgb_preds = pd.read_csv(XGB_PRED_PATH) if os.path.exists(XGB_PRED_PATH) else pd.DataFrame()
    lstm_preds = pd.read_csv(LSTM_PRED_PATH) if os.path.exists(LSTM_PRED_PATH) else pd.DataFrame()
    baseline_preds = pd.read_csv(BASELINE_PRED_PATH) if os.path.exists(BASELINE_PRED_PATH) else pd.DataFrame()

    with st.expander("Preview prediction CSVs"):
        c1, c2 = st.columns(2)
        for col, name, df_ in ((c1, "XGBoost", xgb_preds), (c2, "LSTM", lstm_preds)):
            if df_.empty:
                col.info(f"No {name} predictions file found.")
            else:
                col.dataframe(df_.head(), height=200)

    REQUIRED_COLS = {"Date", "Road/Intersection Name", "Actual Congestion", "Predicted Congestion"}
    missing = [n for n, p in (("xgboost_predictions.csv", XGB_PRED_PATH), ("lstm_predictions.csv", LSTM_PRED_PATH))
               if not os.path.exists(p)]
    bad_cols = [n for n, d in (("xgboost_predictions.csv", xgb_preds), ("lstm_predictions.csv", lstm_preds))
                if not d.empty and not REQUIRED_COLS.issubset(d.columns)]

    if missing:
        st.warning(
            "Missing prediction file(s): " + ", ".join(missing) + ". "
            "Place the dataset at `data/raw/Banglore_traffic_Dataset.csv`, then from the project root run: "
            "`python -m src.modelTraining.trainXgboost`, `python -m src.modelTraining.trainLstm2` and "
            "`python -m src.modelTraining.comparePerformancePlots`."
        )
        return
    if bad_cols:
        st.error("These files lack the columns " + ", ".join(sorted(REQUIRED_COLS)) + ": " + ", ".join(bad_cols))
        return

    # Both files must describe the SAME test observations (same Date + Road + actual value),
    # otherwise the comparison is meaningless, so we refuse to compare instead of trimming.
    keys = ["Date", "Road/Intersection Name"]
    aligned = (len(xgb_preds) == len(lstm_preds)
               and (xgb_preds[keys].values == lstm_preds[keys].values).all()
               and np.allclose(xgb_preds["Actual Congestion"], lstm_preds["Actual Congestion"]))
    if not aligned:
        st.error("xgboost_predictions.csv and lstm_predictions.csv are NOT aligned (different rows, dates, "
                 "roads or actual values). Regenerate both with the training scripts before comparing models.")
        return

    n_rows = len(xgb_preds)
    y_true = xgb_preds["Actual Congestion"].values
    lstm_vals = lstm_preds["Predicted Congestion"].values
    xgb_vals = xgb_preds["Predicted Congestion"].values

    lstm_metrics = compute_metrics(y_true, lstm_vals)
    xgb_metrics = compute_metrics(y_true, xgb_vals)

    # Optional simple baselines (same rows) show whether the models learned anything beyond the obvious.
    baseline_metrics = {}
    if not baseline_preds.empty:
        if (len(baseline_preds) == n_rows and (baseline_preds[keys].values == xgb_preds[keys].values).all()
                and np.allclose(baseline_preds["Actual Congestion"], y_true)):
            for col, label in (("Persistence", "Persistence baseline"), ("Road Mean", "Road-mean baseline")):
                baseline_metrics[label] = compute_metrics(y_true, baseline_preds[col].values)
        else:
            st.warning("baseline_predictions.csv is not aligned with the model predictions and was ignored.")
    else:
        st.info("No baseline_predictions.csv found. Run `python -m src.modelTraining.comparePerformancePlots`.")

    st.caption(f"Task: forecast next-day Congestion Level per road. Held-out test set: {n_rows} observations, "
               f"{xgb_preds['Date'].min()} to {xgb_preds['Date'].max()}. Both models see identical rows.")

    # KPIs
    st.subheader("Key Performance Indicators")
    a, b, c, d = st.columns(4)
    a.metric("LSTM RMSE", round(lstm_metrics["RMSE"], 3))
    b.metric("XGB RMSE", round(xgb_metrics["RMSE"], 3))
    c.metric("LSTM MAE", round(lstm_metrics["MAE"], 3))
    d.metric("XGB MAE", round(xgb_metrics["MAE"], 3))

    best = "LSTM" if lstm_metrics["RMSE"] < xgb_metrics["RMSE"] else "XGBoost"
    best_rmse = min(lstm_metrics["RMSE"], xgb_metrics["RMSE"])
    unc = read_json(METRICS_JSON).get("uncertainty") if os.path.exists(METRICS_JSON) else None
    if unc is None:
        st.info(f"Lowest test RMSE of the two models in the saved prediction files: {best}. The bootstrap confidence intervals "
                "(model/metrics.json) were not found, so it is not known whether this difference is meaningful.")
    else:
        dxl = unc["rmse_difference"]["LSTM - XGBoost"]
        if dxl["ci_excludes_zero"]:
            st.success(f"Lowest test RMSE of the two models: **{best}** (95% CI of the RMSE difference excludes 0)")
        else:
            st.info(f"No winner between LSTM and XGBoost: the RMSE difference is {dxl['observed']:+.2f} with a 95% "
                    f"bootstrap CI [{dxl['ci95'][0]:+.2f}, {dxl['ci95'][1]:+.2f}] that includes 0, so the models are "
                    "statistically indistinguishable on this test set.")
    if baseline_metrics:
        best_base = min(baseline_metrics, key=lambda k: baseline_metrics[k]["RMSE"])
        if best_rmse >= baseline_metrics[best_base]["RMSE"]:
            st.warning(f"Neither model beats the simple **{best_base}** "
                       f"(RMSE {baseline_metrics[best_base]['RMSE']:.3f}) on the test set: the models add no "
                       "measurable skill over that baseline for this data.")

    tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Interactive Plots", "XGBoost Analysis", "Raw Data"])

    # ------------------- METRICS TABLE -------------------
    with tab1:
        columns = {"XGBoost": xgb_metrics, "LSTM": lstm_metrics, **baseline_metrics}
        dfm = pd.DataFrame({name: m for name, m in columns.items()})
        st.table(dfm)
        if unc is not None:
            ci_rows = {name: f"[{lo:.2f}, {hi:.2f}]" for name, (lo, hi) in unc["rmse_ci95"].items()}
            st.write("**Test RMSE 95% confidence intervals** (" + unc["method"] + ")")
            st.table(pd.DataFrame({"RMSE 95% CI": ci_rows}))
            st.caption("Overlapping intervals, and difference CIs that include 0, mean the LSTM, XGBoost and the "
                       "road-mean baseline cannot be ranked on this test set.")

    # ------------------- INTERACTIVE PLOTS -------------------
    with tab2:
        roads = sorted(xgb_preds["Road/Intersection Name"].unique())
        road = st.selectbox("Road / intersection", roads)
        sel = (xgb_preds["Road/Intersection Name"] == road).values
        df_plot = pd.DataFrame({
            "Date": pd.to_datetime(xgb_preds.loc[sel, "Date"]).values,
            "Actual": y_true[sel],
            "LSTM": lstm_vals[sel],
            "XGBoost": xgb_vals[sel],
        }).sort_values("Date").tail(num_plot_samples)

        fig_xgb = px.line(df_plot, x="Date", y=["Actual", "XGBoost"],
                          title=f"XGBoost: Actual vs Predicted next-day congestion ({road})",
                          template=plotly_template())
        st.plotly_chart(fig_xgb, use_container_width=True)

        fig_lstm = px.line(df_plot, x="Date", y=["Actual", "LSTM"],
                           title=f"LSTM: Actual vs Predicted next-day congestion ({road})",
                           template=plotly_template())
        st.plotly_chart(fig_lstm, use_container_width=True)

    # ------------------- XGBOOST ANALYSIS -------------------
    with tab3:
        st.subheader("Feature Importance")
        st.caption("Features use only information available at the end of day t (last 7 observations of the "
                   "road) plus the weekday/month of the target day. Importance shows what the model "
                   "used, not that it forecasts well: check the baseline comparison on the Metrics tab.")

        try:
            if xgb_model is None:
                raise RuntimeError("XGBoost model could not be loaded (see the setup message at the top of the page).")
            fi = get_xgb_importances(xgb_model)
            if fi.empty:
                raise RuntimeError("The XGBoost model reports no feature importances.")
            max_k = min(50, len(fi))
            top_k = st.slider("Top features", min(5, max_k), max_k, min(15, max_k)) if max_k > 5 else max_k
            fi_top = fi.head(top_k)

            for column, title in (("gain", "Gain"), ("weight", "Weight"), ("cover", "Cover")):
                fig = px.bar(fi_top.sort_values(column), x=column, y="feature", orientation="h",
                             title=title, template=plotly_template())
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(str(e))

        st.subheader("XGBoost Training Curves")

        if os.path.exists(XGB_TRAIN_HISTORY_CSV):
            hist = pd.read_csv(XGB_TRAIN_HISTORY_CSV)
            numeric_cols = hist.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) >= 2:
                fig_hist = px.line(hist, y=numeric_cols[:2],
                                   title="Training History",
                                   template=plotly_template())
                st.plotly_chart(fig_hist, use_container_width=True)
        elif os.path.exists(XGB_TRAINING_CURVE_IMG):
            st.image(XGB_TRAINING_CURVE_IMG)
        else:
            st.info("No training history found.")

    # ------------------- RAW DATA -------------------
    with tab4:
        dfc = pd.DataFrame({
            "Date": xgb_preds["Date"],
            "Road": xgb_preds["Road/Intersection Name"],
            "Actual": y_true,
            "XGB Pred": xgb_vals,
            "LSTM Pred": lstm_vals,
        })
        st.dataframe(dfc.head(200), height=350)
        st.download_button("Download CSV", dfc.to_csv(index=False), "comparison.csv")


def render_limits():
    st.subheader("Data source")
    st.markdown(
        "- Kaggle dataset `preethamgouda/banglore-city-traffic-dataset`. Third-party descriptions say it derives from "
        "Government of Karnataka open traffic data; this was **not verified** from an authoritative source, and the "
        "**licence is unknown**: check the Kaggle page before redistributing.\n"
        "- The dataset looks synthetic/aggregated (about 43.5% of congestion values are exactly 100) and has no coordinates.")
    st.subheader("Limitations")
    st.markdown(
        "- Experimental models: beyond each road's usual level, next-day congestion showed no predictable within-road signal in the Stage 5A benchmark "
        "(within-road lag-1 correlation about 0), and on the held-out test set the models do not beat the road-mean baseline.\n"
        "- Recent real-world performance has not been established: the dataset ends on 2024-08-09 and nothing was validated against later conditions.\n"
        "- Daily, gappy data: no hour-of-day (rush-hour) modelling; samples exist only where day t and t+1 are observed.\n"
        "- One chronological split (about 4.5 months of test data), no cross-validation, no uncertainty intervals.\n"
        "- Predictions are clipped to [0, 100]. No congestion categories are shown (no validated thresholds).\n"
        "- Real-time provider data cannot feed these models (missing volume, capacity, incidents, weather, history); "
        "the real-time integration is **unverified** against the live service.\n"
        "- PySpark runs in local mode on a single machine; nothing is distributed or streaming.")
    if os.path.exists(EVAL_REPORT_MD):
        st.caption("Note on the saved report below: it quotes a pooled day-to-day (lag-1) correlation of about 0.21 as showing weak predictability. "
                   "That figure is largely driven by persistent differences between roads, not by within-road predictability. Within roads the "
                   "lag-1 correlation is approximately 0 (0.003 in the Stage 5A benchmark), so it is not evidence that next-day changes can be "
                   "predicted. The report itself is shown unchanged.")
        with st.expander("Full evaluation report (model/evaluation_report.md)"):
            st.markdown(read_text(EVAL_REPORT_MD))



def render():
    theme.page_header(
        "Model Performance & Data Sources",
        "How the historical models were evaluated on 2022-2024 data, and where the data comes from",
        chips=[("Test period 2024-03-21 to 2024-08-09", "info"), ("Road-mean baseline is best on the reported metrics", "warn"),
               ("Not validated for 2026", "bad")])
    xgb_model, _ = ensure_ready()
    with st.sidebar:  # only this page uses the plot-size control, so only this page shows it
        with st.expander("Plot Controls"):
            st.slider("Samples to plot", min_value=100, max_value=2000, value=500, step=50, key="plot_samples")
    num_plot_samples = st.session_state.get("plot_samples", 500)
    t_over, t_res, t_data, t_eda, t_lim = st.tabs(["Overview", "Model Results", "Data & Wrangling", "Historical Analysis",
                                                   "Limitations & Sources"])
    with t_over:
        render_overview()
    with t_res:
        render_results(xgb_model, num_plot_samples)
    with t_data:
        render_data()
    with t_eda:
        render_eda()
    with t_lim:
        render_limits()

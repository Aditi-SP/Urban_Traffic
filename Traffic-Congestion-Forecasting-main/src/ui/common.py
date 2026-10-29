"""
Shared state, loaders and helpers for every dashboard page.
Moved verbatim from the former single-page app.py in Stage 0 (only the chart template and path root changed).
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataProcessing import forecast_data as fd
from src.app import artifacts, demo, realtime
from src.modelTraining import predictLstm, predictXgboost
from src.ui import theme

# -----------------------
# Project paths (absolute)
# -----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SPARK_DIR = os.path.join(PROJECT_ROOT, "reports", "spark")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
LSTM_CFG_PATH = os.path.join(MODEL_DIR, "lstm_features.json")
XGB_CFG_PATH = os.path.join(MODEL_DIR, "xgboost_features.json")
XGB_PRED_PATH = os.path.join(MODEL_DIR, "xgboost_predictions.csv")
LSTM_PRED_PATH = os.path.join(MODEL_DIR, "lstm_predictions.csv")
BASELINE_PRED_PATH = os.path.join(MODEL_DIR, "baseline_predictions.csv")
XGB_TRAIN_HISTORY_CSV = os.path.join(MODEL_DIR, "xgboost_training_history.csv")
XGB_FEATURE_IMPORTANCE_IMG = os.path.join(MODEL_DIR, "xgboost_feature_importance.png")
XGB_TRAINING_CURVE_IMG = os.path.join(MODEL_DIR, "xgboost_training_curve.png")
EVAL_REPORT_MD = os.path.join(MODEL_DIR, "evaluation_report.md")
METRICS_JSON = os.path.join(MODEL_DIR, "metrics.json")


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def file_stamp(*paths):
    return tuple(os.path.getmtime(p) if os.path.exists(p) else None for p in paths)


@st.cache_data(show_spinner=False)
def _load_dataset_cached(stamp):
    try:
        df = fd.load_raw()
        fd.load_dataset()  # also checks that enough usable samples can be built
        return df, None
    except fd.DatasetError as e:
        return None, str(e)


def dataset_status():
    """(dataframe, None) or (None, clear message)."""
    return _load_dataset_cached(file_stamp(fd.DATA_PATH))


def get_data():
    return dataset_status()[0]


@st.cache_data(show_spinner=False)
def split_table_cached(stamp):
    meta, _, _, _ = fd.load_dataset()
    return meta.groupby("split")["Target Date"].agg(["count", "min", "max"]).reindex(["train", "val", "test"])


@st.cache_resource(show_spinner=False)
def _load_booster_cached(stamp):
    return predictXgboost.load_booster()


@st.cache_resource(show_spinner=False)
def _load_lstm_cached(stamp):
    return predictLstm.preload()  # the SAME cached LSTM the demo predictions use: loaded once, not twice


def _safe_load(fn, stamp, what):
    try:
        return fn(stamp), None
    except artifacts.ArtifactError as e:
        return None, str(e)
    except Exception as e:  # e.g. a TensorFlow/Keras version mismatch: shown to the user, never hidden
        return None, f"Could not load the {what}: {type(e).__name__}: {e}. Run `python -m src.preflight` to diagnose."


def load_models():
    """Returns (xgboost_booster_or_None, lstm_ready, [problem messages])."""
    problems, booster, lstm_ready = [], None, False
    gone = artifacts.missing()
    if gone:
        problems.append("Trained model files are missing: " + artifacts.describe_missing(gone) + ". " + gone[0][2])
    gone_names = {n for _, n, _ in gone}
    if not gone_names & {"xgboost_model.json", "xgboost_features.json"}:
        booster, err = _safe_load(_load_booster_cached, file_stamp(XGB_MODEL_PATH, XGB_CFG_PATH), "XGBoost model")
        if err:
            problems.append(err)
    if not gone_names & {"model.h5", "scaler.pkl", "lstm_features.json"}:
        _, err = _safe_load(_load_lstm_cached, file_stamp(LSTM_MODEL_PATH, SCALER_PATH, LSTM_CFG_PATH), "LSTM model")
        lstm_ready = err is None
        if err:
            problems.append(err)
    return booster, lstm_ready, problems


@st.cache_resource(show_spinner=False)
def load_demo_context(models_stamp):
    """Dataset + split info for the demo. `models_stamp` (file mtimes) refreshes it after retraining."""
    return demo.DemoContext()


def render_setup_banner(data_error, model_problems):
    """ONE message for everything that disables a feature (instead of the same error in every tab)."""
    items = []
    if data_error:
        items.append("**Dataset:** " + " ".join(data_error.split()))
    items += [f"**Models:** {m}" for m in model_problems]
    if items:
        st.error("**Setup incomplete: some features are disabled** (the rest of the dashboard keeps working).\n\n"
                 + "\n".join(f"- {i}" for i in items)
                 + "\n\nRun `python -m src.preflight` in the project folder for a full check.")


def tomtom_key():
    """The optional TomTom key from the environment, or from a secrets file IF one exists.
    (Touching st.secrets without a secrets file makes Streamlit show a 'No secrets found' error box.)"""
    key = realtime.get_api_key()
    if key:
        return key
    files = (os.path.join(PROJECT_ROOT, ".streamlit", "secrets.toml"), os.path.expanduser(os.path.join("~", ".streamlit", "secrets.toml")))
    if any(os.path.isfile(f) for f in files):
        try:
            return realtime.get_api_key(st.secrets)
        except Exception:  # unreadable secrets file: treat as not configured
            return None
    return None


def dataset_unavailable():
    st.info("This section needs the dataset, which could not be loaded. See the setup message at the top of the page.")


def get_xgb_importances(booster):
    importance_types = ["weight", "gain", "cover"]
    data = {}
    for itype in importance_types:
        try:
            score = booster.get_score(importance_type=itype)
        except Exception:
            score = {}
        data[itype] = score

    features = set()
    for d in data.values():
        features.update(d.keys())

    rows = []
    for f in sorted(features):
        rows.append({
            "feature": f,
            "weight": data["weight"].get(f, 0.0),
            "gain": data["gain"].get(f, 0.0),
            "cover": data["cover"].get(f, 0.0)
        })

    return pd.DataFrame(rows).sort_values(by="gain", ascending=False).reset_index(drop=True)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# -------------------------------
# Plotly template: dark navy, matches the app theme
# -------------------------------
def _register_template():
    t = go.layout.Template(pio.templates["plotly_dark"])
    t.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.TEXT), colorway=[theme.BLUE, theme.AMBER, theme.GREEN, theme.RED, "#A78BFA", "#22D3EE"],
        xaxis=dict(gridcolor=theme.BORDER, zerolinecolor=theme.BORDER),
        yaxis=dict(gridcolor=theme.BORDER, zerolinecolor=theme.BORDER),
    )
    pio.templates["traffic_dark"] = t


_register_template()


def plotly_template():
    return "traffic_dark"


def ensure_ready(need_models=True):
    """Load models/dataset state for a page and show the ONE setup banner. Returns (xgboost_booster_or_None, lstm_ready)."""
    xgb_model, lstm_ready, problems = None, False, []
    if need_models:
        with st.spinner("Loading models..."):
            xgb_model, lstm_ready, problems = load_models()
    _, data_error = dataset_status()
    render_setup_banner(data_error, problems)
    return xgb_model, lstm_ready


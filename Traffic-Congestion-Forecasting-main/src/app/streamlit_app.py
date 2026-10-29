# src/app/streamlit_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Traffic Congestion Prediction", layout="wide")
st.title("🚦 Traffic Congestion Prediction — LSTM & XGBoost")

# --- Paths (adjust if your project structure differs) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
PREPROCESS_XGB_PATH = os.path.join(BASE_DIR, "src", "dataProcessing", "preprocess_xgboost.py")

# --- Constants (from your training) ---
SEQ_LENGTH = 48  # confirmed by you

# Exact feature order expected by the LSTM (21 features)
LSTM_FEATURE_ORDER = [
    # Numeric traffic features (11 numeric features)
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    # Time features (5)
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "year",
    # Weather one-hot (5)
    "Weather_Clear",
    "Weather_Overcast",
    "Weather_Fog",
    "Weather_Rain",
    "Weather_Windy",
]

WEATHER_CATEGORIES = ["Clear", "Overcast", "Fog", "Rain", "Windy"]

# --- Load models (cached) ---
@st.cache_resource
def load_lstm():
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(LSTM_MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

@st.cache_resource
def load_xgb():
    if os.path.exists(XGB_MODEL_PATH):
        booster = xgb.Booster()
        booster.load_model(XGB_MODEL_PATH)
        return booster
    return None

lstm_model, lstm_scaler = load_lstm()
xgb_model = load_xgb()

# Sidebar controls
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload traffic CSV", type=["csv"])
model_choice = st.sidebar.selectbox("Choose model", ("XGBoost", "LSTM"))
show_plots = st.sidebar.checkbox("Show plots", value=True)

# Helper: preprocess for LSTM — reproduce preprocess2.py behavior
def preprocess_for_lstm(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Ensure Date column present and parse
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        # If no Date, create dummy increasing timestamps (hourly) — fallback
        df["Date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="H")

    # Time-based features
    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # Fill numeric missing values with median (safe)
    numeric_cols = [
        "Traffic Volume", "Average Speed", "Travel Time Index", "Congestion Level",
        "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
        "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            # create default 0 column if missing
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0, inplace=True)

    # Weather one-hot encoding (use fixed set)
    if "Weather Conditions" not in df.columns:
        df["Weather Conditions"] = "Unknown"
    df["Weather Conditions"] = df["Weather Conditions"].fillna("Unknown").astype(str)
    for cat in WEATHER_CATEGORIES:
        colname = f"Weather_{cat}"
        df[colname] = (df["Weather Conditions"].str.strip().str.lower() == cat.lower()).astype(int)

    # Ensure all expected weather one-hot columns exist (if any category missing)
    for cat in WEATHER_CATEGORIES:
        colname = f"Weather_{cat}"
        if colname not in df.columns:
            df[colname] = 0

    # Build final frame with exact LSTM feature order
    final_df = pd.DataFrame()
    for feat in LSTM_FEATURE_ORDER:
        if feat in df.columns:
            final_df[feat] = df[feat]
        else:
            # If some time features or others missing, fill with 0
            final_df[feat] = 0.0

    # Reorder to ensure exact shape
    final_df = final_df[LSTM_FEATURE_ORDER].astype(float)

    return final_df

# Helper: create sequence for LSTM and scale
def create_lstm_input(df_processed: pd.DataFrame, scaler, seq_len=SEQ_LENGTH):
    # scaler expects features in same order it was fit — our final_df uses that order
    X_vals = df_processed.values.astype(np.float32)
    # scaler.n_features_in_ may be used to pad
    n_expected = getattr(scaler, "n_features_in_", X_vals.shape[1])
    if X_vals.shape[1] < n_expected:
        # pad with zeros to match scaler expectation
        X_vals = np.pad(X_vals, ((0, 0), (0, n_expected - X_vals.shape[1])), mode="constant")
    elif X_vals.shape[1] > n_expected:
        # truncate extra columns
        X_vals = X_vals[:, :n_expected]

    X_scaled = scaler.transform(X_vals)
    if X_scaled.shape[0] < seq_len:
        raise ValueError(f"Need at least {seq_len} rows to form a sequence; got {X_scaled.shape[0]}")
    last_seq = X_scaled[-seq_len:, :].reshape(1, seq_len, X_scaled.shape[1])
    return last_seq, X_scaled

# XGBoost: try to import and use preprocess_for_xgboost if available
def preprocess_for_xgb_if_present(uploaded_path_or_df):
    try:
        # dynamic import of your preprocess_xgboost function if present
        import importlib.util, inspect
        spec = importlib.util.spec_from_file_location("preprocess_xgboost", PREPROCESS_XGB_PATH)
        if spec:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "preprocess_for_xgboost"):
                if isinstance(uploaded_path_or_df, (str,)):
                    X, y, df_proc = mod.preprocess_for_xgboost(uploaded_path_or_df)
                else:
                    # write temp file
                    temp = "/tmp/streamlit_tmp_upload.csv"
                    uploaded_path_or_df.to_csv(temp, index=False)
                    X, y, df_proc = mod.preprocess_for_xgboost(temp)
                return X, y, df_proc
    except Exception:
        pass
    return None, None, None

# Main app logic
if uploaded_file is None:
    st.info("Upload your traffic dataset CSV in the sidebar to begin.")
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded CSV file: {e}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    if model_choice == "LSTM":
        if lstm_model is None or lstm_scaler is None:
            st.error("LSTM model or scaler not found in model/ — please train and save them first.")
        else:
            st.write("Preprocessing uploaded data for LSTM (reconstructing training pipeline)...")
            proc_df = preprocess_for_lstm(df)

            st.write(f"Processed dataframe shape: {proc_df.shape}")
            # show a small preview of processed features
            st.dataframe(proc_df.head())

            # create input sequence
            try:
                last_seq, scaled_all = create_lstm_input(proc_df, lstm_scaler, seq_len=SEQ_LENGTH)
            except ValueError as e:
                st.error(str(e))
            else:
                # predict
                pred_scaled = lstm_model.predict(last_seq, verbose=0).flatten()

                # To inverse transform congestion value: we need to pad predictions into full scaler shape
                congestion_idx = LSTM_FEATURE_ORDER.index("Congestion Level")
                n_expected = getattr(lstm_scaler, "n_features_in_", scaled_all.shape[1])
                padded = np.zeros((len(pred_scaled), n_expected))
                padded[:, congestion_idx] = pred_scaled
                inv = lstm_scaler.inverse_transform(padded)[:, congestion_idx]
                predicted_value = float(inv[0])

                st.metric("Predicted congestion level (next timestep)", f"{predicted_value:.2f}")

                if show_plots:
                    # show the last SEQ_LENGTH actual congestion values and predicted next
                    actual_series = proc_df["Congestion Level"].values[-SEQ_LENGTH:]
                    x_axis = np.arange(len(actual_series) + 1)
                    y_vals = np.concatenate([actual_series, [predicted_value]])
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(x_axis[:-1], actual_series, label="Actual (last sequence)")
                    ax.scatter(x_axis[-1], predicted_value, color="red", label="Predicted next")
                    ax.set_xlabel("Timestep (recent -> latest)")
                    ax.set_ylabel("Congestion Level")
                    ax.legend()
                    st.pyplot(fig)

    else:  # XGBoost selected
        # Prefer using your preprocess_for_xgboost if available (so predictions align)
        X, y, df_proc = preprocess_for_xgb_if_present(df)
        if X is None:
            st.info("preprocess_xgboost.py not found or failed — using numeric columns directly for XGBoost.")
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                st.error("No numeric columns found for XGBoost.")
            else:
                X = numeric.values
                feature_names = numeric.columns.tolist()
        else:
            # If preprocess function returned a matrix and features, use them
            feature_names = df_proc.columns.tolist() if df_proc is not None else None

        if xgb_model is None:
            st.error("XGBoost model not found (model/xgboost_model.json). Train and save it first.")
        else:
            try:
                dmat = xgb.DMatrix(X)
                preds = xgb_model.predict(dmat)
                # Attach predictions to df preview
                out_df = df.copy()
                out_df["Predicted Congestion Level"] = preds
                st.subheader("Predictions (head)")
                st.dataframe(out_df.head())
                st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode("utf-8"),
                                   file_name="xgboost_predictions.csv", mime="text/csv")

                if show_plots:
                    # Plot first 200 actual (if available) vs predicted (if target exists)
                    if "Congestion Level" in df.columns:
                        actual = df["Congestion Level"].values
                        pred_plot = preds[: len(actual)]
                        n = min(len(actual), 500)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(actual[:n], label="Actual")
                        ax.plot(pred_plot[:n], label="Predicted", alpha=0.8)
                        ax.set_xlabel("Sample index")
                        ax.set_ylabel("Congestion Level")
                        ax.legend()
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"XGBoost prediction error: {e}")

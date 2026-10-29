"""LEGACY: no longer used by the training scripts. It leaks the target (rolling mean incl. current value, congestion/TTI ratio).
See src/dataProcessing/forecast_data.py for the current pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_for_xgboost(file_path: str):
    """
    Preprocess Bangalore Traffic dataset for XGBoost model training.
    Adds lag, rolling, and temporal cyclic features for better congestion prediction.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # --- Handle and validate Date column ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])  # Drop rows where Date conversion failed

    # --- Ensure numeric columns are properly typed ---
    num_cols = [
        "Average Speed", "Traffic Volume", "Travel Time Index",
        "Congestion Level", "Road Capacity Utilization", "Incident Reports"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Handle missing values ---
    df = df.fillna({
        "Average Speed": df["Average Speed"].median(),
        "Traffic Volume": df["Traffic Volume"].median(),
        "Congestion Level": df["Congestion Level"].median(),
        "Travel Time Index": df["Travel Time Index"].median(),
        "Road Capacity Utilization": df["Road Capacity Utilization"].median(),
        "Incident Reports": df["Incident Reports"].median(),
        "Weather Conditions": "Unknown"
    })

    # --- Encode Weather Conditions (label encoding for XGBoost) ---
    le = LabelEncoder()
    df["Weather Conditions"] = le.fit_transform(df["Weather Conditions"])

    # --- Extract useful datetime features ---
    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # --- Add cyclic encoding for month (better for seasonality) ---
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Sort chronologically before creating lag features ---
    df = df.sort_values("Date")

    # --- Create lag and rolling mean features ---
    for col in ["Traffic Volume", "Average Speed", "Congestion Level"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_rolling_mean3"] = df[col].rolling(3).mean()

    # --- Drop rows with NaN (caused by lagging) ---
    df = df.dropna().reset_index(drop=True)

    # --- Add interaction features ---
    df["speed_volume_ratio"] = df["Average Speed"] / (df["Traffic Volume"] + 1e-5)
    df["congestion_to_travel_ratio"] = df["Congestion Level"] / (df["Travel Time Index"] + 1e-5)

    # --- Define feature columns ---
    feature_cols = [
        "Average Speed", "Traffic Volume", "Travel Time Index",
        "Road Capacity Utilization", "Incident Reports",
        "Weather Conditions", "hour", "day_of_week", "is_weekend",
        "month_sin", "month_cos", "year",
        "Traffic Volume_lag1", "Traffic Volume_lag2",
        "Average Speed_lag1", "Average Speed_lag2",
        "Congestion Level_lag1", "Congestion Level_lag2",
        "Traffic Volume_rolling_mean3", "Average Speed_rolling_mean3",
        "Congestion Level_rolling_mean3",
        "speed_volume_ratio", "congestion_to_travel_ratio"
    ]

    # --- Define target column ---
    target_col = "Congestion Level"

    # --- Split into X and y ---
    X = df[feature_cols]
    y = df[target_col]

    print(f" Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, df

"""LEGACY: no longer used by the training scripts. It fits the scaler on ALL data, does not sort by date and mixes roads (kept because notebooks/test.ipynb imports it).
See src/dataProcessing/forecast_data.py for the current pipeline."""
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.app.utils import get_project_root

# Columns expected in dataset
COLUMNS = [
    "Date", "Area Name", "Road/Intersection Name", "Traffic Volume",
    "Average Speed", "Travel Time Index", "Congestion Level",
    "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count", "Weather Conditions"
]

def convert_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date column to datetime and extract useful features."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["hour"] = df["Date"].dt.hour
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
    return df


def preprocess_data():
    """Preprocess traffic dataset for modeling."""
    root_dir = get_project_root()
    file_path = os.path.join(root_dir, "data", "raw", "Banglore_traffic_Dataset.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Static data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Keep only known columns
    df = df[[col for col in COLUMNS if col in df.columns]]

    # Extract time-based features
    df = convert_datetime_features(df)

    # Handle missing values for numeric features
    numeric_cols = [
        "Average Speed", "Traffic Volume", "Congestion Level",
        "Travel Time Index", "Road Capacity Utilization",
        "Incident Reports"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ✅ One-hot encode categorical Weather Conditions
    if "Weather Conditions" in df.columns:
        df["Weather Conditions"] = df["Weather Conditions"].fillna("Unknown")
        df = pd.get_dummies(df, columns=["Weather Conditions"], prefix="Weather")

    # ✅ Features used for modeling
    features = [
        "Congestion Level", "Average Speed", "Traffic Volume",
        "Travel Time Index", "Road Capacity Utilization",
        "Incident Reports"
    ] + [col for col in df.columns if col.startswith("Weather_")]

    # Scale features to [0,1]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, scaler

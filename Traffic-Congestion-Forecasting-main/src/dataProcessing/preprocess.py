import pandas as pd
import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
from src.app.utils import normalize_data, get_project_root
import config.settings as settings
from sklearn.preprocessing import MinMaxScaler

# Define your exact column names
COLUMNS = [
    'Date', 'Area Name', 'Road/Intersection Name', 'Traffic Volume',
    'Average Speed', 'Travel Time Index', 'Congestion Level',
    'Road Capacity Utilization', 'Incident Reports', 'Environmental Impact',
    'Public Transport Usage', 'Traffic Signal Compliance', 'Parking Usage',
    'Pedestrian and Cyclist Count', 'Weather Conditions',
    'Roadwork and Construction Activity'
]


def preprocess_data():
    """Preprocess static data file"""
    root_dir = get_project_root()
    file_path = os.path.join(root_dir, settings.DATA_SOURCES["static"]["path"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Static data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Ensure only expected columns are present
    df = df[[col for col in COLUMNS if col in df.columns]]
    

    # Add time-based features
    df = convert_datetime_features(df)

    # Handle missing values
    for col in ['Average Speed', 'Traffic Volume', 'Congestion Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Select features for modeling
    features = ['Congestion Level', 'Average Speed', 'Traffic Volume']
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df,scaler


def convert_datetime_features(df):
    """Convert and extract datetime features"""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
    return df
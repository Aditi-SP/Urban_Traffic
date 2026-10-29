import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_project_root():
    """Get the root directory of the project"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    return path

def normalize_data(df, columns):
    """Normalize specified columns"""
    scaler = MinMaxScaler()
    for col in columns:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    return df, scaler

# def get_bengaluru_coordinates():
#     """Return central coordinates of Bengaluru"""
#     return (12.9716, 77.5946)
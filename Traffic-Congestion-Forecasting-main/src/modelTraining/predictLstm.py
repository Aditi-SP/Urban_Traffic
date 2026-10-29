"""
Next-day congestion prediction for ONE road with the saved LSTM.

Input: a DataFrame in the RAW dataset format (columns Date, Road/Intersection Name, Weather Conditions and the
numeric columns) containing at least SEQ_LEN + 1 observations of that road, ending at day t.
Output: predicted Congestion Level (0-100) for day t + 1.

Run an example from the project root:   python -m src.modelTraining.predictLstm
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import functools
import joblib
import numpy as np
import pandas as pd

from src.app import artifacts
from src.dataProcessing.forecast_data import observation_features, ROAD_COL, DATE_COL

MODEL_DIR = root_dir / "model"
_NEEDS = ("LSTM model", "LSTM scaler", "LSTM feature configuration")


@functools.lru_cache(maxsize=2)
def _load_artifacts(model_dir: str, model_mtime: float):
    """Load once per model-file version (the mtime key makes a retrained model reload automatically)."""
    from tensorflow.keras.models import load_model  # imported lazily: TensorFlow is slow to import
    return (load_model(Path(model_dir) / "model.h5", compile=False), joblib.load(Path(model_dir) / "scaler.pkl"))


def preload(model_dir=None):
    """Load (and cache) the LSTM and its scaler. Raises ArtifactError with a clear message if files are missing."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    artifacts.require(_NEEDS, d)
    return _load_artifacts(str(d), (d / "model.h5").stat().st_mtime)


def predict_next_congestion(road_history: pd.DataFrame, model_dir=None) -> float:
    d = Path(model_dir) if model_dir else MODEL_DIR
    artifacts.require(_NEEDS, d)
    cfg = artifacts.read_json("lstm_features.json", "LSTM feature configuration", d)
    try:
        seq_len, roads, features = cfg["seq_len"], cfg["roads"], cfg["features"]
    except KeyError as e:
        raise artifacts.ArtifactError(f"lstm_features.json lacks the key {e}; it was not written by trainLstm2.py.") from e

    if road_history[ROAD_COL].nunique() != 1:
        raise ValueError("Pass the history of exactly one road.")
    if road_history[ROAD_COL].iloc[0] not in roads:
        raise ValueError(f"Unknown road. Known roads: {roads}")

    feats = observation_features(road_history, roads)
    if len(feats) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} observations of this road, got {len(feats)}.")
    window = feats[features].tail(seq_len)  # exact training feature order

    model, scaler = preload(d)
    if list(scaler.feature_names_in_) != features:
        raise artifacts.ArtifactError("scaler.pkl and lstm_features.json disagree on the feature order: "
                                      "they do not belong to the same trained model.")

    scaled = scaler.transform(window).reshape(1, seq_len, len(features))
    pred = float(model.predict(scaled, verbose=0)[0][0]) * cfg["target_scale"]
    return float(np.clip(pred, 0, cfg["target_scale"]))


if __name__ == "__main__":
    from src.dataProcessing.forecast_data import load_raw

    df = load_raw()
    road = df[ROAD_COL].iloc[0]
    hist = df[df[ROAD_COL] == road].tail(10)
    print(f"Road: {road}, last observed day: {hist[DATE_COL].iloc[-1].date()}")
    print("Predicted congestion for the next day:", round(predict_next_congestion(hist), 2))

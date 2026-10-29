"""
Next-day congestion prediction for ONE road with the saved XGBoost model (same input as predictLstm.py).

Run an example from the project root:   python -m src.modelTraining.predictXgboost
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import functools
import numpy as np
import pandas as pd
import xgboost as xgb

from src.app import artifacts
from src.dataProcessing.forecast_data import (
    observation_features, tabular_from_sequences, seq_feature_names, ROAD_COL, DATE_COL, TARGET_SCALE,
)

MODEL_DIR = root_dir / "model"
_NEEDS = ("XGBoost model", "XGBoost feature configuration")


@functools.lru_cache(maxsize=2)
def _load_booster(model_dir: str, model_mtime: float):
    booster = xgb.Booster()
    booster.load_model(str(Path(model_dir) / "xgboost_model.json"))
    return booster


def load_booster(model_dir=None):
    """Load (and cache) the XGBoost booster. Raises ArtifactError with a clear message if files are missing."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    artifacts.require(_NEEDS, d)
    return _load_booster(str(d), (d / "xgboost_model.json").stat().st_mtime)


def predict_next_congestion(road_history: pd.DataFrame, model_dir=None) -> float:
    d = Path(model_dir) if model_dir else MODEL_DIR
    artifacts.require(_NEEDS, d)
    cfg = artifacts.read_json("xgboost_features.json", "XGBoost feature configuration", d)
    try:
        seq_len, roads, columns = cfg["seq_len"], cfg["roads"], cfg["xgboost_features"]
    except KeyError as e:
        raise artifacts.ArtifactError(f"xgboost_features.json lacks the key {e}; it was not written by trainXgboost.py.") from e

    if road_history[ROAD_COL].nunique() != 1:
        raise ValueError("Pass the history of exactly one road.")
    if road_history[ROAD_COL].iloc[0] not in roads:
        raise ValueError(f"Unknown road. Known roads: {roads}")

    feats = observation_features(road_history, roads)
    if len(feats) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} observations of this road, got {len(feats)}.")
    window = feats[seq_feature_names(roads)].tail(seq_len).values.astype(np.float32)
    target_date = feats[DATE_COL].iloc[-1] + pd.Timedelta(days=1)

    X = tabular_from_sequences(window[None, :, :], [target_date], roads)
    X = X[columns]  # exact training column order

    booster = load_booster(d)
    if list(booster.feature_names or []) != list(columns):
        raise artifacts.ArtifactError("xgboost_model.json and xgboost_features.json disagree on the feature list: "
                                      "they do not belong to the same trained model.")
    pred = float(booster.predict(xgb.DMatrix(X), iteration_range=(0, booster.best_iteration + 1))[0])
    return float(np.clip(pred, 0, TARGET_SCALE))


if __name__ == "__main__":
    from src.dataProcessing.forecast_data import load_raw

    df = load_raw()
    road = df[ROAD_COL].iloc[0]
    hist = df[df[ROAD_COL] == road].tail(10)
    print(f"Road: {road}, last observed day: {hist[DATE_COL].iloc[-1].date()}")
    print("Predicted congestion for the next day:", round(predict_next_congestion(hist), 2))

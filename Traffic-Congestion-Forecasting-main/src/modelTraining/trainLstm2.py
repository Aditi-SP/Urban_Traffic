"""
Train an LSTM to forecast NEXT-DAY congestion per road (see src/dataProcessing/forecast_data.py).
Uses exactly the same samples, split and target as trainXgboost.py.

Run from the project root:   python -m src.modelTraining.trainLstm2
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, callbacks, Sequential

from src.dataProcessing.forecast_data import (
    load_dataset, seq_feature_names, SEQ_LEN, TARGET_SCALE,
)

MODEL_DIR = root_dir / "model"
SEED = 42


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def fit_scaler_on_train(X_seq, train_mask, roads):
    """MinMax scaler fitted on TRAINING windows only; returns (scaler, feature_names)."""
    names = seq_feature_names(roads)
    flat_train = pd.DataFrame(X_seq[train_mask].reshape(-1, X_seq.shape[2]), columns=names)
    scaler = MinMaxScaler().fit(flat_train)  # feature_names_in_ records the exact order
    return scaler, names


def scale_windows(scaler, X_seq):
    n, t, f = X_seq.shape
    names = list(scaler.feature_names_in_)
    flat = pd.DataFrame(X_seq.reshape(-1, f), columns=names)
    return scaler.transform(flat).reshape(n, t, f).astype(np.float32)


def train_model():
    set_seeds()
    meta, X_seq, roads, (train_end, val_end) = load_dataset()
    tr = (meta["split"] == "train").values
    va = (meta["split"] == "val").values
    te = (meta["split"] == "test").values
    print(f"Samples: train={tr.sum()} val={va.sum()} test={te.sum()}  | window={X_seq.shape[1:]}")

    scaler, names = fit_scaler_on_train(X_seq, tr, roads)
    X = scale_windows(scaler, X_seq)
    y = (meta["y"].values / TARGET_SCALE).astype(np.float32)  # congestion is a 0-100 percentage

    model = Sequential([
        layers.Input(shape=(SEQ_LEN, X.shape[2])),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    # Early stopping on the VALIDATION split; the test split is never seen during training.
    history = model.fit(
        X[tr], y[tr], validation_data=(X[va], y[va]),
        epochs=100, batch_size=32, verbose=2,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )

    y_pred = np.clip(model.predict(X[te], verbose=0).flatten() * TARGET_SCALE, 0, TARGET_SCALE)
    y_true = meta.loc[te, "y"].values
    print(f"Test RMSE: {np.sqrt(np.mean((y_true - y_pred) ** 2)):.3f}  (epochs run: {len(history.history['loss'])})")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_DIR / "model.h5"))
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    (MODEL_DIR / "lstm_features.json").write_text(json.dumps(
        {"seq_len": SEQ_LEN, "roads": roads, "features": names, "target_scale": TARGET_SCALE,
         "target": "Congestion Level at t+1 day"}, indent=2), encoding="utf-8")

    pd.DataFrame({
        "Date": meta.loc[te, "Target Date"].dt.strftime("%Y-%m-%d").values,
        "Road/Intersection Name": meta.loc[te, "Road"].values,
        "Actual Congestion": y_true,
        "Predicted Congestion": y_pred,
    }).to_csv(MODEL_DIR / "lstm_predictions.csv", index=False)

    print(f"Saved model, scaler and predictions to {MODEL_DIR}")
    return model, history


if __name__ == "__main__":
    train_model()

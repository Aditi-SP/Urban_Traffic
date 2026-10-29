"""
Train XGBoost to forecast NEXT-DAY congestion per road (see src/dataProcessing/forecast_data.py).

Run from the project root:   python -m src.modelTraining.trainXgboost
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files; never block on a window
import matplotlib.pyplot as plt
import xgboost as xgb

from src.dataProcessing.forecast_data import (
    load_dataset, tabular_from_sequences, SEQ_LEN, TARGET_SCALE,
)

MODEL_DIR = root_dir / "model"


def train_xgboost_model():
    meta, X_seq, roads, (train_end, val_end) = load_dataset()
    X = tabular_from_sequences(X_seq, meta["Target Date"], roads)
    y = meta["y"].values

    tr, va, te = (meta["split"] == s for s in ("train", "val", "test"))
    print(f"Samples: train={tr.sum()} val={va.sum()} test={te.sum()}  | features={X.shape[1]}")
    print(f"Train targets up to {pd.Timestamp(train_end).date()}, val up to {pd.Timestamp(val_end).date()}, "
          f"test {meta.loc[te, 'Target Date'].min().date()} .. {meta.loc[te, 'Target Date'].max().date()}")

    dtrain = xgb.DMatrix(X[tr], label=y[tr])
    dval = xgb.DMatrix(X[va], label=y[va])
    dtest = xgb.DMatrix(X[te], label=y[te])

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 4,  # shallower than before: only ~4k training samples
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.5,
        "min_child_weight": 3,
        "random_state": 42,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }

    evals_result = {}
    # Early stopping uses the VALIDATION split; the test split is only used for the final evaluation.
    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=30, verbose_eval=50, evals_result=evals_result,
    )

    y_pred = np.clip(model.predict(dtest, iteration_range=(0, model.best_iteration + 1)), 0, TARGET_SCALE)
    rmse = float(np.sqrt(np.mean((y[te] - y_pred) ** 2)))
    print(f"Test RMSE: {rmse:.3f}  (best iteration {model.best_iteration})")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgboost_model.json"))

    # Predictions on the held-out test samples (same rows the LSTM is scored on)
    pd.DataFrame({
        "Date": meta.loc[te, "Target Date"].dt.strftime("%Y-%m-%d").values,
        "Road/Intersection Name": meta.loc[te, "Road"].values,
        "Actual Congestion": y[te],
        "Predicted Congestion": y_pred,
    }).to_csv(MODEL_DIR / "xgboost_predictions.csv", index=False)

    # Training history for the dashboard
    pd.DataFrame({"train_rmse": evals_result["train"]["rmse"],
                  "validation_rmse": evals_result["eval"]["rmse"]}).to_csv(
        MODEL_DIR / "xgboost_training_history.csv", index=False)

    config = {
        "seq_len": SEQ_LEN, "roads": roads, "xgboost_features": list(X.columns),
        "target": "Congestion Level at t+1 day", "split_train_end": str(pd.Timestamp(train_end).date()),
        "split_val_end": str(pd.Timestamp(val_end).date()),
    }
    (MODEL_DIR / "xgboost_features.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Plots
    plot_importance_png(model)
    plot_curves_png(evals_result)
    plot_actual_vs_pred_png(y[te], y_pred)
    print(f"Saved model, predictions, history and plots to {MODEL_DIR}")
    return model


def plot_importance_png(model):
    scores = model.get_score(importance_type="gain")
    top = sorted(scores.items(), key=lambda kv: kv[1])[-15:]
    plt.figure(figsize=(10, 7))
    plt.barh([k for k, _ in top], [v for _, v in top], color="steelblue")
    plt.xlabel("Gain")
    plt.title("XGBoost feature importance (top 15, gain)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "xgboost_feature_importance.png")
    plt.close()


def plot_curves_png(evals_result):
    train_rmse, eval_rmse = evals_result["train"]["rmse"], evals_result["eval"]["rmse"]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label="Train RMSE")
    plt.plot(range(1, len(eval_rmse) + 1), eval_rmse, label="Validation RMSE")
    plt.xlabel("Boosting rounds")
    plt.ylabel("RMSE")
    plt.title("Training vs validation RMSE (XGBoost)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "xgboost_training_curve.png")
    plt.close()


def plot_actual_vs_pred_png(actual, pred, n=300):
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:n], label="Actual", color="black")
    plt.plot(pred[:n], label="Predicted", color="orange", alpha=0.8)
    plt.title(f"XGBoost: actual vs predicted next-day congestion (first {n} test samples)")
    plt.xlabel("Test sample index")
    plt.ylabel("Congestion Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "xgboost_actual_vs_predicted.png")
    plt.close()


if __name__ == "__main__":
    train_xgboost_model()

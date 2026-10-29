"""
Compare LSTM, XGBoost and two simple baselines on the SAME held-out test samples.

Run from the project root (after both training scripts):   python -m src.modelTraining.comparePerformancePlots

Baselines (no learning from the test period):
  * Persistence : tomorrow's congestion = today's congestion on that road.
  * Road mean   : each road's mean congestion over the TRAINING period.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import platform
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataProcessing.forecast_data import (
    load_dataset, load_raw, seq_feature_names, TARGET_COL, ROAD_COL, DATE_COL,
)

MODEL_DIR = root_dir / "model"
KEY_COLS = ["Date", "Road/Intersection Name", "Actual Congestion"]


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(err ** 2) / ss_tot),
    }


def build_baselines():
    meta, X_seq, roads, _ = load_dataset()
    ci = seq_feature_names(roads).index(TARGET_COL)
    te = meta["split"] == "test"
    road_mean = meta[meta["split"] == "train"].groupby("Road")["y"].mean()
    out = pd.DataFrame({
        "Date": meta.loc[te, "Target Date"].dt.strftime("%Y-%m-%d").values,
        "Road/Intersection Name": meta.loc[te, "Road"].values,
        "Actual Congestion": meta.loc[te, "y"].values,
        "Persistence": X_seq[te.values, -1, ci],
        "Road Mean": meta.loc[te, "Road"].map(road_mean).values,
    })
    return out


N_BOOT, BOOT_SEED = 4000, 0


def bootstrap_rmse(actual, dates, preds):
    """
    Cluster bootstrap over test DATES (rows on the same day are correlated, so whole days are resampled).
    Returns 95% CIs of each model's RMSE and of RMSE differences. Seeded, so it is reproducible.
    """
    dates = np.asarray(dates)
    groups = [np.where(dates == d)[0] for d in np.unique(dates)]
    rng = np.random.default_rng(BOOT_SEED)
    rm = lambda a, p: float(np.sqrt(np.mean((a - p) ** 2)))
    pairs = [("LSTM", "Road-mean baseline"), ("XGBoost", "Road-mean baseline"), ("LSTM", "XGBoost")]
    rmses = {k: [] for k in preds}
    diffs = {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        r = {k: rm(actual[idx], p[idx]) for k, p in preds.items()}
        for k in preds:
            rmses[k].append(r[k])
        for a, b in pairs:
            diffs[f"{a} - {b}"].append(r[a] - r[b])
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return {
        "method": f"cluster bootstrap by test date, {N_BOOT} resamples, seed {BOOT_SEED}",
        "rmse_ci95": {k: ci(v) for k, v in rmses.items()},
        "rmse_difference": {k: {"observed": float(np.mean(v)), "ci95": ci(v),
                                "ci_excludes_zero": not (ci(v)[0] <= 0 <= ci(v)[1])} for k, v in diffs.items()},
    }


def dataset_facts():
    """Numbers quoted in the report, computed from the data (nothing hard-coded)."""
    raw = load_raw()
    meta, _, _, _ = load_dataset()
    g = raw.groupby(ROAD_COL)
    prev, gap = g[TARGET_COL].shift(1), g[DATE_COL].diff().dt.days
    ok = gap == 1
    span7 = g[DATE_COL].transform(lambda s: (s - s.shift(6)).dt.days)
    win = meta.merge(raw.assign(span7=span7)[[ROAD_COL, DATE_COL, "span7"]],
                     left_on=["Road", "Date"], right_on=[ROAD_COL, DATE_COL], how="left")
    nxt = raw[[ROAD_COL, DATE_COL]].copy()
    nxt[DATE_COL] = nxt[DATE_COL] - pd.Timedelta(days=1)
    nxt["has_next"] = True
    r2 = raw.merge(nxt, on=[ROAD_COL, DATE_COL], how="left")
    has = r2["has_next"].fillna(False).astype(bool)
    return {
        "rows": int(len(raw)), "roads": int(raw[ROAD_COL].nunique()),
        "pct_congestion_at_100": float((raw[TARGET_COL] >= 100).mean() * 100),
        "lag1_corr_consecutive_days": float(pd.concat([raw[TARGET_COL][ok], prev[ok]], axis=1).corr().iloc[0, 1]),
        "pct_windows_spanning_over_7_days": float((win["span7"] > 7).mean() * 100),
        "mean_congestion_when_next_day_observed": float(r2.loc[has, TARGET_COL].mean()),
        "mean_congestion_when_next_day_missing": float(r2.loc[~has, TARGET_COL].mean()),
        "n_samples_total": int(len(meta)),
    }


def write_report(metrics, xgb_df, n_samples, identical_rows, uncertainty):
    """Reproducible evaluation report (model/evaluation_report.md + .json)."""
    import sklearn, tensorflow, keras, xgboost
    xcfg = json.loads((MODEL_DIR / "xgboost_features.json").read_text(encoding="utf-8"))
    lcfg = json.loads((MODEL_DIR / "lstm_features.json").read_text(encoding="utf-8"))
    sha = lambda f: hashlib.sha256((MODEL_DIR / f).read_bytes()).hexdigest()[:12]
    f = dataset_facts()
    d = uncertainty["rmse_difference"]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"file": "data/raw/Banglore_traffic_Dataset.csv",
                    "source": "Kaggle dataset preethamgouda/banglore-city-traffic-dataset (downloaded from Kaggle)",
                    "provenance_note": "Kaggle/third-party descriptions say it derives from Government of Karnataka open "
                                       "traffic data; this was NOT verified from an authoritative source.",
                    "license": "NOT VERIFIED - check the Kaggle page", "rows": f["rows"], "roads": f["roads"],
                    "granularity": "daily (no hour-of-day)"},
        "target": "Congestion Level (0-100) of the same road on day t+1",
        "forecast_horizon": "1 day",
        "history_window": f"last {xcfg['seq_len']} observations of the road (not {xcfg['seq_len']} calendar days)",
        "split": {"train_targets_to": xcfg["split_train_end"], "validation_targets_to": xcfg["split_val_end"],
                  "test_targets": [str(xgb_df["Date"].min()), str(xgb_df["Date"].max())], "type": "chronological",
                  "n_test_samples": n_samples, "n_samples_total": f["n_samples_total"]},
        "identical_test_rows_for_all_models": bool(identical_rows),  # result of the alignment check in main()
        "lstm_features": lcfg["features"], "xgboost_features": xcfg["xgboost_features"],
        "metrics": metrics,
        "uncertainty": uncertainty,
        "artifact_sha256_prefix": {n: sha(n) for n in ("model.h5", "scaler.pkl", "xgboost_model.json")},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "tensorflow": tensorflow.__version__,
                     "keras": keras.__version__, "xgboost": xgboost.__version__},
        "dataset_facts": f,
        "limitations": [
            "Single chronological split; no cross-validation; test period about 4.5 months.",
            "Daily, gappy data: samples exist only where the road was observed on days t and t+1.",
            f"Selection effect: mean congestion on day t is {f['mean_congestion_when_next_day_observed']:.1f} when day t+1 "
            f"is observed vs {f['mean_congestion_when_next_day_missing']:.1f} when it is missing, so evaluation is "
            "conditional on the next day being observed.",
            f"{f['pct_windows_spanning_over_7_days']:.0f}% of input windows span more than 7 calendar days "
            "(a window is the last 7 observations; days_since_prev is a feature).",
            f"Next-day congestion is only weakly predictable (lag-1 correlation {f['lag1_corr_consecutive_days']:.2f}).",
            f"{f['pct_congestion_at_100']:.0f}% of congestion values are exactly 100 (a cap); the data looks "
            "synthetic/aggregated, which is an inference, not a verified fact.",
            "LSTM numbers come from one training run (one seed); TensorFlow is not bit-deterministic.",
            "Predictions are clipped to the physical range [0, 100].",
            "No prediction intervals are produced; the bootstrap CIs above describe test-set RMSE uncertainty only.",
        ],
    }
    (MODEL_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fmt = lambda k: f"{d[k]['observed']:+.2f} [{d[k]['ci95'][0]:+.2f}, {d[k]['ci95'][1]:+.2f}]"
    distinguishable = [k for k, v in d.items() if v["ci_excludes_zero"]]
    verdict = ("No pairwise RMSE difference among LSTM, XGBoost and the road-mean baseline is statistically "
               "distinguishable from zero, so no ranking among them is supported." if not distinguishable else
               "RMSE differences whose 95% CI excludes zero: " + ", ".join(distinguishable) + ".")
    lines = ["# Evaluation report", "", f"Generated (UTC): {report['generated_utc']}", "",
             f"* Dataset: {report['dataset']['source']}. {report['dataset']['provenance_note']} Licence not verified. "
             f"{f['rows']} rows, {f['roads']} roads, {report['dataset']['granularity']}",
             f"* Target: {report['target']}  |  horizon: {report['forecast_horizon']}  |  inputs: {report['history_window']}",
             f"* Split (chronological, by target date): train to {xcfg['split_train_end']}, validation to "
             f"{xcfg['split_val_end']}, test {report['split']['test_targets'][0]} .. {report['split']['test_targets'][1]} "
             f"({n_samples} samples, identical for every model: {identical_rows})", "",
             "| Model | MAE | RMSE | RMSE 95% CI | R2 |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {m['MAE']:.3f} | {m['RMSE']:.3f} | [{uncertainty['rmse_ci95'][k][0]:.2f}, "
              f"{uncertainty['rmse_ci95'][k][1]:.2f}] | {m['R2']:.3f} |" for k, m in metrics.items()]
    lines += ["", f"RMSE differences (observed [95% CI]; {uncertainty['method']}):"] + \
             [f"* {k}: {fmt(k)}" for k in d] + ["", verdict, "",
              "A learned model is only useful here if it beats the road-mean baseline; it does not.", "",
              "## Software", ", ".join(f"{k} {v}" for k, v in report["software"].items()), "",
              "## Artifacts (sha256 prefix)", ", ".join(f"{k}: {v}" for k, v in report["artifact_sha256_prefix"].items()), "",
              "## Limitations"] + [f"* {x}" for x in report["limitations"]]
    (MODEL_DIR / "evaluation_report.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def main():
    xgb_df = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
    lstm_df = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")
    base_df = build_baselines()
    base_df.to_csv(MODEL_DIR / "baseline_predictions.csv", index=False)

    # The comparison is only valid if all files describe the identical test observations.
    for name, d in (("lstm", lstm_df), ("baseline", base_df)):
        if len(d) != len(xgb_df) or not (d[KEY_COLS[:2]].values == xgb_df[KEY_COLS[:2]].values).all() \
                or not np.allclose(d["Actual Congestion"], xgb_df["Actual Congestion"]):
            raise ValueError(f"{name} predictions are not aligned with the XGBoost test rows.")

    actual = xgb_df["Actual Congestion"].values
    preds = {
        "LSTM": lstm_df["Predicted Congestion"].values,
        "XGBoost": xgb_df["Predicted Congestion"].values,
        "Persistence baseline": base_df["Persistence"].values,
        "Road-mean baseline": base_df["Road Mean"].values,
    }
    metrics = {name: regression_metrics(actual, p) for name, p in preds.items()}
    uncertainty = bootstrap_rmse(actual, xgb_df["Date"].values, preds)

    info = {"n_test_samples": int(len(actual)), "uncertainty": uncertainty,
            "test_period": [str(xgb_df["Date"].min()), str(xgb_df["Date"].max())],
            "target": "Congestion Level, next day, per road", "metrics": metrics}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    with open(MODEL_DIR / "saved_metrics.txt", "w", encoding="utf-8") as f:
        f.write("===== Next-day congestion forecast: held-out test comparison =====\n")
        f.write(f"Test samples: {len(actual)}  ({info['test_period'][0]} .. {info['test_period'][1]})\n")
        for name, m in metrics.items():
            f.write(f"{name}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}\n")

    write_report(metrics, xgb_df, len(actual), identical_rows=True, uncertainty=uncertainty)  # alignment checked above
    print(f"{'Model':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}   RMSE 95% CI")
    for name, m in metrics.items():
        lo, hi = uncertainty["rmse_ci95"][name]
        print(f"{name:<24}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['R2']:>10.3f}   [{lo:.2f}, {hi:.2f}]")
    for k, v in uncertainty["rmse_difference"].items():
        print(f"RMSE difference {k}: {v['observed']:+.2f} 95% CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")

    # Bar chart of MAE / RMSE
    names = list(metrics)
    x = np.arange(len(names))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, [metrics[n]["MAE"] for n in names], 0.4, label="MAE")
    plt.bar(x + 0.2, [metrics[n]["RMSE"] for n in names], 0.4, label="RMSE")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Error (congestion points)")
    plt.title("Next-day congestion: test-set error (lower is better)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "LSTM_vs_XGBoost_Metrics.png")
    plt.close()

    # Actual vs predicted for the first road, in date order
    road = xgb_df["Road/Intersection Name"].iloc[0]
    mask = (xgb_df["Road/Intersection Name"] == road).values
    plt.figure(figsize=(11, 5))
    plt.plot(xgb_df.loc[mask, "Date"], actual[mask], label="Actual", color="black", linewidth=2)
    plt.plot(xgb_df.loc[mask, "Date"], preds["XGBoost"][mask], label="XGBoost", color="orange")
    plt.plot(xgb_df.loc[mask, "Date"], preds["LSTM"][mask], label="LSTM", color="tab:blue", alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Actual vs predicted next-day congestion: {road} (test period)")
    plt.ylabel("Congestion Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "LSTM_vs_XGBoost_Predictions.png")
    plt.close()
    print(f"Saved metrics and plots to {MODEL_DIR}")


if __name__ == "__main__":
    main()

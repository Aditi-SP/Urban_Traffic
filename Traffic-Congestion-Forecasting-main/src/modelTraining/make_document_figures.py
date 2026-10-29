"""
Figures for the research paper and slides, drawn ONLY from the verified evaluation outputs:
model/metrics.json, model/xgboost_predictions.csv, model/lstm_predictions.csv, model/baseline_predictions.csv.
Nothing is re-trained or re-predicted here.

Run from the project root:   python -m src.modelTraining.make_document_figures
Writes reports/figures/fig1_rmse_ci.png, fig2_actual_vs_predicted_road.png, fig3_predicted_vs_actual.png,
fig4_mean_congestion_by_road.png (from the PySpark output)
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MODEL_DIR = root_dir / "model"
OUT_DIR = root_dir / "reports" / "figures"
COLORS = {"LSTM": "#1f77b4", "XGBoost": "#ff7f0e", "Persistence baseline": "#7f7f7f", "Road-mean baseline": "#2ca02c"}
KEYS = ["Date", "Road/Intersection Name"]


def load_verified():
    x = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
    l = pd.read_csv(MODEL_DIR / "lstm_predictions.csv")
    b = pd.read_csv(MODEL_DIR / "baseline_predictions.csv")
    for name, d in (("lstm", l), ("baseline", b)):  # refuse to plot misaligned files
        if len(d) != len(x) or not (d[KEYS].values == x[KEYS].values).all() \
                or not np.allclose(d["Actual Congestion"], x["Actual Congestion"]):
            raise ValueError(f"{name} predictions are not aligned with the XGBoost test rows.")
    return x, l, b, json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))


def fig_rmse_ci(metrics):
    order = ["Persistence baseline", "Road-mean baseline", "LSTM", "XGBoost"]
    rmse = [metrics["metrics"][k]["RMSE"] for k in order]
    ci = [metrics["uncertainty"]["rmse_ci95"][k] for k in order]
    err = np.array([[r - lo, hi - r] for r, (lo, hi) in zip(rmse, ci)]).T
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(order, rmse, yerr=err, capsize=6, color=[COLORS[k] for k in order], edgecolor="black", linewidth=0.6)
    for i, (r, (lo, hi)) in enumerate(zip(rmse, ci)):
        ax.text(i, hi + 0.5, f"{r:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test RMSE (congestion points)")
    ax.set_title("Next-day congestion: test RMSE with 95% bootstrap CI\n(856 common test rows; lower is better)", fontsize=11)
    ax.set_ylim(0, max(hi for _, hi in ci) + 4)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_rmse_ci.png", dpi=200)
    plt.close(fig)


def fig_actual_vs_predicted_road(x, l, b, road):
    m = (x["Road/Intersection Name"] == road).values
    d = pd.to_datetime(x.loc[m, "Date"])
    order = np.argsort(d.values)
    d = d.iloc[order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(d, x.loc[m, "Actual Congestion"].values[order], "o-", color="black", lw=1.2, ms=3, label="Actual")
    ax.plot(d, b.loc[m, "Road Mean"].values[order], "--", color=COLORS["Road-mean baseline"], lw=1.6, label="Road-mean baseline")
    ax.plot(d, x.loc[m, "Predicted Congestion"].values[order], "-", color=COLORS["XGBoost"], lw=1.4, label="XGBoost")
    ax.plot(d, l.loc[m, "Predicted Congestion"].values[order], "-", color=COLORS["LSTM"], lw=1.4, label="LSTM")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Congestion Level (0-100)")
    ax.set_title(f"Actual vs predicted next-day congestion: {road}\n(held-out test period; only observed days are plotted)", fontsize=11)
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_actual_vs_predicted_road.png", dpi=200)
    plt.close(fig)


def fig_scatter(x, l, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    y = x["Actual Congestion"].values
    for ax, name, p in ((axes[0], "LSTM", l["Predicted Congestion"].values), (axes[1], "XGBoost", x["Predicted Congestion"].values)):
        ax.scatter(y, p, s=10, alpha=0.35, color=COLORS[name])
        ax.plot([0, 100], [0, 100], "k--", lw=1, label="perfect forecast")
        ax.set_title(f"{name}  (R² = {metrics['metrics'][name]['R2']:.3f})", fontsize=10)
        ax.set_xlabel("Actual congestion")
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Predicted congestion")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Predicted vs actual on the 856 test rows: predictions stay near the average", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_predicted_vs_actual.png", dpi=200)
    plt.close(fig)


def fig_mean_congestion_by_road():
    """Mean congestion per road from the PySpark output (reports/spark/by_road.csv), for the slides."""
    path = root_dir / "reports" / "spark" / "by_road.csv"
    if not path.exists():
        print("skipped fig4: run python -m src.bigdata.spark_analysis first")
        return
    d = pd.read_csv(path).sort_values("mean_congestion")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(d["Road/Intersection Name"], d["mean_congestion"], color="#5b5f71")
    for y, v in enumerate(d["mean_congestion"]):
        ax.text(v + 0.8, y, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Congestion Level (0-100), 2022-2024")
    ax.set_title("Mean congestion by road (PySpark, local mode)\nroads differ a lot, which is the main source of predictability", fontsize=10)
    ax.set_xlim(0, 105)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_mean_congestion_by_road.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, l, b, metrics = load_verified()
    road = sorted(x["Road/Intersection Name"].unique())[0]
    fig_rmse_ci(metrics)
    fig_actual_vs_predicted_road(x, l, b, road)
    fig_scatter(x, l, metrics)
    fig_mean_congestion_by_road()
    print("Wrote figures to", OUT_DIR, "| road shown in fig 2:", road)


if __name__ == "__main__":
    main()

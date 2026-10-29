"""
Logic behind the dashboard's "Live Prediction Demo" tab (no Streamlit code here, so it can be tested).

What the models forecast: the Congestion Level (0-100) of ONE road on the day after the last observation (t + 1),
from that road's recent observations (see src/dataProcessing/forecast_data.py).

Input sources the demo supports (always labelled in the UI):
  * "Historical replay"   : real observations from the dataset ending at a chosen day t.
  * "What-if scenario"    : a historical replay whose LAST observation has been edited by the user.
Neither is real-time traffic.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.app import artifacts
from src.dataProcessing import forecast_data as fd
from src.modelTraining import predictLstm, predictXgboost

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"

# Editable fields of the last observation (the day-t conditions) and the dataset's valid range is read from the data.
EDITABLE_NUMERIC = ["Congestion Level", "Average Speed", "Traffic Volume", "Travel Time Index",
                    "Road Capacity Utilization", "Incident Reports"]
PRESETS = {
    "Heavy congestion (road's 90th-percentile conditions)": 0.9,
    "Light traffic (road's 10th-percentile conditions)": 0.1,
}
SPEED_COLUMN = "Average Speed"  # for speed the "heavy" direction is the opposite (low speed)


OBS_FORMAT = {"Congestion Level": "{:.1f}", "Average Speed": "{:.1f}", "Traffic Volume": "{:,.0f}",
              "Travel Time Index": "{:.2f}", "Road Capacity Utilization": "{:.1f}", "Incident Reports": "{:.0f}"}


def observation_table(row) -> pd.DataFrame:
    """Day-t values for display, rounded sensibly (the raw floats have many digits). Display only: models get the raw values."""
    rows = [(k, OBS_FORMAT[k].format(row[k])) for k in EDITABLE_NUMERIC] + [("Weather Conditions", str(row["Weather Conditions"]))]
    return pd.DataFrame(rows, columns=["Field", "Day-t value"])


class DemoInputError(ValueError):
    """The supplied history cannot be fed to the models; the message says how to fix it."""


def file_version(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]


def model_version() -> dict:
    """Short content hashes so the presenter can show exactly which trained artifacts produced a prediction."""
    artifacts.require(("XGBoost model", "LSTM model", "LSTM scaler"))
    return {
        "XGBoost": file_version(MODEL_DIR / "xgboost_model.json"),
        "LSTM": file_version(MODEL_DIR / "model.h5"),
        "LSTM scaler": file_version(MODEL_DIR / "scaler.pkl"),
    }


class DemoContext:
    """Dataset + split information needed by the demo (built once and cached by the app)."""

    def __init__(self):
        """Raises DatasetError / ArtifactError (clear messages) if the dataset or the trained files cannot be used."""
        self.df = fd.load_raw()
        self.cfg = artifacts.read_json("xgboost_features.json", "XGBoost feature configuration")
        try:
            self.roads = self.cfg["roads"]
            self.train_end = pd.Timestamp(self.cfg["split_train_end"])
            self.val_end = pd.Timestamp(self.cfg["split_val_end"])
            seq_len = self.cfg["seq_len"]
        except KeyError as e:
            raise artifacts.ArtifactError(f"xgboost_features.json lacks the key {e}; it was not written by trainXgboost.py.") from e
        if seq_len != fd.SEQ_LEN:
            raise artifacts.ArtifactError(f"The trained models use windows of {seq_len} observations but this code expects {fd.SEQ_LEN}: "
                                          "the model files and the code are from different versions.")
        data_roads, model_roads = set(fd.road_names(self.df)), set(self.roads)
        if data_roads != model_roads:
            raise fd.DatasetError("The dataset is not compatible with the trained models: roads only in the dataset: "
                                  f"{sorted(data_roads - model_roads)[:5]}; roads the models know but the dataset lacks: "
                                  f"{sorted(model_roads - data_roads)[:5]}. Use the original Banglore_traffic_Dataset.csv.")
        meta, _, _, _ = fd.load_dataset()
        self.road_train_mean = meta[meta["split"] == "train"].groupby("Road")["y"].mean().to_dict()
        self.ranges = self.df[EDITABLE_NUMERIC].agg(["min", "max"])
        self.test_metrics = None
        mp = MODEL_DIR / "metrics.json"
        if mp.exists():
            try:
                self.test_metrics = json.loads(mp.read_text(encoding="utf-8"))["metrics"]
            except (json.JSONDecodeError, KeyError):
                self.test_metrics = None  # optional reference numbers only; the demo works without them

    def replay_dates(self, road: str, test_only: bool = True) -> list:
        """Days t for which a valid history exists. test_only: t+1 lies in the held-out test period."""
        obs = self.df[self.df[fd.ROAD_COL] == road][fd.DATE_COL].sort_values().reset_index(drop=True)
        dates = obs.iloc[fd.SEQ_LEN:]  # needs SEQ_LEN + 1 observations up to and including day t
        if test_only:
            dates = dates[dates + pd.Timedelta(days=1) > self.val_end]
        return list(dates)

    def history(self, road: str, t, n_obs: int = 14) -> pd.DataFrame:
        d = self.df[(self.df[fd.ROAD_COL] == road) & (self.df[fd.DATE_COL] <= pd.Timestamp(t))]
        return d.sort_values(fd.DATE_COL).tail(n_obs).reset_index(drop=True)

    def actual_next_day(self, road: str, t):
        nxt = pd.Timestamp(t) + pd.Timedelta(days=1)
        row = self.df[(self.df[fd.ROAD_COL] == road) & (self.df[fd.DATE_COL] == nxt)]
        return None if row.empty else float(row[fd.TARGET_COL].iloc[0])

    def split_of(self, target_date) -> str:
        td = pd.Timestamp(target_date)
        return "train" if td <= self.train_end else "validation" if td <= self.val_end else "test"

    def preset_values(self, road: str, preset: str) -> dict:
        """Last-observation values at the road's 10th/90th percentile over the TRAINING period only."""
        q = PRESETS[preset]
        d = self.df[(self.df[fd.ROAD_COL] == road) & (self.df[fd.DATE_COL] <= self.train_end)]
        out = {}
        for col in EDITABLE_NUMERIC:
            qq = 1 - q if col == SPEED_COLUMN else q  # heavy congestion = LOW speed
            out[col] = float(d[col].quantile(qq))
        out["Incident Reports"] = float(round(out["Incident Reports"]))
        return out


def validate_history(hist: pd.DataFrame, ctx: DemoContext) -> list:
    """Raise DemoInputError for unusable input; return a list of (non-fatal) warnings."""
    need = fd.REQUIRED_RAW_COLS
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise DemoInputError(f"The history is missing required columns: {missing}.")
    if hist[fd.ROAD_COL].nunique() != 1:
        raise DemoInputError("The history must contain exactly one road.")
    if hist[fd.ROAD_COL].iloc[0] not in ctx.roads:
        raise DemoInputError(f"Unknown road '{hist[fd.ROAD_COL].iloc[0]}'. The models know: {ctx.roads}")
    if len(hist) < fd.SEQ_LEN + 1:
        raise DemoInputError(f"Need at least {fd.SEQ_LEN + 1} consecutive observations of this road; got {len(hist)}. "
                             "Pick a later date for this road.")
    if hist[need].isna().any().any():
        raise DemoInputError("The history contains missing values; the models do not accept gaps in the features.")
    if not hist[fd.DATE_COL].is_monotonic_increasing or hist[fd.DATE_COL].duplicated().any():
        raise DemoInputError("Observations must be in strictly increasing date order without duplicates.")
    if hist["Weather Conditions"].iloc[-1] not in fd.WEATHER_CATEGORIES:
        raise DemoInputError(f"Weather must be one of {fd.WEATHER_CATEGORIES}.")

    warnings = []
    last = hist.iloc[-1]
    for col in EDITABLE_NUMERIC:
        lo, hi = ctx.ranges.loc["min", col], ctx.ranges.loc["max", col]
        if not lo <= last[col] <= hi:
            warnings.append(f"{col}={last[col]:.2f} is outside the range seen in the dataset [{lo:.2f}, {hi:.2f}]; "
                            "the models were never trained on such values.")
    return warnings


def apply_scenario(hist: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Return a copy of the history whose LAST row (day t) has the given values; earlier rows are untouched."""
    out = hist.copy()
    for col in overrides:
        if col in EDITABLE_NUMERIC:
            out[col] = out[col].astype(float)  # avoid int->float dtype warnings when assigning edited values
    for col, val in overrides.items():
        if col not in out.columns:
            raise DemoInputError(f"Unknown field '{col}'.")
        out.loc[out.index[-1], col] = val
    return out


def predict_all(hist: pd.DataFrame, ctx: DemoContext) -> dict:
    """Run both trained models and both baselines on the same history. Errors propagate (nothing is hidden)."""
    warnings = validate_history(hist, ctx)
    road = hist[fd.ROAD_COL].iloc[0]
    last_date = hist[fd.DATE_COL].iloc[-1]
    return {
        "road": road,
        "last_observation_date": last_date,
        "target_date": last_date + pd.Timedelta(days=1),
        "horizon": "1 day ahead (next calendar day)",
        "XGBoost": predictXgboost.predict_next_congestion(hist),
        "LSTM": predictLstm.predict_next_congestion(hist),
        "Persistence baseline": float(hist[fd.TARGET_COL].iloc[-1]),
        "Road-mean baseline": float(ctx.road_train_mean[road]),
        "warnings": warnings,
        "model_version": model_version(),
    }

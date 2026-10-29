"""
Shared data pipeline for BOTH models (XGBoost and LSTM).

Forecasting task (the same for both models)
-------------------------------------------
* Series      : one series per road/intersection (16 roads, one row per day when observed).
* Target      : "Congestion Level" (0-100) of the SAME road on the NEXT calendar day (t + 1).
* Prediction time: the end of day t. Only information from day t and earlier is used,
                   plus the calendar of the target day (weekday / month), which is known in advance.
* Inputs      : the last SEQ_LEN observations of that road (up to and including day t).
* Samples     : a (road, day t) pair is used only if the road was also observed on day t + 1
                (so the target exists) and has SEQ_LEN observations of history.
* Split       : chronological, by TARGET date (train 70% / validation 15% / test 15% of the dates).
                Nothing is shuffled.
* Both models are trained/evaluated on exactly the same samples.

The dataset is daily (every timestamp is midnight) and has gaps, so a road's history window is
"the last SEQ_LEN observations", and the feature `days_since_prev` tells the model how far apart they were.
"""
import os
import numpy as np
import pandas as pd

from src.app.utils import get_project_root

DATA_PATH = os.path.join(get_project_root(), "data", "raw", "Banglore_traffic_Dataset.csv")

DATE_COL = "Date"
ROAD_COL = "Road/Intersection Name"
TARGET_COL = "Congestion Level"
SEQ_LEN = 7
TARGET_SCALE = 100.0  # Congestion Level is a percentage (0-100)

NUMERIC_COLS = [
    "Congestion Level", "Average Speed", "Traffic Volume",
    "Travel Time Index", "Road Capacity Utilization", "Incident Reports",
]
WEATHER_CATEGORIES = ["Clear", "Fog", "Overcast", "Rain", "Windy"]
WEATHER_COLS = [f"Weather_{w}" for w in WEATHER_CATEGORIES]
REQUIRED_RAW_COLS = [DATE_COL, ROAD_COL, "Weather Conditions"] + NUMERIC_COLS

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 15% of the dates = test


MIN_USABLE_SAMPLES = 300  # below this the chronological 70/15/15 split is meaningless


class DatasetError(ValueError):
    """The dataset cannot be used; the message says what is wrong and how to fix it."""


class DatasetNotFoundError(DatasetError, FileNotFoundError):
    """The dataset file does not exist."""


def _first_bad(series: pd.Series, bad_mask) -> str:
    bad = series[bad_mask]
    return ", ".join(repr(v) for v in bad.head(3).tolist()) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset (no values are filled in or invented). Raises DatasetError with a clear reason."""
    if not os.path.exists(path):
        raise DatasetNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Banglore_traffic_Dataset.csv' (Kaggle: preethamgouda/banglore-city-traffic-dataset) "
            "and place it in data/raw/."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise DatasetError(f"Dataset file is empty: {path}. Download the original CSV again.") from e
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DatasetError(f"Dataset file is not a readable CSV ({type(e).__name__}: {e}). Download the original CSV again.") from e
    if df.empty:
        raise DatasetError("Dataset has a header but no rows. Download the original CSV again.")
    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns {missing}. Found columns: {list(df.columns)}. "
                           "This does not look like Banglore_traffic_Dataset.csv.")
    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    except (ValueError, TypeError) as e:
        raise DatasetError(f"Column '{DATE_COL}' contains values that are not dates ({e}).") from e
    for c in NUMERIC_COLS:
        coerced = pd.to_numeric(df[c], errors="coerce")
        bad = coerced.isna() & df[c].notna()
        if bad.any():
            raise DatasetError(f"Column '{c}' contains non-numeric values: {_first_bad(df[c], bad)}.")
        df[c] = coerced
    nulls = df[REQUIRED_RAW_COLS].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise DatasetError("Dataset has missing values (nothing is imputed): "
                           + ", ".join(f"{k}: {v} rows" for k, v in nulls.items()) + ". Fix the file explicitly before use.")
    unknown = sorted(set(df["Weather Conditions"]) - set(WEATHER_CATEGORIES))
    if unknown:
        raise DatasetError(f"Unknown weather categories {unknown}; the models know {WEATHER_CATEGORIES}.")
    if df.duplicated([DATE_COL, ROAD_COL]).any():
        raise DatasetError(f"Dataset has {int(df.duplicated([DATE_COL, ROAD_COL]).sum())} duplicate (Date, Road) rows.")
    out = df[TARGET_COL][(df[TARGET_COL] < 0) | (df[TARGET_COL] > 100)]
    if len(out):
        raise DatasetError(f"'{TARGET_COL}' must be within 0-100 but {len(out)} rows are outside (e.g. {out.head(3).tolist()}).")
    return df.sort_values([ROAD_COL, DATE_COL]).reset_index(drop=True)


def road_names(df: pd.DataFrame) -> list:
    return sorted(df[ROAD_COL].unique())


def seq_feature_names(roads: list) -> list:
    return NUMERIC_COLS + WEATHER_COLS + ["days_since_prev"] + [f"Road_{r}" for r in roads]


def observation_features(df_road: pd.DataFrame, roads: list) -> pd.DataFrame:
    """
    Per-observation features for ONE road (rows must be that road only).
    Returns a frame sorted by date with columns seq_feature_names(roads) plus Date.
    Used identically for training and for inference.
    """
    d = df_road.sort_values(DATE_COL).reset_index(drop=True)
    out = pd.DataFrame({DATE_COL: d[DATE_COL]})
    for c in NUMERIC_COLS:
        out[c] = d[c].astype(float)
    for w in WEATHER_CATEGORIES:
        out[f"Weather_{w}"] = (d["Weather Conditions"] == w).astype(float)
    out["days_since_prev"] = d[DATE_COL].diff().dt.days.astype(float)  # NaN for the first observation
    name = d[ROAD_COL].iloc[0]
    for r in roads:
        out[f"Road_{r}"] = float(name == r)
    return out


def tabular_from_sequences(X_seq: np.ndarray, target_dates, roads: list) -> pd.DataFrame:
    """
    Flat feature table for XGBoost, built from the SAME windows the LSTM sees.
    X_seq: (n, SEQ_LEN, n_features) raw (unscaled) windows ending at day t.
    Every feature uses day t or earlier, except the calendar of the target day.
    """
    names = seq_feature_names(roads)
    idx = {n: i for i, n in enumerate(names)}
    last = X_seq[:, -1, :]
    tab = pd.DataFrame(last, columns=names)  # state at day t (incl. day-t congestion = lag 0)

    for col in ["Congestion Level", "Traffic Volume", "Average Speed"]:
        j = idx[col]
        tab[f"{col}_prev1"] = X_seq[:, -2, j]  # previous observation
        tab[f"{col}_prev2"] = X_seq[:, -3, j]
        tab[f"{col}_mean3"] = X_seq[:, -3:, j].mean(axis=1)  # last 3 observations, ending at day t
    tab["Congestion Level_mean7"] = X_seq[:, :, idx["Congestion Level"]].mean(axis=1)

    td = pd.to_datetime(pd.Series(np.asarray(target_dates)))
    tab["target_dayofweek"] = td.dt.dayofweek.values
    tab["target_is_weekend"] = td.dt.dayofweek.isin([5, 6]).astype(int).values
    tab["target_month_sin"] = np.sin(2 * np.pi * td.dt.month.values / 12)
    tab["target_month_cos"] = np.cos(2 * np.pi * td.dt.month.values / 12)
    return tab


def build_samples(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Build windows per road (windows never cross roads).
    Returns (meta, X_seq): meta has Road, Date (day t), Target Date (t+1), y; X_seq is (n, seq_len, F).
    """
    roads = road_names(df)
    metas, windows = [], []
    for road, g in df.groupby(ROAD_COL, sort=True):
        feats = observation_features(g, roads)
        dates = feats[DATE_COL].values
        values = feats.drop(columns=[DATE_COL]).values
        y_all = feats[TARGET_COL].values
        for i in range(seq_len, len(feats) - 1):  # i >= seq_len: window has a defined days_since_prev
            if (dates[i + 1] - dates[i]) != np.timedelta64(1, "D"):
                continue  # target day (t+1) not observed for this road
            windows.append(values[i - seq_len + 1: i + 1])
            metas.append((road, dates[i], dates[i + 1], y_all[i + 1]))
    if not windows:
        raise DatasetError(
            f"No usable samples: a sample needs a road with at least {seq_len + 2} observations in which day t and day t+1 are both observed. "
            f"The dataset has {len(df)} rows over {df[ROAD_COL].nunique()} road(s), which is too little. "
            "Use the full Banglore_traffic_Dataset.csv.")
    meta = pd.DataFrame(metas, columns=["Road", "Date", "Target Date", "y"])
    X_seq = np.stack(windows).astype(np.float32)
    return meta, X_seq


def chronological_split(meta: pd.DataFrame):
    """Assign split labels by TARGET date: earliest 70% of dates = train, next 15% = val, last 15% = test."""
    dates = np.sort(meta["Target Date"].unique())
    n = len(dates)
    train_end = dates[int(n * TRAIN_FRAC) - 1]
    val_end = dates[int(n * (TRAIN_FRAC + VAL_FRAC)) - 1]
    split = np.where(meta["Target Date"] <= train_end, "train",
                     np.where(meta["Target Date"] <= val_end, "val", "test"))
    return pd.Series(split, index=meta.index, name="split"), train_end, val_end


def load_dataset(path: str = DATA_PATH):
    """Everything the training scripts need: meta (with split), raw windows, roads."""
    df = load_raw(path)
    roads = road_names(df)
    meta, X_seq = build_samples(df)
    if len(meta) < MIN_USABLE_SAMPLES:
        raise DatasetError(f"Dataset is too small: only {len(meta)} usable samples (at least {MIN_USABLE_SAMPLES} are needed for a "
                           "meaningful chronological train/validation/test split). Use the full Banglore_traffic_Dataset.csv.")
    meta["split"], train_end, val_end = chronological_split(meta)
    return meta, X_seq, roads, (train_end, val_end)

"""
Stage 5A: signal-gate benchmark for next-day congestion forecasting.

QUESTION: does the data contain within-road, time-dependent signal that a learned model could use beyond each road's usual level?
This module answers it WITHOUT training any model. It uses only fixed-form statistics of past data (means, last values) and
permutation / bootstrap tests, and it never changes anything outside its own output directory.

What it reuses, read-only
  * src/dataProcessing/forecast_data.py: load_raw(), SEQ_LEN and chronological_split(), so the target (the SAME road on the NEXT
    calendar day), the sample rules (7 earlier observations, next day observed) and the 70/15/15 split by target date are the project's own.

What it does
  1. Builds the project's samples and checks nothing else about them (tests assert they equal forecast_data.load_dataset()).
  2. Evaluates baselines on (a) the project's official test split and (b) a rolling-origin evaluation (about 5 chronological folds
     after a warm-up of the first half of the target dates; fewer folds when the sample size does not permit).
  3. Reports MAE, RMSE, R2, the within-road correlation of prediction and actual, and the RMSE difference to the REFERENCE
     (the expanding per-road mean) with a date-cluster bootstrap confidence interval.
  4. Tests for within-road temporal autocorrelation and calendar effects on the DEVELOPMENT period only (observations dated up to the
     end of the validation split), so the test labels play no part in the gate decision.
  5. Gives a verdict: is there evidence of usable signal? If not, training a new learned model on this data is not justified.

Information rules (no label leakage)
  * Every baseline prediction for day t+1 uses only observations dated <= t. The expanding per-road mean includes day t itself
    (known at the end of day t, the project's prediction time) and nothing later.
  * Fold-origin baselines (per-road train mean, per-road weekday mean, global mean) use only samples whose target date is BEFORE the
    first target date of the evaluated block.
  * No parameter is tuned anywhere: there is nothing to select on any labels.
  * Bootstrap and permutation tests use fixed seeds; the same input gives the same output.

Output: signal_check.json and signal_check.md in an output directory of its own (default experiments/stage5a/). It refuses to write
into model/, data/, the documents or any other project directory, so saved models, scalers, predictions, metrics and reports cannot be
overwritten. Saved prediction CSVs in model/ are only READ, to compare the saved LSTM and XGBoost with the same reference.

Run from the project root:   python -m src.analysis.signal_check            (add --out DIR, --n-boot N, --seed S)
"""
import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.dataProcessing import forecast_data as fd  # noqa: E402

DEFAULT_OUT = root_dir / "experiments" / "stage5a"
PROTECTED_TOP_LEVEL = {"model", "data", "doucments", "reports", "src", "tests", "docs_tests", "notebooks", "audit_backup", ".streamlit"}
OUTPUT_FILES = ("signal_check.json", "signal_check.md")

N_FOLDS = 5
WARMUP_FRAC = 0.5          # the first half of the target dates is only ever training data in the rolling-origin evaluation
MIN_FOLD_SAMPLES = 100     # a fold needs at least this many test samples ("where sample sizes permit")
MIN_TRAIN_SAMPLES = 300
N_BOOT = 2000
N_PERM = 1000
SEED = 0
ALPHA = 0.05
LAGS = (1, 2, 3, 7, 14)
MIN_AUTOCORR = 0.05        # smallest autocorrelation considered useful, in addition to being distinguishable from zero
MIN_ADJ_R2 = 0.01          # smallest adjusted R2 of a calendar effect considered useful
CALENDAR_TESTS = ("weekday", "month_of_year", "calendar_date", "road_month")
REFERENCE = "expanding_road_mean"
BASELINES = ("global_train_mean", "road_train_mean", "expanding_road_mean", "persistence", "mean_last_7_obs", "road_weekday_train_mean")
SAVED_MODELS = {"saved_LSTM": "lstm_predictions.csv", "saved_XGBoost": "xgboost_predictions.csv"}


# ================================================================================================ samples and folds
def observations_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """The generic observation table used everywhere here: columns road, date, y (one row per road and observed day)."""
    return pd.DataFrame({"road": raw[fd.ROAD_COL].values, "date": pd.to_datetime(raw[fd.DATE_COL]).values,
                         "y": raw[fd.TARGET_COL].astype(float).values})


def make_samples(obs: pd.DataFrame, seq_len: int = fd.SEQ_LEN) -> pd.DataFrame:
    """
    One row per (road, day t) whose target day t+1 is observed and which has seq_len earlier observations (the project's rule).
    Columns: Road, Date (day t), Target Date (t+1), y (target), y_t (today), mean_last_7 (last 7 observations up to t),
    expanding_mean (mean of ALL observations of the road up to and including day t).
    """
    rows = []
    for road, g in obs.sort_values(["road", "date"]).groupby("road", sort=True):
        d, y = g["date"].values, g["y"].values.astype(float)
        csum = np.cumsum(y)
        for i in range(seq_len, len(g) - 1):
            if d[i + 1] - d[i] != np.timedelta64(1, "D"):
                continue  # the next calendar day is not observed
            rows.append((road, d[i], d[i + 1], y[i + 1], y[i], y[i - 6: i + 1].mean(), csum[i] / (i + 1)))
    if not rows:
        raise fd.DatasetError("No usable samples (a road needs seq_len + 2 observations with consecutive days).")
    s = pd.DataFrame(rows, columns=["Road", "Date", "Target Date", "y", "y_t", "mean_last_7", "expanding_mean"])
    s["split"], _, _ = fd.chronological_split(s)  # the project's own 70/15/15 split by target date
    return s


@dataclass
class Fold:
    index: int
    train: np.ndarray            # boolean mask over the samples
    test: np.ndarray
    first_test_date: pd.Timestamp
    last_test_date: pd.Timestamp


def rolling_origin_folds(samples: pd.DataFrame, n_folds: int = N_FOLDS, warmup_frac: float = WARMUP_FRAC,
                         min_fold_samples: int = MIN_FOLD_SAMPLES, min_train_samples: int = MIN_TRAIN_SAMPLES) -> list:
    """
    Expanding-window, chronological folds BY TARGET DATE. The first `warmup_frac` of the dates is training only; the remaining dates
    are cut into contiguous blocks. Fold k trains on every sample whose target date is before its block and tests on the block.
    The number of folds is reduced when the samples cannot give `min_fold_samples` per fold.
    """
    dates = np.sort(samples["Target Date"].unique())
    rest = dates[int(len(dates) * warmup_frac):]
    if len(rest) == 0:
        raise ValueError("Too few dates for a rolling-origin evaluation.")
    n_rest = int((samples["Target Date"] >= rest[0]).sum())
    k = max(1, min(n_folds, n_rest // min_fold_samples))
    folds = []
    for i, block in enumerate(np.array_split(rest, k)):
        first, last = block[0], block[-1]
        train = (samples["Target Date"] < first).values
        test = ((samples["Target Date"] >= first) & (samples["Target Date"] <= last)).values
        if train.sum() < min_train_samples:
            raise ValueError(f"Fold {i + 1} would train on only {int(train.sum())} samples (< {min_train_samples}).")
        folds.append(Fold(i + 1, train, test, pd.Timestamp(first), pd.Timestamp(last)))
    return folds


# ================================================================================================ baselines and metrics
def baseline_predictions(samples: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    """
    Predictions of every baseline for the test rows. `train` selects the samples whose labels may be used by fold-origin baselines;
    the history-based baselines (expanding mean, last value, last-7 mean) only ever look at observations up to day t.
    """
    tr, te = samples[train], samples[test]
    glob = float(tr["y"].mean())
    road_mean = tr.groupby("Road")["y"].mean()
    p_road = te["Road"].map(road_mean).fillna(glob).values
    wd = tr.assign(dow=tr["Target Date"].dt.dayofweek).groupby(["Road", "dow"])["y"].mean()
    dow = te["Target Date"].dt.dayofweek.values  # the weekday of the target day is known in advance
    p_wd = np.array([wd.get((r, d), np.nan) for r, d in zip(te["Road"].values, dow)])
    p_wd = np.where(np.isnan(p_wd), p_road, p_wd)
    return pd.DataFrame({"global_train_mean": np.full(len(te), glob), "road_train_mean": p_road,
                         "expanding_road_mean": te["expanding_mean"].values, "persistence": te["y_t"].values,
                         "mean_last_7_obs": te["mean_last_7"].values, "road_weekday_train_mean": p_wd}, index=te.index)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sst = float(((y - y.mean()) ** 2).sum())
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "R2": float(1 - (e ** 2).sum() / sst) if sst > 0 else float("nan")}


def within_road_corr(roads, y, p) -> float:
    """Correlation of prediction and actual after removing each road's mean OVER THE EVALUATED ROWS (NaN if a prediction does not vary within roads)."""
    df = pd.DataFrame({"r": roads, "y": np.asarray(y, float), "p": np.asarray(p, float)})
    dy = df["y"] - df.groupby("r")["y"].transform("mean")
    dp = df["p"] - df.groupby("r")["p"].transform("mean")
    if dp.std() < 1e-9 or dy.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(dy, dp)[0, 1])


class DateBootstrap:
    """Cluster bootstrap over target DATES (rows on one date are correlated, so whole dates are resampled). Seeded and reusable."""

    def __init__(self, dates, n_boot=N_BOOT, seed=SEED):
        dates = np.asarray(dates)
        groups = [np.where(dates == d)[0] for d in np.unique(dates)]
        rng = np.random.default_rng(seed)
        self.idx = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]

    def rmse_diff(self, y, p_a, p_b):
        """RMSE(a) - RMSE(b): observed value and the bootstrap draws."""
        ea, eb = (np.asarray(y, float) - p_a) ** 2, (np.asarray(y, float) - p_b) ** 2
        obs = float(np.sqrt(ea.mean()) - np.sqrt(eb.mean()))
        return obs, np.array([np.sqrt(ea[ix].mean()) - np.sqrt(eb[ix].mean()) for ix in self.idx])


def summarize(frame: pd.DataFrame, candidates, n_boot=N_BOOT, seed=SEED, alpha=ALPHA) -> dict:
    """
    frame: columns Road, Target Date, y and one column per candidate. Returns metrics of every candidate and, for every candidate other
    than the reference, RMSE(candidate) - RMSE(reference) with a 95% date-cluster bootstrap CI and a Bonferroni-adjusted CI
    (adjusted for the number of comparisons made here); positive = WORSE than the reference.
    """
    y = frame["y"].values
    boot = DateBootstrap(frame["Target Date"].values, n_boot, seed)
    others = [c for c in candidates if c != REFERENCE]
    lo_b, hi_b = 100 * alpha / (2 * len(others)), 100 * (1 - alpha / (2 * len(others)))
    out = {"n_samples": int(len(frame)), "n_dates": int(frame["Target Date"].nunique()), "candidates": {}}
    for c in candidates:
        m = metrics(y, frame[c].values)
        m["within_road_corr"] = within_road_corr(frame["Road"].values, y, frame[c].values)
        if c != REFERENCE:
            obs, draws = boot.rmse_diff(y, frame[c].values, frame[REFERENCE].values)
            ci, cib = np.percentile(draws, [2.5, 97.5]), np.percentile(draws, [lo_b, hi_b])
            m["rmse_minus_reference"] = {"observed": obs, "ci95": [float(ci[0]), float(ci[1])], "ci_bonferroni": [float(cib[0]), float(cib[1])],
                                         "excludes_zero_bonferroni": bool(cib[0] > 0 or cib[1] < 0)}
        out["candidates"][c] = m
    return out


def official_frame(samples: pd.DataFrame, model_dir=None) -> tuple:
    """The project's official test split scored with baselines fitted on the official TRAIN split; saved models are added when readable."""
    train, test = (samples["split"] == "train").values, (samples["split"] == "test").values
    frame = samples.loc[test, ["Road", "Target Date", "y"]].join(baseline_predictions(samples, train, test))
    candidates, note = list(BASELINES), None
    mdir = Path(model_dir) if model_dir else root_dir / "model"
    added = []
    for name, fname in SAVED_MODELS.items():
        path = mdir / fname
        if not path.is_file():
            continue
        saved = pd.read_csv(path)  # READ ONLY
        saved = saved.rename(columns={"Road/Intersection Name": "Road", "Predicted Congestion": name})
        saved["Target Date"] = pd.to_datetime(saved["Date"])
        merged = frame.reset_index().merge(saved[["Road", "Target Date", name, "Actual Congestion"]], on=["Road", "Target Date"], how="left")
        if len(merged) == len(frame) and merged[name].notna().all() and np.allclose(merged["Actual Congestion"], merged["y"]):
            frame[name] = merged[name].values
            candidates.append(name)
            added.append(name)
        else:
            note = f"{fname} is not aligned with the official test rows and was not used."
    return frame, candidates, added, note


# ================================================================================================ signal diagnostics
def standardized_deviations(obs: pd.DataFrame) -> pd.DataFrame:
    """Each observation minus its road's mean, divided by the road's standard deviation (computed on the given observations only)."""
    g = obs.groupby("road")["y"]
    z = obs.assign(mu=g.transform("mean"), sd=g.transform("std"), n=g.transform("size"))
    z = z[(z["sd"] > 0) & (z["n"] >= 3)].copy()
    z["z"] = (z["y"] - z["mu"]) / z["sd"]
    return z[["road", "date", "z"]].reset_index(drop=True)


def _corr_from_sums(n, sa, sb, saa, sbb, sab):
    den = np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
    return (n * sab - sa * sb) / den if den > 0 else float("nan")


def autocorrelation(z: pd.DataFrame, lag: int, n_boot=N_BOOT, seed=SEED, alpha=ALPHA) -> dict:
    """Correlation of road-standardised values `lag` calendar days apart (exact lag, same road), with a date-cluster bootstrap CI."""
    later = z.assign(date=z["date"] - pd.Timedelta(days=lag)).rename(columns={"z": "b"})
    pairs = z.rename(columns={"z": "a"}).merge(later, on=["road", "date"])
    if len(pairs) < 30:
        return {"lag_days": lag, "n_pairs": int(len(pairs)), "r": float("nan"), "ci95": [float("nan")] * 2, "ci_bonferroni": [float("nan")] * 2, "useful": False}
    pairs["aa"], pairs["bb"], pairs["ab"] = pairs.a ** 2, pairs.b ** 2, pairs.a * pairs.b
    st = pairs.groupby("date").agg(n=("a", "size"), sa=("a", "sum"), sb=("b", "sum"), saa=("aa", "sum"), sbb=("bb", "sum"), sab=("ab", "sum")).values
    tot = st.sum(axis=0)
    r = _corr_from_sums(*tot)
    rng = np.random.default_rng(seed + lag)
    draws = np.array([_corr_from_sums(*st[rng.integers(0, len(st), len(st))].sum(axis=0)) for _ in range(n_boot)])
    m = len(LAGS) + len(CALENDAR_TESTS)
    ci, cib = np.nanpercentile(draws, [2.5, 97.5]), np.nanpercentile(draws, [100 * alpha / (2 * m), 100 * (1 - alpha / (2 * m))])
    excl = bool(cib[0] > 0 or cib[1] < 0)
    return {"lag_days": lag, "n_pairs": int(len(pairs)), "r": float(r), "ci95": [float(ci[0]), float(ci[1])],
            "ci_bonferroni": [float(cib[0]), float(cib[1])], "useful": bool(excl and abs(r) >= MIN_AUTOCORR)}


def calendar_effect(z: pd.DataFrame, labels, name: str, n_perm=N_PERM, seed=SEED, alpha=ALPHA) -> dict:
    """
    Share of the road-standardised variance explained by a grouping (adjusted R2, which is ~0 for pure noise) and a permutation p-value.
    labels: one group label per row of z.
    """
    codes, uniques = pd.factorize(pd.Series(np.asarray(labels)))
    k, n = len(uniques), len(z)
    v = z["z"].values
    cnt = np.bincount(codes, minlength=k).astype(float)

    def ss_between(values):
        s = np.bincount(codes, weights=values, minlength=k)
        return float((s ** 2 / np.maximum(cnt, 1)).sum() - values.sum() ** 2 / n)
    sst = float(((v - v.mean()) ** 2).sum())
    obs = ss_between(v)
    r2 = obs / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    rng = np.random.default_rng(seed)
    perm = np.array([ss_between(rng.permutation(v)) for _ in range(n_perm)])
    p = float((1 + (perm >= obs).sum()) / (1 + n_perm))
    m = len(LAGS) + len(CALENDAR_TESTS)
    return {"effect": name, "n_groups": int(k), "n_rows": int(n), "r2": float(r2), "adjusted_r2": float(adj), "permutation_p": p,
            "useful": bool(adj >= MIN_ADJ_R2 and p < alpha / m)}


def signal_diagnostics(obs: pd.DataFrame, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED) -> dict:
    z = standardized_deviations(obs)
    d = pd.to_datetime(z["date"])
    cal = [calendar_effect(z, d.dt.dayofweek.values, "weekday", n_perm, seed),
           calendar_effect(z, d.dt.month.values, "month_of_year", n_perm, seed),
           calendar_effect(z, d.values, "calendar_date", n_perm, seed),
           calendar_effect(z, (z["road"].astype(str) + "|" + d.dt.to_period("M").astype(str)).values, "road_month", n_perm, seed)]
    return {"n_observations": int(len(z)), "autocorrelation": [autocorrelation(z, k, n_boot, seed) for k in LAGS], "calendar_effects": cal}


# ================================================================================================ gate
def gate_verdict(dev_diag: dict, pooled: dict) -> dict:
    """Signal is declared only if an effect is both distinguishable from noise (Bonferroni-adjusted) AND large enough to matter."""
    found = []
    for a in dev_diag["autocorrelation"]:
        if a["useful"]:
            found.append(f"within-road autocorrelation at lag {a['lag_days']} days: r = {a['r']:.3f}")
    for c in dev_diag["calendar_effects"]:
        if c["useful"]:
            found.append(f"calendar effect '{c['effect']}': adjusted R2 = {c['adjusted_r2']:.3f}")
    for name, m in pooled["candidates"].items():
        d = m.get("rmse_minus_reference")
        if d and d["observed"] < 0 and d["excludes_zero_bonferroni"]:
            found.append(f"'{name}' has lower RMSE than the expanding road mean ({d['observed']:+.3f}, CI excludes 0)")
    ok = not found
    return {"verdict": "NO_USABLE_SIGNAL" if ok else "SIGNAL_FOUND", "evidence_of_signal": found,
            "meaning": ("No evidence of within-road, time-dependent signal beyond each road's usual level was found on this data. Training a new learned "
                        "model on it is not justified; any learned model should be expected to match, not beat, the per-road mean."
                        if ok else
                        "Evidence of usable signal was found (see the list). A learning experiment on exactly that signal is justified, using the same "
                        "rolling-origin evaluation and the expanding road mean as its reference.")}


# ================================================================================================ run, output
def safe_output_dir(path) -> Path:
    """Refuse any directory that holds saved models, data, documents, reports, source or tests."""
    p = Path(path).resolve()
    if p == root_dir:
        raise ValueError("Refusing to write into the project root.")
    try:
        top = p.relative_to(root_dir).parts[0]
    except ValueError:
        return p  # outside the project (for example a temporary directory)
    if top in PROTECTED_TOP_LEVEL:
        raise ValueError(f"Refusing to write into '{top}/': it holds protected project files. Use a directory such as experiments/stage5a.")
    return p


def _py(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    raise TypeError(type(o))


def run(obs: pd.DataFrame = None, *, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED, model_dir=None, dataset_path=None) -> dict:
    """Run the whole benchmark and return the report (a plain dict). Nothing is written here."""
    provenance = {}
    if obs is None:
        path = Path(dataset_path) if dataset_path else Path(fd.DATA_PATH)
        raw = fd.load_raw(str(path))
        obs = observations_from_raw(raw)
        provenance = {"dataset_file": "data/raw/Banglore_traffic_Dataset.csv", "dataset_sha256_prefix": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}
    samples = make_samples(obs)
    split_dates = {s: [str(samples.loc[samples.split == s, "Target Date"].min().date()), str(samples.loc[samples.split == s, "Target Date"].max().date()),
                       int((samples.split == s).sum())] for s in ("train", "val", "test")}

    # --- official split
    frame, cand, saved_added, saved_note = official_frame(samples, model_dir)
    official = summarize(frame, cand, n_boot, seed)
    official["split"] = "project's official test split (baselines fitted on the official train split only)"
    official["saved_models_included"] = saved_added
    if saved_note:
        official["note"] = saved_note

    # --- rolling origin
    folds = rolling_origin_folds(samples)
    parts, fold_reports = [], []
    for f in folds:
        part = samples.loc[f.test, ["Road", "Target Date", "y"]].join(baseline_predictions(samples, f.train, f.test))
        parts.append(part)
        rep = summarize(part, BASELINES, n_boot, seed + f.index)
        rep.update({"fold": f.index, "train_samples": int(f.train.sum()), "train_target_dates_before": str(f.first_test_date.date()),
                    "test_target_dates": [str(f.first_test_date.date()), str(f.last_test_date.date())]})
        fold_reports.append(rep)
    pooled = summarize(pd.concat(parts), BASELINES, n_boot, seed)
    pooled["description"] = "all rolling-origin test rows pooled (each row scored by baselines that used only data available at its fold origin)"

    # --- signal diagnostics: decision uses development observations only (dates up to the end of the validation split)
    val_end = pd.Timestamp(samples.loc[samples.split == "val", "Target Date"].max())
    dev = obs[pd.to_datetime(obs["date"]) <= val_end]
    dev_diag = signal_diagnostics(dev, n_boot, n_perm, seed)
    all_diag = signal_diagnostics(obs, n_boot, n_perm, seed)
    gate = gate_verdict(dev_diag, pooled)

    return {
        "stage": "5A signal-gate benchmark", "provenance": provenance, "seed": seed, "n_boot": n_boot, "n_perm": n_perm,
        "data": {"observations": int(len(obs)), "roads": int(obs["road"].nunique()),
                 "date_range": [str(pd.to_datetime(obs["date"]).min().date()), str(pd.to_datetime(obs["date"]).max().date())]},
        "samples": {"total": int(len(samples)), "split_target_dates_and_counts": split_dates,
                    "rule": "the same road on the next calendar day; 7 earlier observations required; chronological 70/15/15 by target date (project's own function)"},
        "methodology": {
            "reference": REFERENCE + " (mean of every observation of the road up to and including day t)",
            "baselines": list(BASELINES),
            "no_tuning": "no parameter is fitted or selected on any labels",
            "rolling_origin": f"expanding window by target date; first {int(WARMUP_FRAC * 100)}% of dates are training only; remaining dates split into contiguous blocks; "
                              f"at most {N_FOLDS} folds, fewer when fewer than {MIN_FOLD_SAMPLES} test samples per fold are available",
            "diagnostics_period": f"observations dated up to {val_end.date()} (end of the validation split); test labels are not used for the gate",
            "bonferroni": f"{len(LAGS) + len(CALENDAR_TESTS)} diagnostic tests; {len(BASELINES) - 1} baseline comparisons (+ saved models on the official split)",
            "minimum_useful_effect": {"autocorrelation": MIN_AUTOCORR, "calendar_adjusted_r2": MIN_ADJ_R2}},
        "official_test": official, "rolling_origin": {"n_folds": len(folds), "folds": fold_reports, "pooled": pooled},
        "diagnostics_development_period": dev_diag, "diagnostics_all_data_descriptive": all_diag, "gate": gate,
        "limitations": [
            "Results describe THIS dataset only. Its provenance and measurement process are unverified, so nothing here is a statement about real-world traffic.",
            "About 43% of target values are exactly 100 (a cap); RMSE and MAE rank methods differently because of it.",
            "Each calendar day has only a few roads observed (about 9 of 16), and 35% of road days are missing; samples need the next day to be observed.",
            "The rolling-origin evaluation uses simple fixed baselines, not learned models: it tests whether signal exists, not how well a model could use it.",
            "Tests have limited power: effects smaller than the thresholds above cannot be excluded.",
            "Saved LSTM/XGBoost predictions are only read, and only the official test split is available for them."],
    }


def render_markdown(rep: dict) -> str:
    f = lambda x, n=3: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"
    L = ["# Stage 5A signal-gate benchmark", "",
         f"Verdict: **{rep['gate']['verdict']}**. {rep['gate']['meaning']}", ""]
    if rep["gate"]["evidence_of_signal"]:
        L += ["Evidence: " + "; ".join(rep["gate"]["evidence_of_signal"]), ""]
    d = rep["data"]
    L += [f"Data: {d['observations']} observations, {d['roads']} roads, {d['date_range'][0]} to {d['date_range'][1]}. "
          f"Samples: {rep['samples']['total']} (the same road on the next calendar day). Seed {rep['seed']}, {rep['n_boot']} bootstrap resamples, {rep['n_perm']} permutations.", ""]
    sp = rep["samples"]["split_target_dates_and_counts"]
    L += ["## Project split (target dates)", "", "| split | first | last | samples |", "|---|---|---|---|"] + [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in sp.items()] + [""]

    def table(block, cols=True):
        lines = ["| candidate | MAE | RMSE | R2 | within-road corr | RMSE minus reference (95% CI) |", "|---|---|---|---|---|---|"]
        for name, m in block["candidates"].items():
            r = m.get("rmse_minus_reference")
            diff = "reference" if r is None else f"{r['observed']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" + (" *" if r["excludes_zero_bonferroni"] else "")
            lines.append(f"| {name} | {f(m['MAE'])} | {f(m['RMSE'])} | {f(m['R2'])} | {f(m['within_road_corr'])} | {diff} |")
        return lines + ["", "Reference = expanding per-road mean. Positive difference = worse than the reference. `*` = CI excludes 0 after Bonferroni adjustment.", ""]
    o = rep["official_test"]
    L += [f"## Official test split ({o['n_samples']} samples, {o['n_dates']} dates)", ""] + table(o)
    L += ["## Rolling-origin folds", "", "| fold | train samples | test target dates | test samples |", "|---|---|---|---|"]
    L += [f"| {x['fold']} | {x['train_samples']} | {x['test_target_dates'][0]} to {x['test_target_dates'][1]} | {x['n_samples']} |" for x in rep["rolling_origin"]["folds"]] + [""]
    p = rep["rolling_origin"]["pooled"]
    L += [f"### Pooled over folds ({p['n_samples']} samples)", ""] + table(p)
    for key, title in (("diagnostics_development_period", "Signal diagnostics, development period (used for the gate)"),
                       ("diagnostics_all_data_descriptive", "Signal diagnostics, all data (descriptive only)")):
        dg = rep[key]
        L += [f"## {title}", "", "Road-standardised values; exact calendar lags within the same road.", "",
              "| lag (days) | pairs | r | 95% CI | useful |", "|---|---|---|---|---|"]
        L += [f"| {a['lag_days']} | {a['n_pairs']} | {f(a['r'])} | [{f(a['ci95'][0])}, {f(a['ci95'][1])}] | {a['useful']} |" for a in dg["autocorrelation"]]
        L += ["", "| calendar effect | groups | adjusted R2 | permutation p | useful |", "|---|---|---|---|---|"]
        L += [f"| {c['effect']} | {c['n_groups']} | {f(c['adjusted_r2'], 4)} | {f(c['permutation_p'])} | {c['useful']} |" for c in dg["calendar_effects"]] + [""]
    L += ["## Methodology", ""] + [f"* **{k}**: {v}" for k, v in rep["methodology"].items() if isinstance(v, str)] + ["", "## Limitations", ""] + [f"* {x}" for x in rep["limitations"]]
    return "\n".join(L) + "\n"


def write_outputs(rep: dict, out_dir) -> list:
    out = safe_output_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUTPUT_FILES[0]).write_text(json.dumps(rep, indent=2, default=_py), encoding="utf-8")
    (out / OUTPUT_FILES[1]).write_text(render_markdown(rep), encoding="utf-8")
    return [out / n for n in OUTPUT_FILES]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 5A signal-gate benchmark (no model is trained).")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (never model/, data/, documents, source or tests)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    safe_output_dir(a.out)  # fail before doing any work
    rep = run(n_boot=a.n_boot, n_perm=a.n_perm, seed=a.seed)
    paths = write_outputs(rep, a.out)
    g = rep["gate"]
    print(f"Verdict: {g['verdict']}\n{g['meaning']}")
    for e in g["evidence_of_signal"]:
        print("  evidence:", e)
    print("Wrote:", ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())

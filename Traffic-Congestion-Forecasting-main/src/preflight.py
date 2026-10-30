"""
Preflight check: is this machine ready to run the dashboard and the live demo?

    python -m src.preflight            full check (loads the models and makes a real sample prediction, ~15 s)
    python -m src.preflight --fast     quick check (no TensorFlow import, no prediction; used by the launchers)
    python -m src.preflight --json     machine-readable output

It only READS files; it never writes, trains or regenerates anything.

Groups
  Required: dashboard             Python, packages, app.py                              -> failure blocks the launch
  Required: live prediction demo  dataset, trained models, scaler, feature configs,     -> failure disables the demo
                                  consistency between them, and a sample prediction
  Recommended: Model Results tab  saved prediction files and metrics                    -> warning only
  Optional: PySpark / Java        only src.bigdata.spark_analysis needs it              -> never blocks anything
  Optional: TomTom real-time      only Mode 2 of the demo; it cannot make predictions   -> never blocks anything
Exit code: 0 when both required groups pass, 1 otherwise.
"""
import argparse
import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

G_DASH = "Required: dashboard"
G_DEMO = "Required: live prediction demo"
G_RESULTS = "Recommended: Model Results tab"
G_SPARK = "Optional: PySpark / Java (only the big-data script needs these)"
G_TOMTOM = "Optional: TomTom real-time mode (Mode 2 only; it cannot make predictions)"
REQUIRED_GROUPS = (G_DASH, G_DEMO)

# status values: OK, FAIL, WARN, SKIP, OPT (optional component absent: harmless)


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str = ""
    hint: str = ""


def _pins(requirements: Path) -> dict:
    pins = {}
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def _dashboard_checks(root: Path, fast: bool) -> list:
    out = []
    v = sys.version_info
    out.append(Check(G_DASH, "Python", "OK" if (3, 9) <= (v.major, v.minor) <= (3, 12) else "WARN",
                     f"{v.major}.{v.minor}.{v.micro} (tested with 3.11)",
                     "TensorFlow 2.16 supports Python 3.9-3.12; use the project's .venv." if not (3, 9) <= (v.major, v.minor) <= (3, 12) else ""))
    out.append(Check(G_DASH, "app.py", "OK" if (root / "app.py").is_file() else "FAIL", str(root / "app.py")))
    from src.ui.navigation import PAGES  # plain dataclasses: no Streamlit import
    gone = [pg.path for pg in PAGES if not (root / pg.path).is_file()]
    out.append(Check(G_DASH, f"dashboard pages ({len(PAGES)})", "FAIL" if gone else "OK",
                     "missing: " + ", ".join(gone) if gone else ", ".join(pg.title for pg in PAGES),
                     "Restore the missing files in src/ui/pages/." if gone else ""))
    for name, ver in _pins(root / "requirements.txt").items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            out.append(Check(G_DASH, f"package {name}", "FAIL", "not installed", "pip install -r requirements.txt"))
            continue
        out.append(Check(G_DASH, f"package {name}", "OK" if have == ver else "WARN",
                         f"{have}" + ("" if have == ver else f" (tested with {ver})")))
    # the two failures seen in practice: NumPy 2 with TensorFlow 2.16, and PyArrow built for NumPy 2
    for mod in ("numpy", "pandas", "pyarrow", "streamlit", "plotly", "sklearn", "joblib", "xgboost"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # any import-time failure is a real problem for the dashboard
            out.append(Check(G_DASH, f"import {mod}", "FAIL", f"{type(e).__name__}: {str(e)[:160]}",
                             "Reinstall the pinned versions: pip install -r requirements.txt"))
    if not fast:
        try:
            importlib.import_module("tensorflow")
            out.append(Check(G_DASH, "import tensorflow", "OK", md.version("tensorflow")))
        except Exception as e:
            out.append(Check(G_DASH, "import tensorflow", "FAIL", f"{type(e).__name__}: {str(e)[:160]}",
                             "Reinstall the pinned versions: pip install -r requirements.txt"))
    return out


def _demo_checks(root: Path, fast: bool) -> list:
    from src.app import artifacts
    from src.dataProcessing import forecast_data as fd

    out = []
    df, roads = None, None
    try:
        df = fd.load_raw()
        roads = fd.road_names(df)
        out.append(Check(G_DEMO, "dataset", "OK", f"{len(df)} rows, {len(roads)} roads, {df[fd.DATE_COL].min().date()} to {df[fd.DATE_COL].max().date()}"))
    except fd.DatasetError as e:
        out.append(Check(G_DEMO, "dataset", "FAIL", str(e).splitlines()[0][:230],
                         "Place the original Banglore_traffic_Dataset.csv in data/raw/ (Kaggle: preethamgouda/banglore-city-traffic-dataset)."))
    if df is not None:
        try:
            meta, _, _, _ = fd.load_dataset()
            out.append(Check(G_DEMO, "usable samples", "OK", f"{len(meta)} (need at least {fd.MIN_USABLE_SAMPLES})"))
        except fd.DatasetError as e:
            out.append(Check(G_DEMO, "usable samples", "FAIL", str(e)[:230]))

    gone = {name: hint for _, name, hint in artifacts.missing()}
    for label, (name, hint) in artifacts.REQUIRED.items():
        out.append(Check(G_DEMO, f"{label} ({name})", "FAIL" if name in gone else "OK",
                         "missing" if name in gone else "found", hint if name in gone else ""))

    # consistency between the files (only for files that exist)
    def cfg(name, label):
        try:
            return artifacts.read_json(name, label)
        except artifacts.ArtifactError as e:
            out.append(Check(G_DEMO, f"{name} readable", "FAIL", str(e)[:230]))
            return None

    xcfg = cfg("xgboost_features.json", "XGBoost feature configuration") if "xgboost_features.json" not in gone else None
    lcfg = cfg("lstm_features.json", "LSTM feature configuration") if "lstm_features.json" not in gone else None
    if xcfg is not None:
        out.append(Check(G_DEMO, "window length", "OK" if xcfg.get("seq_len") == fd.SEQ_LEN else "FAIL",
                         f"models: {xcfg.get('seq_len')}, code: {fd.SEQ_LEN}"))
        if roads is not None:
            same = sorted(xcfg.get("roads", [])) == sorted(roads)
            out.append(Check(G_DEMO, "roads match the dataset", "OK" if same else "FAIL",
                             f"{len(roads)} roads" if same else "the dataset and the trained models know different roads",
                             "" if same else "Use the original Banglore_traffic_Dataset.csv."))
            try:
                import numpy as np
                cols = list(fd.tabular_from_sequences(np.zeros((1, fd.SEQ_LEN, len(fd.seq_feature_names(roads)))), ["2024-01-02"], roads).columns)
                out.append(Check(G_DEMO, "XGBoost feature list matches the code", "OK" if cols == xcfg.get("xgboost_features") else "FAIL",
                                 f"{len(cols)} features" if cols == xcfg.get("xgboost_features") else "xgboost_features.json differs from the code's feature builder"))
            except Exception as e:  # pragma: no cover - defensive
                out.append(Check(G_DEMO, "XGBoost feature list matches the code", "FAIL", str(e)[:200]))
    if lcfg is not None and roads is not None:
        ok = lcfg.get("features") == fd.seq_feature_names(roads) and lcfg.get("seq_len") == fd.SEQ_LEN
        out.append(Check(G_DEMO, "LSTM feature list matches the code", "OK" if ok else "FAIL",
                         f"{len(lcfg.get('features', []))} features x {lcfg.get('seq_len')} steps" if ok else "lstm_features.json differs from the code's feature builder"))
    if lcfg is not None and "scaler.pkl" not in gone:
        try:
            import joblib
            sc = joblib.load(artifacts.MODEL_DIR / "scaler.pkl")
            ok = list(sc.feature_names_in_) == lcfg.get("features")
            out.append(Check(G_DEMO, "scaler matches LSTM features", "OK" if ok else "FAIL",
                             "same names and order" if ok else "scaler.pkl and lstm_features.json come from different trainings"))
        except Exception as e:
            out.append(Check(G_DEMO, "scaler.pkl loads", "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))
    if xcfg is not None and "xgboost_model.json" not in gone:
        try:
            from src.modelTraining import predictXgboost
            b = predictXgboost.load_booster()
            ok = list(b.feature_names or []) == xcfg.get("xgboost_features")
            out.append(Check(G_DEMO, "XGBoost model matches its feature list", "OK" if ok else "FAIL",
                             f"{b.num_features()} features, {b.num_boosted_rounds()} trees" if ok else "xgboost_model.json and xgboost_features.json come from different trainings"))
        except Exception as e:
            out.append(Check(G_DEMO, "xgboost_model.json loads", "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))

    if fast:
        out.append(Check(G_DEMO, "LSTM input shape + sample prediction", "SKIP", "skipped in --fast mode", "Run `python -m src.preflight` for the full check."))
        return out
    if any(c.status == "FAIL" for c in out):
        out.append(Check(G_DEMO, "LSTM input shape + sample prediction", "SKIP", "skipped because a required component above failed"))
        return out
    try:
        import numpy as np
        import pandas as pd
        from src.app import demo
        from src.modelTraining import predictLstm
        model, _ = predictLstm.preload()
        want = (None, fd.SEQ_LEN, len(fd.seq_feature_names(roads)))
        out.append(Check(G_DEMO, "LSTM input shape", "OK" if model.input_shape == want else "FAIL", f"{model.input_shape}"))
        ctx = demo.DemoContext()
        road = ctx.roads[0]
        t = ctx.replay_dates(road, test_only=True)[0]
        res = demo.predict_all(ctx.history(road, t), ctx)
        finite = all(np.isfinite(res[k]) and 0 <= res[k] <= 100 for k in ("XGBoost", "LSTM"))
        detail = f"{road}, forecast for {res['target_date'].date()}: XGBoost {res['XGBoost']:.1f}, LSTM {res['LSTM']:.1f}"
        status = "OK" if finite else "FAIL"
        px_path, pl_path = root / "model" / "xgboost_predictions.csv", root / "model" / "lstm_predictions.csv"
        if finite and px_path.is_file() and pl_path.is_file():
            px, pl = pd.read_csv(px_path), pd.read_csv(pl_path)
            row = (px["Date"] == str(res["target_date"].date())) & (px["Road/Intersection Name"] == road)
            if row.any():
                dx = abs(px.loc[row, "Predicted Congestion"].iloc[0] - res["XGBoost"])
                dl = abs(pl.loc[row, "Predicted Congestion"].iloc[0] - res["LSTM"])
                same = dx < 1e-3 and dl < 1e-2
                status = "OK" if same else "FAIL"
                detail += f" | reproduces the saved test prediction (diff XGB {dx:.1e}, LSTM {dl:.1e})" if same else \
                    f" | DOES NOT reproduce the saved prediction (diff XGB {dx:.2g}, LSTM {dl:.2g}): the models or the dataset changed"
        out.append(Check(G_DEMO, "sample prediction", status, detail))
    except Exception as e:
        out.append(Check(G_DEMO, "sample prediction", "FAIL", f"{type(e).__name__}: {str(e)[:220]}"))
    return out


def _results_checks(root: Path) -> list:
    import pandas as pd
    m = root / "model"
    out = []
    need = {"xgboost_predictions.csv": None, "lstm_predictions.csv": None}
    for name in need:
        p = m / name
        if not p.is_file():
            out.append(Check(G_RESULTS, name, "WARN", "missing", "The Model Results tab will explain what to run. Restore it from your backup or regenerate with the training scripts."))
            continue
        try:
            need[name] = pd.read_csv(p)
            cols = {"Date", "Road/Intersection Name", "Actual Congestion", "Predicted Congestion"}
            out.append(Check(G_RESULTS, name, "OK" if cols <= set(need[name].columns) else "WARN",
                             f"{len(need[name])} rows" if cols <= set(need[name].columns) else f"lacks columns {sorted(cols - set(need[name].columns))}"))
        except Exception as e:
            out.append(Check(G_RESULTS, name, "WARN", f"unreadable: {type(e).__name__}"))
    a, b = need["xgboost_predictions.csv"], need["lstm_predictions.csv"]
    if a is not None and b is not None:
        keys = ["Date", "Road/Intersection Name"]
        try:
            ok = len(a) == len(b) and (a[keys].values == b[keys].values).all() and abs(a["Actual Congestion"] - b["Actual Congestion"]).max() < 1e-6
        except Exception:
            ok = False
        out.append(Check(G_RESULTS, "both models on identical test rows", "OK" if ok else "WARN", "aligned" if ok else "the two files are not aligned"))
    for name in ("baseline_predictions.csv", "metrics.json"):
        out.append(Check(G_RESULTS, name, "OK" if (m / name).is_file() else "WARN", "found" if (m / name).is_file() else "missing (baselines / confidence intervals will not be shown)"))
    return out


def _java_major():
    exe = shutil.which("java")
    if not exe and os.environ.get("JAVA_HOME"):
        cand = Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe"
        exe = str(cand) if cand.is_file() else None
    if not exe:
        return None, None
    try:
        r = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20)
        text = (r.stderr or r.stdout).strip()
        m = re.search(r'version "(\d+)(?:\.(\d+))?', text)
        major = int(m.group(1)) if m else None
        if major == 1 and m and m.group(2):
            major = int(m.group(2))
        return exe, major
    except Exception:
        return exe, None


def _spark_checks(root: Path) -> list:
    out = []
    exe, major = _java_major()
    if exe is None:
        out.append(Check(G_SPARK, "Java", "OPT", "not found", "Only `python -m src.bigdata.spark_analysis` needs Java 17+ (tested with 25). The dashboard is unaffected."))
    else:
        out.append(Check(G_SPARK, "Java", "OK" if (major or 0) >= 17 else "OPT", f"{exe} (major version {major})",
                         "" if (major or 0) >= 17 else "PySpark 4 needs Java 17 or newer."))
    try:
        out.append(Check(G_SPARK, "pyspark", "OK", md.version("pyspark")))
    except md.PackageNotFoundError:
        out.append(Check(G_SPARK, "pyspark", "OPT", "not installed", "Only needed to re-run the Spark script: pip install -r requirements-bigdata.txt"))
    have = (root / "reports" / "spark" / "summary.json").is_file()
    out.append(Check(G_SPARK, "saved Spark results", "OK" if have else "OPT", "found (the EDA tab shows them)" if have else "not found (the EDA tab will say so)"))
    return out


def _tomtom_checks(root: Path) -> list:
    from src.live import credentials  # the same rules the dashboard uses; the value is never printed
    cred = credentials.load(project_root=root)
    if cred.configured:
        return [Check(G_TOMTOM, "TomTom API key", "OK", f"configured via the {cred.source} (value not shown). NOT verified against the live service: "
                                                          "run  python -m src.live.check_key --request  to test it once.")]
    return [Check(G_TOMTOM, "TomTom API key", "OPT", "not configured" + (": " + cred.hint_text() if cred.hint_text() else ""),
                  "Optional: without a key the map and place search work but there is no live traffic. See the README, 'Setting up TOMTOM_API_KEY'.")]


def run_checks(root=None, fast=False) -> list:
    root = Path(root) if root else ROOT
    checks = _dashboard_checks(root, fast)
    try:
        checks += _demo_checks(root, fast)
    except Exception as e:  # a bug in a check must never hide the other results
        checks.append(Check(G_DEMO, "preflight internal error", "FAIL", f"{type(e).__name__}: {str(e)[:200]}"))
    try:
        checks += _results_checks(root)
    except Exception as e:
        checks.append(Check(G_RESULTS, "preflight internal error", "WARN", f"{type(e).__name__}: {str(e)[:200]}"))
    return checks + _spark_checks(root) + _tomtom_checks(root)


def summarize(checks: list) -> dict:
    def ok(group):
        return not any(c.group == group and c.status == "FAIL" for c in checks)
    return {"dashboard_can_launch": ok(G_DASH), "live_demo_available": ok(G_DASH) and ok(G_DEMO),
            "results_tab_complete": not any(c.group == G_RESULTS and c.status != "OK" for c in checks),
            "spark_available": all(c.status == "OK" for c in checks if c.group == G_SPARK and c.name in ("Java", "pyspark")),
            "tomtom_configured": any(c.group == G_TOMTOM and c.status == "OK" for c in checks)}


_TAG = {"OK": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]", "OPT": "[ -- ]"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fast", action="store_true", help="skip TensorFlow import and the sample prediction")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--quiet", action="store_true", help="print only problems and the summary")
    a = ap.parse_args(argv)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    checks = run_checks(fast=a.fast)
    s = summarize(checks)
    if a.json:
        print(json.dumps({"checks": [asdict(c) for c in checks], "summary": s}, indent=2))
    else:
        group = None
        for c in checks:
            if a.quiet and c.status in ("OK", "SKIP", "OPT"):
                continue
            if c.group != group:
                group = c.group
                print(f"\n{group}")
            print(f"  {_TAG[c.status]} {c.name}" + (f": {c.detail}" if c.detail else ""))
            if c.hint and c.status in ("FAIL", "WARN", "OPT"):
                print(f"         -> {c.hint}")
        yn = lambda v: "YES" if v else "NO"
        print("\nSummary")
        print(f"  Dashboard can launch ........... {yn(s['dashboard_can_launch'])}")
        print(f"  Live prediction demo available . {yn(s['live_demo_available'])}" + ("" if s["live_demo_available"] else "   (fix the [FAIL] items above)"))
        print(f"  Model Results tab complete ..... {yn(s['results_tab_complete'])}")
        print(f"  Optional PySpark/Java ready .... {yn(s['spark_available'])}   (not needed for the dashboard)")
        print(f"  Optional TomTom key configured . {yn(s['tomtom_configured'])}   (not needed; real-time mode cannot predict)")
    return 0 if s["dashboard_can_launch"] and s["live_demo_available"] else 1


if __name__ == "__main__":
    sys.exit(main())

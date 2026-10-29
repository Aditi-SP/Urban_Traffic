"""
Failure-scenario matrix: breaks the project in many ways (on scratch COPIES; your files are never touched) and checks that the
dashboard, the preflight check and the Spark script report each problem clearly instead of crashing.

    python tests/failure_matrix.py              run every scenario (about 5-6 minutes)
    python tests/failure_matrix.py lstm_features tiny     run only scenarios whose name contains these words

Exit code 0 only if every scenario behaved as expected.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = Path("data/raw/Banglore_traffic_Dataset.csv")
CASE = Path(__file__).with_name("_app_case.py")


# ----------------------------------------------------------------------------------- scratch copies
def make_copy(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "app.py", dst)
    for f in ("requirements.txt", "requirements-bigdata.txt"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, dst)
    for d in ("src", "config", "reports"):
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "model", dst / "model", ignore=shutil.ignore_patterns("original_backup"))
    (dst / ".streamlit").mkdir(exist_ok=True)
    shutil.copy(ROOT / ".streamlit" / "config.toml", dst / ".streamlit")
    (dst / CSV.parent).mkdir(parents=True)
    shutil.copy(ROOT / CSV, dst / CSV)
    return dst


def _rm(*rels):
    def f(d):
        for r in rels:
            t = d / r
            shutil.rmtree(t) if t.is_dir() else t.unlink()
    return f


def _df(fn):
    def f(d):
        import pandas as pd
        df = pd.read_csv(d / CSV)
        fn(df).to_csv(d / CSV, index=False)
    return f


def _text(rel, content):
    def f(d):
        (d / rel).write_text(content, encoding="utf-8")
    return f


def _bytes(rel, content):
    def f(d):
        (d / rel).write_bytes(content)
    return f


def _set(col, rows, value):
    def g(df):
        df[col] = df[col].astype(object)
        df.loc[df.index[rows], col] = value
        return df
    return _df(g)


def _alter_xgb_features(d):
    p = d / "model" / "xgboost_features.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["xgboost_features"] = cfg["xgboost_features"][::-1]  # same names, wrong order: the model must refuse it
    p.write_text(json.dumps(cfg), encoding="utf-8")


def _other_roads(df):
    df["Road/Intersection Name"] = df["Road/Intersection Name"] + " (other)"
    return df


# name, mutate, banner substring (None = no setup banner expected), error-text substring (anywhere, optional),
# predict outcome ("forecast" | "unavailable" | "error:<substring>"), preflight expects live demo available?
SCENARIOS = [
    ("baseline (nothing broken)", lambda d: None, None, None, "forecast", True),
    ("dataset CSV missing", _rm(CSV), "Dataset not found", None, "unavailable", False),
    ("dataset empty file", _bytes(CSV, b""), "empty", None, "unavailable", False),
    ("dataset header only", _text(CSV, "Date,Area Name,Road/Intersection Name,Traffic Volume\n"), "no rows", None, "unavailable", False),
    ("dataset malformed binary", _bytes(CSV, bytes(range(256)) * 20), "Dataset:", None, "unavailable", False),
    ("dataset missing column", _df(lambda df: df.drop(columns=["Traffic Volume"])), "missing required columns", None, "unavailable", False),
    ("dataset non-numeric value", _set("Traffic Volume", [0, 1], "abc"), "non-numeric", None, "unavailable", False),
    ("dataset bad date", _set("Date", [0], "not-a-date"), "not dates", None, "unavailable", False),
    ("dataset missing values", _set("Traffic Volume", [0, 1, 2], None), "missing values", None, "unavailable", False),
    ("dataset duplicate rows", _df(lambda df: __import__("pandas").concat([df, df.head(3)])), "duplicate", None, "unavailable", False),
    ("dataset unknown weather", _set("Weather Conditions", [0], "Snow"), "Unknown weather", None, "unavailable", False),
    ("dataset congestion out of range", _set("Congestion Level", [0], 250), "within 0-100", None, "unavailable", False),
    ("dataset tiny (50 rows)", _df(lambda df: df.head(50)), "No usable samples", None, "unavailable", False),
    ("dataset 5 rows", _df(lambda df: df.head(5)), "No usable samples", None, "unavailable", False),
    ("dataset incompatible (other roads)", _df(_other_roads), None, "not compatible", "unavailable", False),
    ("model.h5 missing", _rm("model/model.h5"), "model.h5", None, "unavailable", False),
    ("scaler.pkl missing", _rm("model/scaler.pkl"), "scaler.pkl", None, "unavailable", False),
    ("xgboost_model.json missing", _rm("model/xgboost_model.json"), "xgboost_model.json", None, "unavailable", False),
    ("xgboost_features.json missing", _rm("model/xgboost_features.json"), "xgboost_features.json", None, "unavailable", False),
    ("lstm_features.json missing", _rm("model/lstm_features.json"), "lstm_features.json", None, "unavailable", False),
    ("both feature JSON files missing", _rm("model/lstm_features.json", "model/xgboost_features.json"), "lstm_features.json", None, "unavailable", False),
    ("xgboost_features.json corrupt", _text("model/xgboost_features.json", "{not json"), None, "not valid JSON", "unavailable", False),
    ("xgboost feature order tampered", _alter_xgb_features, None, None, "error:disagree", False),
    ("prediction CSVs missing", _rm("model/xgboost_predictions.csv", "model/lstm_predictions.csv"), None, None, "forecast", True),
    ("metrics.json missing", _rm("model/metrics.json"), None, None, "forecast", True),
    ("Spark outputs missing", _rm("reports/spark"), None, None, "forecast", True),
]


# ----------------------------------------------------------------------------------- runners
def run_app(copy: Path) -> dict:
    r = subprocess.run([sys.executable, str(CASE), str(copy)], capture_output=True, text=True, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
    if line is None:
        return {"exceptions": ["child process failed: " + (r.stderr or r.stdout)[-300:]], "errors": [], "warnings": [], "infos": [], "predict": "?"}
    return json.loads(line[len("RESULT_JSON "):])


def run_preflight(copy: Path) -> dict:
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    r = subprocess.run([sys.executable, "-m", "src.preflight", "--fast", "--json"], cwd=copy, capture_output=True, text=True, env=env, timeout=600)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"exit": r.returncode, "summary": None, "error": (r.stderr or r.stdout)[-300:]}
    return {"exit": r.returncode, "summary": data["summary"]}


def check_scenario(name, banner, err_sub, predict, pre_demo, res, pre):
    """Returns a list of problems (empty = the scenario behaved as expected)."""
    bad = []
    if res["exceptions"]:
        bad.append(f"uncaught exception on page load: {res['exceptions'][0][:120]}")
    if res.get("perf_exceptions"):
        bad.append(f"uncaught exception on the Performance page: {res['perf_exceptions'][0][:120]}")
    if res.get("shell_exceptions"):
        bad.append(f"uncaught exception on a shell page: {res['shell_exceptions'][0][:120]}")
    if banner is not None and sum("Setup incomplete" in e for e in res.get("perf_errors", [])) != 1:
        bad.append("the Performance page should show the setup banner exactly once")
    if res.get("predict_exceptions"):
        bad.append(f"uncaught exception after Predict: {res['predict_exceptions'][0][:120]}")
    errors = " | ".join(res["errors"])
    if banner is None and any("Setup incomplete" in e for e in res["errors"]):
        bad.append("unexpected setup banner")
    if banner is not None:
        banners = [e for e in res["errors"] if "Setup incomplete" in e]
        if len(banners) != 1:
            bad.append(f"expected exactly one setup banner, found {len(banners)}")
        elif banner.lower() not in banners[0].lower():
            bad.append(f"banner lacks {banner!r}")
        if sum(1 for e in res["errors"] if banner.lower() in e.lower()) > 1:
            bad.append("the same problem is repeated in several messages")
    if banner is None and err_sub is None and predict == "forecast" and res["errors"]:
        bad.append(f"unexpected st.error in a healthy scenario: {res['errors'][0][:100]}")
    if err_sub and err_sub.lower() not in (errors + " " + " | ".join(res.get("predict_errors", []))).lower():
        bad.append(f"no message containing {err_sub!r}")
    if predict == "forecast" and res["predict"] != "forecast":
        bad.append(f"Predict should work but gave {res['predict']!r}")
    if predict == "unavailable" and res["predict"] == "forecast":
        bad.append("Predict produced a forecast although it should be disabled")
    if predict.startswith("error:"):
        sub = predict.split(":", 1)[1]
        if sub not in " | ".join(res.get("predict_errors", [])).lower():
            bad.append(f"Predict should show an error containing {sub!r}")
    if name.startswith("metrics.json") and any("no winner is declared" in i for i in res["infos"]):
        bad.append("contradictory 'no winner is declared' message")
    if pre.get("summary") is None:
        bad.append("preflight produced no JSON: " + pre.get("error", ""))
    elif pre["summary"]["live_demo_available"] != pre_demo and not name.startswith(("xgboost feature order", "dataset incompatible")):
        bad.append(f"preflight says live_demo_available={pre['summary']['live_demo_available']}, expected {pre_demo}")
    elif pre["summary"]["dashboard_can_launch"] is not True:
        bad.append("preflight claims the dashboard cannot launch")
    return bad


def spark_without_java() -> list:
    """The Spark script must explain a missing Java instead of printing a Py4J traceback; the dashboard is unaffected."""
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if "java" not in p.lower() and "oracle" not in p.lower())
    env = {k: v for k, v in os.environ.items() if k != "JAVA_HOME"}
    env["PATH"] = path
    r = subprocess.run([sys.executable, "-m", "src.bigdata.spark_analysis"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    bad = []
    if r.returncode != 2:
        bad.append(f"exit code {r.returncode}, expected 2")
    if "Traceback" in r.stderr or "Py4J" in r.stderr:
        bad.append("traceback printed instead of a clear message")
    if "Java" not in r.stderr:
        bad.append("message does not mention Java")
    return bad


def main(argv):
    words = [a.lower() for a in argv]
    tmp = Path(tempfile.mkdtemp(prefix="tcf_matrix_"))
    failures = 0
    try:
        for i, (name, mutate, banner, err_sub, predict, pre_demo) in enumerate(SCENARIOS):
            if words and not any(w in name.lower() for w in words):
                continue
            copy = make_copy(tmp / f"c{i}")
            mutate(copy)
            res, pre = run_app(copy), run_preflight(copy)
            bad = check_scenario(name, banner, err_sub, predict, pre_demo, res, pre)
            failures += bool(bad)
            first_err = (res["errors"][0] if res["errors"] else "-")[:110]
            print(f"[{'PASS' if not bad else 'FAIL'}] {name:38s} predict={res['predict']:11s} preflight_demo={pre.get('summary', {}) and pre['summary']['live_demo_available']!s:5s} | {first_err}")
            for b in bad:
                print(f"        ! {b}")
            shutil.rmtree(copy, ignore_errors=True)
        if not words or any(w in "spark java" for w in words):
            bad = spark_without_java()
            failures += bool(bad)
            print(f"[{'PASS' if not bad else 'FAIL'}] {'Spark script without Java on PATH':38s}")
            for b in bad:
                print(f"        ! {b}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL SCENARIOS BEHAVED AS EXPECTED' if not failures else str(failures) + ' scenario(s) need attention'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

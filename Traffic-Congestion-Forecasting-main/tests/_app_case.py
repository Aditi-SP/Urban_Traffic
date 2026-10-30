"""
Child process used by the failure tests: runs the dashboard (Streamlit AppTest) from ONE project folder and prints
what a user would see as a single JSON line. It must run in its own process because the project's modules are
imported from that folder.

    python tests/_app_case.py <project folder>
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("COMMUNITY_DB_PATH", os.path.join(os.environ.get("TEMP", "."), "tc_failure_case_community.db"))  # never the real community.db
root = sys.argv[1]
os.chdir(root)
sys.path.insert(0, root)
from streamlit.testing.v1 import AppTest  # noqa: E402

short = lambda s, n=400: " ".join(str(s).split())[:n]
PAGES = os.path.join(root, "src", "ui", "pages")
at = AppTest.from_file(os.path.join(PAGES, "forecasts.py"), default_timeout=300)  # banner + the historical demo (Predict)
t0 = time.perf_counter()
at.run()
out = {
    "seconds": round(time.perf_counter() - t0, 1),
    "exceptions": [short(e.value) for e in at.exception],
    "errors": [short(e.value) for e in at.error],
    "warnings": [short(w.value) for w in at.warning],
    "infos": [short(i.value) for i in at.info],
}
try:
    at.button(key="demo_predict").click().run()
    out["predict"] = "forecast" if "demo_result" in at.session_state else "no forecast"
    out["predict_errors"] = [short(e.value) for e in at.error]
    out["predict_exceptions"] = [short(e.value) for e in at.exception]
except KeyError:  # no Predict button on the page (the demo is disabled)
    out["predict"] = "unavailable"

# the Performance page (model results, data, historical analysis): its own banner and messages
perf = AppTest.from_file(os.path.join(PAGES, "performance.py"), default_timeout=300).run()
out["perf_exceptions"] = [short(e.value) for e in perf.exception]
out["perf_errors"] = [short(e.value) for e in perf.error]
out["warnings"] += [short(w.value) for w in perf.warning]
out["infos"] += [short(i.value) for i in perf.info]

# the three shell/detail pages must never crash, whatever is broken
out["shell_exceptions"] = []
for page in ("live_map", "community", "details"):
    a = AppTest.from_file(os.path.join(PAGES, page + ".py"), default_timeout=300).run()
    out["shell_exceptions"] += [f"{page}: {short(e.value)}" for e in a.exception]
print("RESULT_JSON " + json.dumps(out))

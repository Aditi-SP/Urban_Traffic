# Next-steps audit: Traffic Congestion Forecasting

Date: 2026-10-29 (audit run 20:35 to 21:45 local). Scope: read-only. **No source, test, model, data or config file was changed**; this file is the only addition.
Paths are relative to the project folder (the one that contains `app.py`).

External calls made by the audit (all real, none mocked): one TomTom Traffic Flow request with the saved key via the project's own client, one more
through the dashboard (the real Live Traffic Map fetch), one request with an obviously **fake** key to see TomTom's real error answer, one OpenStreetMap
Nominatim search, and OpenStreetMap map tiles loaded by the test browser. The key was never printed; it appears in **0 of 4,018** project and log files,
in none of the page HTML scanned, and the audit dashboard's server log has no traceback or error.

## 1. Verdict

| Area | Status | Evidence |
|---|---|---|
| Live traffic, real TomTom response | **Working** | 5 km/h vs 18 km/h free-flow (28%, confidence 1.00, FRC1, 15-point segment, 1.55 s) via `src/live/tomtom.py`; 6 vs 21 km/h (29%) drawn as a red segment in the dashboard; your own 7 vs 21 reading |
| Live Traffic Monitor scope | **Partial**: on-demand reading for one chosen point | no auto-refresh, watchlist, history or incidents (`src/ui/live_map.py`, `src/live/*.py` contain none) |
| Place search, map tiles | Working | real Nominatim result for "silk board junction"; 9 to 12 tiles rendered |
| Traffic Forecasts | Working on **historical data only** | see section 3 |
| Model Insights | Working | headline RMSE equal the saved evaluation; provenance and metric explanations present |
| Community Reports | Working (local, anonymous, unverified) | scratch-database persistence verified; your real database has 1 report |
| System status, Diagnostics | Working (D4 and D5 fixed in Tier 2 Phase 1, 2026-10-30) | D4, D5 below |
| Road & Incident Details | Historical road profile only; no live data, **incidents not implemented** (D7 wording fixed in Tier 2 Phase 1) | `src/ui/details.py`: the page now says what it implements, points to the Live Traffic Map for live values, and states that no incident source is connected |
| Newer historical data / 2026 forecasts | **Blocked** | section 3 |
| Model skill | **None beyond the per-road mean** | section 4 |

## 2. How the key is loaded when started with `run_dashboard.ps1`

`run_dashboard.ps1` (`Import-MissingEnvironment`) copies `TOMTOM_API_KEY`, `LIVE_CACHE_TTL_S`, `LIVE_MAX_REQUESTS` from the saved Windows **User** (then Machine)
environment into the launcher's process **only if the terminal lacks them**; it never prints the key. The dashboard inherits that process environment, then:
`src/live/credentials.py: load()` (env var, then project and home `secrets.toml`) -> `src/ui/common.py: tomtom_key()` -> `src/ui/live_map.py: request_reading()`
-> `src/live/tomtom.py: SERVICE.get()` (the one shared client; the only file with the TomTom URL).

* **Verified:** the launcher started from a terminal with neither variable loaded both ("TomTom key: loaded from your Windows user environment", "LIVE_CACHE_TTL_S = 0"),
  and that dashboard's Diagnostics said `configured (source: environment variable)`, TTL 0.
* **Not covered:** a bare `streamlit run app.py` from a stale terminal (for example VS Code) does **not** load saved values. **The dashboard on port 8599
  (started 19:04 from a VS Code terminal) has no key and is running with `LIVE_CACHE_TTL_S=300`, not your saved `0`.** The one on port 8501 (started 20:32) is correct
  (key present, TTL 0, "Live traffic: last response OK 20:37:12").

## 3. Data and forecast coverage

* `data/raw/Banglore_traffic_Dataset.csv`: 8,936 rows, 16 roads, 2022-01-01 to **2024-08-09** (sha256 prefix 2d7636e53b64, same as the Stage 5A run). **Zero rows after 2024-08-09**;
  no CSV in `model/`, `reports/` or `data/` has a later date. The gap to 2026-10-28 is 810 days.
* Real code path (`src/app/demo.py: DemoContext.replay_dates`): the latest start day is 2024-08-09, so the latest target is **2024-08-10**, a day with no observation. Nothing later can be selected or forecast.
  There is no hard-coded year; the limit is the data. The UI states this (`src/ui/forecasts_sections.py: render_data_status`).
* Real newer observations do not exist in the project and none are invented. `python -m src.dataProcessing.check_new_data <file.csv>` is the read-only validator for any you obtain.

## 4. Model and metrics (saved evaluation, recomputed independently from `model/*_predictions.csv`; all 4 rows match to 1e-6)

Target: congestion (0 to 100) of the same road on the next day; chronological split, test 2024-03-21 to 2024-08-09, n = 856, rows equal real dataset targets.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LSTM | 15.463 | 19.933 | 0.200 |
| XGBoost | 15.931 | 20.074 | 0.189 |
| Road-mean baseline | 15.338 | 19.846 | 0.207 |
| Persistence | 19.105 | 28.232 | -0.605 |

Bootstrap RMSE differences vs road mean: LSTM +0.09 [-0.04, +0.21], XGBoost +0.23 [-0.10, +0.56]; both include 0. Stage 5A (`experiments/stage5a/signal_check.json`): `NO_USABLE_SIGNAL`,
within-road lag-1 r = 0.003 [-0.021, 0.028]. Artifact hashes equal the evaluation report; 3 model hashes and all 48 protected-manifest entries verified. Dataset provenance and licence are **unverified**;
43.5% of targets are exactly 100. "Real data" here means the Kaggle file as supplied, not independently validated traffic.

## 5. Tests actually run (current code, `LIVE_CACHE_TTL_S=0`)

| Check | Result |
|---|---|
| Core suite (`unittest discover -s tests`) | 366 run: **365 passed, 1 failed**, 0 skipped. Failure = D3 (stale test, not an app defect) |
| `docs_tests` | 11 passed |
| `pip check`, `compileall`, launcher `-CheckOnly` | clean / exit 0 / exit 0 ("Optional TomTom key configured: YES") |
| Browser checks (dashboard started without a key, as the checks assume) | 1536x700: 83/83. 1280x650: 83/83. **800x900: 82/83** (D2) |
| Failure matrix | run 1 **aborted** after 5 scenarios (`subprocess.TimeoutExpired` in the scenario-6 preflight, reported 361 s past its 600 s limit; the same preflight takes ~13 s alone). Run 2: **27/27**, empty stderr, 20 min 45 s |
| Real-browser audit journey (real key, real Nominatim and TomTom) | 20/23: 1 real defect (D4), 2 were my script's case/scope errors, re-verified correct |
| Community persistence (scratch DB) | save, same-token idempotence, 3 invalid inputs refused, reopen count 1, corrupt file -> `storage` error without path |
| Real error answer (fake key) | HTTP 401 `{"detailedError":{"code":"Unauthorized",...}}` -> classified `auth`, message correct |

## 6. Defects and gaps

| ID | Sev | Finding and evidence | Where |
|---|---|---|---|
| D1 | High (ops) | Dashboard on 8599 runs without the key and with TTL 300; your TTL=0 safeguard is **not in force** there. Diagnostics already warns | restart it via `run_dashboard.ps1` |
| D2 | Med | **Regression:** when a key is saved in Windows but the process lacks it, the "Why it was not found" text makes the banner **528 px** tall at 800 px width; the map top is at y=1058 in a 900 px window (check "map above the fold" fails) | `src/ui/live_map.py` (banner in `render`), `src/live/credentials.py: Credentials.hint_text` |
| D3 | Med | `test_stage0_creates_no_database_and_no_upload_folder` globs for any `*.db` in the project; it fails because your real `data/community/community.db` (created 20:36:43, 1 report) is legitimate | `tests/test_ui_shell.py` (`ShellPagesTests`) |
| D4 | Low-Med | Sidebar status chips lag one interaction: after "Get traffic" they still say "key found, not yet tested"; a fresh load says "last response OK" | `app.py` (sidebar built before `page.run()`) **Fixed in Tier 2 Phase 1 (2026-10-30).** |
| D5 | Low | Diagnostics "process started at" is the **module import time**, wrong for a long-running process (8599 showed 20:42 though started 19:04) | `src/live/credentials.py:33`, `src/ui/diagnostics.py` **Fixed in Tier 2 Phase 1 (2026-10-30).** |
| D6 | Low-Med | `tests/browser_check.py` assumes no key exists; with the saved key a launcher-started dashboard fails its landing-page checks | `tests/browser_check.py` **Fixed in Tier 2 Phase 1 (2026-10-30).** |
| D7 | Med | Road & Incident Details is a shell; its text says the provider "is not configured" (now false) | `src/ui/details.py:27` **Fixed in Tier 2 Phase 1 (2026-10-30).** |
| D8 | Low | Failure matrix has a fixed 600 s limit per preflight, hung once, and now takes ~21 min | `tests/failure_matrix.py:131` |
| D9 | Low | Stale or legacy files: `DEMO_SCRIPT.md` (old tab layout), `AUDIT_REPORT.md`, unreferenced `src/app/streamlit_app.py`, `requirements.txt.bak` | project root, `src/app/` |
| D10 | Med (risk) | 29 uncommitted entries (16 tracked files changed, +899/-230 since commit 020d6915); the work has no checkpoint | git (your decision) |
| D11 | Product | Live monitor = one point on demand; no pinned places, auto-refresh, history or incidents | `src/ui/live_map.py` |
| D12 | Unverified | Only a real **401** was seen. How TomTom words a 403 (no Traffic API access vs quota) is unknown; `src/live/tomtom.py: _denied` classifies by keywords and falls back to "unclear" | `src/live/tomtom.py` |
| D13 | Unresolved | TomTom caching/storage/ML terms unread (TTL 0 and no storage stay); only a per-run request cap exists, no monthly quota tracking | `src/live/tomtom.py` |

## 7. Prioritized repair plan (nothing implemented yet)

**P0 (do first, small, each with a test)**
1. D1: stop the 8599 dashboard and start with `.\run_dashboard.ps1` (your action). Optionally make Diagnostics show a red "TTL safeguard NOT in force" line when TTL != saved value.
2. D3: make the test hermetic: run the pages with a temporary `COMMUNITY_DB_PATH` and assert viewing creates no file there; do not glob the real folder.
3. D2: shorten the banner to one sentence plus "details in Diagnostics"; keep the full reasons in the Diagnostics panel. Done when `browser_check.py` passes at 800x900 with a Windows-saved key and no process key.

**P1**
4. D4: build the sidebar after `page.run()` (or rerun after a fetch); test that the chip says "last response OK" right after a fetch.
5. D5: use the real process creation time for "started at".
6. D6: make `browser_check.py` key-agnostic (assert on whichever state Diagnostics reports) or start its server hermetically.
7. D7 and D11 (decision first): decide whether Road & Incident Details becomes the verified live reading for the chosen point (current vs free-flow, travel time) and whether to add pinned places and a limited manual refresh. Incidents need the TomTom Incidents API and its terms. Fix the stale wording regardless.

**P2**
8. D12: capture TomTom's real 403 wording if it ever occurs naturally (do not provoke it) and tighten `_denied`.
9. D13: read TomTom's developer terms; only then consider any caching or storage. Optionally show remaining monthly quota from your TomTom account (a manual note), not a guess.
10. D8, D9: raise or scale the matrix timeouts; refresh or retire stale docs and the legacy files.

**P3 (decisions for you)**
11. D10: commit the work in reviewable pieces when you are ready (I did not and will not without being asked).
12. Forecasting future dates needs real newer observations plus retraining, re-evaluation and approval. Given Stage 5A, consider presenting the per-road historical mean as the reference and putting effort into the live monitor instead.

## 8. Not verified here
A 403 answer from TomTom; your key's quota and plan; TomTom's terms; coverage of roads other than the two points tried; whether any real newer data source exists; a second real traffic request through Mode 2 on the Forecasts page (covered by mocked tests only).

## 9. Final hardening (branch `final-hardening`, 2026-10-30)

Status of the limitations that remained after the release `251f86f4`. Only what was measured or read is claimed.

| Area | Status | What was done / why it remains open |
|---|---|---|
| D12 / TomTom 401, 403, 400, 404 handling | **Fixed where observable, documented where not** | Real responses were recorded: no-road is HTTP 400 "Point too far from nearest existing segment."; a wrong or missing key is HTTP 401; other 400s and an unknown service version (404) are not "no segment" (they were wrongly mapped before). A 403 quotes TomTom's documented meaning; its causes stay hedged because a real 403 body has never been seen. New kind `unexpected_status` for anything not understood. Tests mark each body as recorded, constructed or unverified. |
| D13 / TomTom terms, quota, attribution | **Partly resolved, blocker documented** | Official documentation read and cited with the date (status codes, error format, Cache-Control, the pricing page's "Free 20K monthly" for this API, no per-second limit stated). The **binding developer terms could not be read** with the tools available, so caching, storage, ML-use and attribution rules remain unresolved. Nothing is stored; `LIVE_CACHE_TTL_S=0` and the safeguards stay. Needs a human to read TomTom's terms or ask TomTom. |
| Live traffic coverage | **Audited, no coverage claim** | 16 of 16 main-road points and 4 of 4 scope-edge points returned a valid segment. TomTom answers for the nearest road, so the page now says when that road is more than 150 m from the chosen point. Nothing stored. A 20-point probe is not city-wide coverage. |
| Model performance | **Audited, no defect found, nothing retrained** | `tests/test_evaluation_audit.py` re-verifies identical test rows, recomputed metrics, baselines, chronological split, train-only scaler. Result unchanged: the road mean is at least as good as both learned models; no model is claimed to win. Extra data needed is documented (real, newer, finer than daily, usable licence). |
| Historical data | **Blocker documented** | Sources searched; none legitimate and usable was found (see README). The data-coverage warnings and the read-only validator already existed and are unchanged. Needs real newer observations you may legally use. |
| Fresh installation | **Improved and tested** | README: Python 3.11 checklist, dataset placement, behaviour without the dataset, the git-ignored `doucments/` folder is no longer assumed, actionable failure table. The suite is now hermetic against a developer's saved TomTom key (a test failed on its own when a key was saved). |
| Loading state | **Implemented and tested** | Specific spinners for the search, traffic request, prediction, model loading, dataset loading and the two Diagnostics checks; the sidebar says its status is from the previous action while a run is busy and the line is always cleared. Streamlit shows a spinner only after about 0.5 s, so a fast operation shows none. |
| D8 failure-matrix timeouts | **Unchanged** | Passed 27/27 (about 14 minutes); its fixed per-step limits were not changed. |
| D10 uncommitted work | **Resolved** | Committed and pushed in `251f86f4` and earlier. |

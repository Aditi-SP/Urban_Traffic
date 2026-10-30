# 🚦 Traffic Congestion Forecasting: LSTM vs XGBoost

Forecasts **next-day congestion level per road** in Bangalore with an LSTM and an XGBoost model,
compares them with simple baselines on the same held-out data, and shows the results in a Streamlit dashboard.

> **Status: experimental.** On the held-out test set neither model beat the simple per-road historical-mean baseline
> (see "Results"), so this evaluation does **not** show that either model predicts day-to-day changes in congestion. How
> they perform on recent real-world conditions has **not** been established: the dataset ends on 2024-08-09. The
> dashboard's forecasts are historical demonstrations, not live traffic and not validated forecasts.

## What works today, and what does not

| Area | Status |
|---|---|
| Dashboard, navigation, historical replay and what-if forecasts | Working, on **historical data only** (see "Forecast data coverage") |
| Live Traffic Map: basemap and place search | Working (OpenStreetMap tiles and Nominatim search; both need an internet connection) |
| Live traffic readings (TomTom) | **Needs your own TomTom key. No genuine TomTom response has been verified**, so live traffic is not claimed to work |
| Forecasts for recent dates (up to October 2026) | **Not possible:** the dataset ends on 2024-08-09 and no newer observations are available. None are invented |
| Community Reports | Working: local, anonymous, unverified text reports |
| System status and Diagnostics | Real checks (see "System status and Diagnostics") |
| Model quality | Experimental: neither model beat the per-road historical mean; not validated for 2026 |

## Dataset

* File: `data/raw/Banglore_traffic_Dataset.csv` (not committed; `*.csv` is git-ignored except `model/*.csv`).
* Source: Kaggle, *Banglore City Traffic Dataset* (`preethamgouda/banglore-city-traffic-dataset`). Third-party write-ups
  say it derives from Government of Karnataka open traffic data, but this was **not verified** from an authoritative
  source, and the **licence is unknown** (the Kaggle page could not be read). Check it before redistributing.
* Identity check: the downloaded file's min/max for all six numeric features equal those stored in the project's
  original `scaler.pkl` (`model/original_backup/`), which is strong evidence it is the dataset the project was built on.
* 8,936 rows, 16 roads in 8 areas, 2022-01-01 to 2024-08-09. It is **daily** data (no hourly information) and
  each road is observed on only 275–860 of the ~950 days, so there are gaps.

## Forecasting task (identical for both models)

| Item | Definition |
|---|---|
| Target | `Congestion Level` (0-100) of the same road on day **t+1** |
| Prediction time | End of day t. Only data from day t and earlier is used, plus the weekday/month of the target day |
| Inputs | The last 7 observations of that road (congestion, speed, volume, travel-time index, capacity use, incidents, weather, gap in days since the previous observation, road identity) |
| Samples | A (road, day t) pair is used only if the road was also observed on day t+1 and has 7 observations of history. Windows never cross roads (5,754 samples) |
| Split | Chronological by target date: 70% train (to 2023-10-30), 15% validation (to 2024-03-20), 15% test (2024-03-21 to 2024-08-09). Nothing is shuffled |
| Scaling | The LSTM's MinMax scaler is fitted on training windows only (XGBoost needs none) |
| Early stopping | Uses the validation split; the test split is only used for the final evaluation |
| Test set | Both models and both baselines are scored on exactly the same 856 test samples |

Baselines: **Persistence** (tomorrow = today) and **Road mean** (each road's training-period mean).

## Results (test set, 856 samples; produced by `comparePerformancePlots.py`)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LSTM | 15.463 | 19.933 | 0.200 |
| XGBoost | 15.931 | 20.074 | 0.189 |
| Persistence baseline | 19.105 | 28.232 | -0.605 |
| Road-mean baseline (each road's training-period average) | 15.338 | 19.846 | 0.207 |

MAE and RMSE are in congestion-level points (0-100 scale; lower is better); R² is higher-is-better. The dashboard shows the
same four rows beside each other (baseline first) on the **Traffic Forecasts** and **Model Insights** pages, with the RMSE
difference to the road-mean baseline. Those figures are read from `model/metrics.json`; they are not recomputed.

**Interpretation:** both models beat persistence but do **not** beat the road-mean baseline, i.e. they capture which
roads are congested on average but essentially no day-to-day signal.

*A note on the "lag-1 correlation of about 0.2".* Earlier versions of the saved evaluation report quote a day-to-day (lag-1)
correlation of about 0.21 as showing that next-day congestion is "weakly predictable". That figure is **pooled across roads**,
so it is largely driven by persistent differences between roads (some roads are almost always congested, others rarely), not
by within-road predictability. Once each road's own level is removed, the within-road lag-1 correlation is **approximately 0**
(0.003 on all data, 95% CI -0.02 to +0.03, in the Stage 5A benchmark below; an earlier exploratory calculation gave 0.002),
and no weekday, month-of-year or shared-date effect was found either. Congestion on a given day is mostly explained by
same-day measurements (volume, capacity use) that are not available the day before. The saved `model/evaluation_report.*`
still contains the pooled figure; it was not regenerated.

Earlier versions of this project reported XGBoost R² = 0.9995; that came from target
leakage (features computed from the target itself) and is not a valid forecasting result.
Re-run `python -m src.modelTraining.comparePerformancePlots` to regenerate these numbers
(`model/saved_metrics.txt`, `model/metrics.json`).

## Quick start (Windows)

**New here? Three steps** (each is explained in the section named):

1. **Install the dependencies once**: "Setup from scratch" below (creates `.venv` and installs `requirements.txt`). You also need the dataset and the model files listed there.
2. **Optional, for live traffic: configure TomTom.** Without a key everything else works and the map shows no traffic. See "Setting up `TOMTOM_API_KEY`" (a TomTom developer key; the commands hide what you type and never show the key).
3. **Run the dashboard**: `.\run_dashboard.ps1` from the project folder (below). The sidebar **System status** and **Diagnostics** then show what works on this computer.

The project's virtual environment sits next to this folder (`..\.venv`). The launchers call it directly, so **you do not
need to activate anything**.

```powershell
cd "<the folder that contains app.py>"          # for example: open the project folder in File Explorer, then "Open in Terminal"
.\run_dashboard.ps1                 # quick preflight, then opens the dashboard in your browser
```

Or double-click `run_dashboard.bat` (it runs the same script and bypasses the PowerShell execution policy for that one command only).

| Option | Effect |
|---|---|
| `-FullCheck` | full preflight first: loads both models and makes a real sample prediction (about 15 s) |
| `-CheckOnly` | run the preflight and exit without starting the dashboard |
| `-SkipPreflight` | start immediately (affected features are then disabled with a message if something is missing) |
| `-Port 8600` | use another port (a busy port is skipped automatically) |
| `-NoBrowser` | do not open a browser window |

The first start takes about 10 seconds (TensorFlow is slow to import); every click afterwards takes about 1 second.
On a new machine Streamlit may ask once for an e-mail address: just press Enter.

### Start it and check that it is healthy (Windows PowerShell)

```powershell
cd "<the folder that contains app.py>"          # for example: open the project folder in File Explorer, then "Open in Terminal"

.\run_dashboard.ps1 -CheckOnly                  # preflight only; the summary must say "Dashboard can launch ... YES"
.\run_dashboard.ps1 -NoBrowser -Port 8599       # start the dashboard (leave this window open; Ctrl+C stops it)

# in a SECOND PowerShell window:
(Invoke-WebRequest http://localhost:8599/_stcore/health -UseBasicParsing).Content     # prints: ok
Get-NetTCPConnection -LocalPort 8599 -State Listen                                    # shows the listening process
```

Then open http://localhost:8599 . The sidebar **System status** and **Diagnostics** show the real state of each part.

## Preflight check

```powershell
..\.venv\Scripts\python.exe -m src.preflight            # full check, about 15 s
..\.venv\Scripts\python.exe -m src.preflight --fast     # quick check (what the launchers run)
```

It only reads files (never writes, trains or regenerates anything) and separates:

| Group | If something is wrong |
|---|---|
| **Required: dashboard** (Python, packages, `app.py`) | the launcher stops and tells you what to install |
| **Required: live prediction demo** (dataset, models, scaler, feature configs, they match each other, and a real sample prediction that must reproduce the saved test prediction) | the dashboard still starts; the demo explains what is missing and is disabled |
| **Recommended: Model Results tab** (saved prediction files, metrics) | warning only; that tab explains what is missing |
| **Optional: PySpark / Java** | never blocks anything; only `src/bigdata/spark_analysis.py` needs them |
| **Optional: TomTom API key** | never blocks anything; real-time mode cannot make predictions anyway |

## Setup from scratch (only if `..\.venv` does not exist)

This is the checklist for a **clean Windows machine** (Windows 10 or 11, PowerShell). The project was developed and tested with **Python 3.11** (3.11.9); other versions are not supported
(TensorFlow 2.16 and the pinned NumPy need it). The first install downloads about 1 GB (TensorFlow) and takes several minutes.

```powershell
# 0. Python 3.11:  py -3.11 --version     (if that fails, install Python 3.11 from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH")
git clone https://github.com/Aditi-SP/Urban_Traffic.git
cd Urban_Traffic                       # the folder that CONTAINS the project folder: the virtual environment is created here, next to it
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r Traffic-Congestion-Forecasting-main\requirements.txt
# optional, only to re-run the PySpark script (also needs Java 17+):
.\.venv\Scripts\python.exe -m pip install -r Traffic-Congestion-Forecasting-main\requirements-bigdata.txt
```

**Put the dataset in place.** The dataset is **not in the repository** (`*.csv` is git-ignored; see "Dataset"). Download `Banglore_traffic_Dataset.csv` (Kaggle:
`preethamgouda/banglore-city-traffic-dataset`) and save it as `Traffic-Congestion-Forecasting-main\data\raw\Banglore_traffic_Dataset.csv` (create the folders if they do not exist).
The trained model files in `model/` **are** in the repository.

```powershell
cd Traffic-Congestion-Forecasting-main
.\run_dashboard.ps1 -CheckOnly        # the summary must say "Dashboard can launch ... YES"; anything else names what to fix
.\run_dashboard.ps1                   # starts the dashboard and opens it in your browser
```

**What happens without the dataset:** the dashboard still starts. A red "Setup incomplete" box names the missing file, the forecasts and replays are disabled with the reason, and
`.\run_dashboard.ps1 -CheckOnly` reports "Live prediction demo available ... NO". Nothing is invented to fill the gap. Without the models or with a different dataset the same happens,
each with its own message (this is tested by `tests/failure_matrix.py`).

**About the `doucments/` folder.** The research paper, slides and their backups are kept **locally only** and are **not part of the repository**, so a fresh clone does not have `doucments/`.
Nothing in the dashboard needs it. `docs_tests/` checks those documents against the evaluation outputs and **skips cleanly (all tests skipped)** when they are absent: that is expected in a clone.

**If something goes wrong**

| What you see | Fix |
|---|---|
| `py -3.11` is not recognised | install Python 3.11 (link above) and open a **new** PowerShell window |
| `pip install` fails on TensorFlow or NumPy | you are not using Python 3.11: delete `.venv`, create it again with `py -3.11 -m venv .venv` |
| Launcher: "the project virtual environment was not found" | the `.venv` must be in the project folder or directly above it; repeat the setup above |
| `running scripts is disabled on this system` | run `run_dashboard.bat` instead, or `powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1` |
| Red "Setup incomplete" box, "Dataset ... not found" | save the CSV exactly at `data\raw\Banglore_traffic_Dataset.csv` (see above) |
| The first start is slow | TensorFlow takes about 10 seconds to import; the first model load takes 10 to 20 seconds (the page says so) |

## Other commands

Run from the project folder; replace `python` by `..\.venv\Scripts\python.exe` if the environment is not activated.

```powershell
# retrain (OVERWRITES model files and results: back them up first; the old model folder is only for reference)
python -m src.modelTraining.trainXgboost
python -m src.modelTraining.trainLstm2
python -m src.modelTraining.comparePerformancePlots

# tests
python -m unittest discover -s tests -v               # core software tests (skip cleanly if the dataset/models are absent)
python -m unittest discover -s docs_tests -v          # separate: checks paper/slides against model/metrics.json (read-only)
python tests/failure_matrix.py                        # 27 broken-project scenarios + Spark-without-Java, on scratch copies (about 7 min)
# real-browser check (start the dashboard first, e.g. .\run_dashboard.ps1 -Port 8599 -NoBrowser; needs Chrome or Edge):
python tests/browser_check.py shots 1536 700 8599
python tests/browser_check.py shots 1280 650 8599
python tests/browser_check.py shots 800 900 8599
# these never save a community report; they load map tiles from OpenStreetMap in the test browser

# check a file of newer observations (READ-ONLY; changes nothing); see "Forecast data coverage"
python -m src.dataProcessing.check_new_data path\to\new_observations.csv --target 2026-10-28

# new-feature tests: status/health, homepage, forecast data gap, Model Insights, validator; community reports; signal-gate benchmark
python -m unittest tests.test_status_and_data tests.test_community tests.test_signal_check -v

# big-data workflow (PySpark, local mode; needs Java 17+ and requirements-bigdata.txt; REWRITES reports/spark/)
python -m src.bigdata.spark_analysis

# figures for the paper/slides (drawn from the verified outputs only)
python -m src.modelTraining.make_document_figures

# single-road example predictions
python -m src.modelTraining.predictXgboost
python -m src.modelTraining.predictLstm
```

## Troubleshooting

| What you see | Meaning | Fix |
|---|---|---|
| Red "Setup incomplete" box at the top of the dashboard | a required file is missing or invalid; it names the file | restore it, or run `python -m src.preflight` |
| Demo tab says "Prediction is disabled" | trained model files or the dataset could not be loaded | same message at the top of the page |
| "Dataset ... " message | missing, empty, malformed, too small, or a different dataset | put the original `Banglore_traffic_Dataset.csv` in `data/raw/` |
| Launcher says the virtual environment was not found | `..\.venv` is missing | see "Setup from scratch" |
| PowerShell: "running scripts is disabled" | execution policy | use `run_dashboard.bat`, or `powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1` |
| Spark script: "Java was not found" | PySpark needs Java 17+ | install a JDK (e.g. Temurin from adoptium.net), open a new terminal; the dashboard does not need it |
| TomTom mode shows "No API key configured" | normal; the key is optional | see "Setting up `TOMTOM_API_KEY`" below |
| Live Traffic Map shows "Live traffic is not configured" | the **server process** has no usable `TOMTOM_API_KEY`; the page says why ("Why it was not found") | see "Setting up `TOMTOM_API_KEY`", then "Check the setup" and "Restart the dashboard"; the usual cause is a variable set in another window or after VS Code started |
| Sidebar says "Live traffic: provider problem" | the last real request to TomTom failed (key rejected, timeout, rate limit, no data near the point) | the text next to it says which; see "Check the setup" (`python -m src.live.check_key --request`) |
| Sidebar says "Live traffic: key found, not yet tested" | a key exists but no request has been made in this run | search a place and press **Get traffic for this place** |
| Forecasts and replays stop in August 2024 | the dataset ends on 2024-08-09 | expected; see "Forecast data coverage" |
| Community Reports says the report database is unavailable | the folder `data\community` cannot be written, or the file is damaged | make the folder writable; to start over, close the dashboard and delete `data\community\community.db` (it holds only local reports) |

## Live prediction demo (dashboard tab)

* **Mode 1 – Interactive demo:** pick a road and a day from the held-out test period; both trained models and both
  baselines forecast the next day from the real preceding observations ("historical replay"), or from a **what-if
  scenario** where you edit the day-t conditions (presets use the road's own training-period percentiles).
  The output is always labelled as historical/what-if, never as live traffic. No congestion category and no
  uncertainty interval is shown (no validated method/thresholds exist).
* **Mode 2 – Real-time API:** shows a TomTom Traffic Flow observation for a location you enter (inside the Bengaluru
  area). The API only gives a current speed/travel-time snapshot, so it **cannot** supply the models' inputs (volume,
  capacity, incidents, weather, history); the app displays the compatibility gap and blocks prediction. The reading is a
  live observation, never a forecast, and is labelled "Last fetched" (the provider gives no measurement time). Requests
  go through the **same provider service as the Live Traffic Map** (`src/live/tomtom.py`: shared request cap, refresh
  limit, 429 cooldown and `LIVE_CACHE_TTL_S` setting, in memory only; see "Live Traffic Map" below), so a reading or
  limit hit on one page applies to the other. The provider path itself was checked with real single-point requests (see "Verification status of the TomTom integration"); this page's Mode 2 uses that same path.
  Key: environment variable `TOMTOM_API_KEY` or `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`).
  `src/live/tomtom.py` is the only TomTom client in the application; the older stand-alone request function in
  `src/app/realtime.py` (no cache or request cap) has been removed, and `realtime.py` now only describes the gap between
  provider fields and model inputs.
* **Experimental models:** the LSTM and XGBoost forecasts are shown beside the persistence and road-mean baselines, and the
  pages state that neither model beat the road-mean baseline on the test set.
* A 2–3 minute presentation script is in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## Live Traffic Map (dashboard page)

**Purpose:** a first step towards live monitoring. You search for a place in Bengaluru, pick a result, and, if a TomTom
key is configured, the page requests the current traffic of the road segment nearest to that place and draws it on the
map. It is a separate, read-only view of provider observations. **It is not connected to the forecasting models**: their
inputs (volume, capacity, incidents, weather, history) are not available from the provider, so no live reading is ever
fed to them and no prediction is made from it. The historical dataset has no coordinates, so none of its roads are placed
on the map.

What it does:

* Shows an OpenStreetMap basemap centred on Bengaluru. With no key, or before anything is selected, the map is simply
  the basemap: no markers and no traffic values are drawn.
* Place search (Nominatim) runs only when you press **Search** (no autocomplete), is limited to the Bengaluru area, and
  results are cached in memory.
* Traffic is drawn only from a valid provider response for the place you selected. Invalid responses, timeouts, rate
  limits and other errors are reported on the page and draw nothing.
* The time shown is **"Last fetched"** (this computer's clock when the app received the response). The API provides no
  measurement time, so none is shown. A reading older than 10 minutes is labelled stale and drawn in grey.
* Provider data is kept **in memory only**. Nothing is written to disk, because TomTom's storage/caching terms have not
  been verified. Safeguards (all in `src/live/tomtom.py`, tested with mocked responses):

  | Safeguard | Setting |
  |---|---|
  | In-memory cache: a repeated request for the same point (about 11 m) reuses the reading | 300 s; override with `LIVE_CACHE_TTL_S`. With `0` no reading is kept or reused at all (see below) |
  | Refresh limit: an explicit refresh inside this window is answered from memory (with `LIVE_CACHE_TTL_S=0` it is refused instead, and the reading already on screen stays) | 30 s |
  | Simultaneous requests for the same point share **one** provider request | always on |
  | Request cap per app run (a failed attempt counts; cache hits do not) | 300; override with `LIVE_MAX_REQUESTS` |
  | After HTTP 429 no provider request is sent for a while | 60 s |
  | Cache size limit | 200 readings |
  | A reading older than 10 minutes is labelled stale and never coloured as current | 600 s |

  **`LIVE_CACHE_TTL_S=0` (strictest setting):** every request goes to TomTom and nothing is retained between requests;
  only the time of the last request per point is remembered, to enforce the 30 s refresh limit, the 429 cooldown and the
  request cap. Simultaneous requests for the same point still share one provider call while it is in progress. The
  reading already shown on the page remains visible with its own "Last fetched" time (and becomes stale after 10 minutes).
  To set it for your Windows user (new terminals only; remove it again with the second line):

  ```powershell
  [Environment]::SetEnvironmentVariable("LIVE_CACHE_TTL_S", "0", "User")
  [Environment]::SetEnvironmentVariable("LIVE_CACHE_TTL_S", $null, "User")
  ```

  The request cap is per application run, so restarting the app resets it; it is **not** a monthly quota. TomTom's
  pricing page (read 2026-10-29) lists 20,000 free monthly requests for Flow Segment Data, but your key's actual
  allowance and limits should be checked in your TomTom account.
* The key is used only by the Python server process; it is never sent to the browser or displayed.
* The same provider service (and so the same safeguards below) also serves Mode 2 of the **Traffic Forecasts** page.

### Setting up `TOMTOM_API_KEY` (optional)

The map and place search work without a key; only live traffic needs one. You need your own TomTom developer key **with the Traffic API
enabled for it** (check TomTom's current plans, request limits and terms yourself; they are not verified here). Never put a real key in
source files, screenshots, Git or a chat. `your-key-here` below is a placeholder.

**The one rule:** the variable must be named exactly `TOMTOM_API_KEY`, and it must be present in the environment of the **server process**
(the python process that runs the dashboard) when that process **starts**. Windows copies variables into a process at its start. So a
variable you set later, in another terminal tab, or saved for your user *after* a terminal was opened (**terminals inside VS Code keep the
environment VS Code started with**), is not seen by a dashboard that was started from that terminal, nor by one that is already running.
That is the usual reason for "I set it but it still says not configured".

**Recommended (saved once; the launcher picks it up even from an older terminal).** In PowerShell, type this and paste the key at the
hidden prompt (it is not echoed and does not enter your command history):

```powershell
$secure = Read-Host "Paste your TomTom key (input is hidden)" -AsSecureString
$plain  = [System.Net.NetworkCredential]::new('', $secure).Password
[Environment]::SetEnvironmentVariable("TOMTOM_API_KEY", $plain, "User")
Remove-Variable plain, secure
```

Then **stop any running dashboard and start it with the launcher** (see "Restart the dashboard" below):

```powershell
cd "<the folder that contains app.py>"          # for example: open the project folder in File Explorer, then "Open in Terminal"
.\run_dashboard.ps1
```

`run_dashboard.ps1` loads `TOMTOM_API_KEY`, `LIVE_CACHE_TTL_S` and `LIVE_MAX_REQUESTS` from your saved Windows settings whenever the terminal
you start it from does not have them, and tells you so (it never prints the key). Starting with a bare `streamlit run app.py` does **not**
do this. To remove the saved key: `[Environment]::SetEnvironmentVariable("TOMTOM_API_KEY", $null, "User")`.

*Alternative, for this PowerShell window only:* `$env:TOMTOM_API_KEY = "your-key-here"` and then `.\run_dashboard.ps1` **in the same window**.

*Alternative, a file:* create `.streamlit\secrets.toml` **inside the inner project folder** (the one that contains `app.py`) or
`%USERPROFILE%\.streamlit\secrets.toml` with exactly one line (git-ignored; see `.streamlit\secrets.toml.example`):

```toml
TOMTOM_API_KEY = "your-key-here"
```

The environment variable wins over the file. A file that is not valid TOML, or has a different entry name, is reported by the checks below
instead of being ignored silently.

### Check the setup, in this order (the key is never printed)

```powershell
# 0. Only in a terminal that was opened BEFORE you saved the variable: copy the saved value into THIS window (nothing is printed)
$env:TOMTOM_API_KEY = [Environment]::GetEnvironmentVariable("TOMTOM_API_KEY", "User")

# 1. What would a dashboard started from this window see? (no network request)
python -m src.live.check_key
#    expected: "TomTom key: configured (source: environment variable). The value is not shown."

# 2. Does the launcher see it?
.\run_dashboard.ps1 -CheckOnly
#    expected: "TomTom key: found in the environment (value not shown)."   and   "Optional TomTom key configured . YES"

# 3. ONE real request to TomTom with your key (uses one request of your quota; nothing is stored)
python -m src.live.check_key --request
#    expected: "RESULT: OK. TomTom returned a valid traffic reading: current speed ... km/h ..."
```

If step 1 or 2 says the key is not configured, the text next to it says why: no variable with that name, a similarly named variable (for
example `TOMTOM_KEY`), a value with spaces, a secrets file that is not valid TOML, or "saved in your Windows user environment but this process
does not have it". If step 3 fails, its `RESULT:` line says which of these it is:

| `RESULT` / message | Meaning | What to do |
|---|---|---|
| `NOT CONFIGURED` | no usable key in this terminal | follow the lines above it |
| `KEY NOT ACCEPTED` after **HTTP 401** | **observed** with a wrong or missing key: TomTom says "You are missing valid authentication credentials" | copy the key again (completely, no spaces); create a new one if needed |
| `KEY MAY NOT BE ENABLED FOR THE TRAFFIC FLOW API` | HTTP 403 whose answer suggests the Traffic API is not enabled for the key (**the wording of a real 403 is unverified; this is a guess from the answer's text**) | in your TomTom developer account check that the Traffic API is enabled for that key |
| `QUOTA MAY BE USED UP` | HTTP 403 whose answer suggests the quota is exhausted (**same caveat: unverified wording**) | check usage and plan in your TomTom account |
| `KEY NOT ACCEPTED` after a 403 that says nothing useful | the neutral message: "the key may be invalid, expired, not enabled for the Traffic API, or over its quota. TomTom's answer did not say which" | check the key in your TomTom developer account |
| `RATE LIMITED` | HTTP 429 | wait; the dashboard pauses requests for 60 s |
| `TIMEOUT` / `NO CONNECTION` | network problem | check your internet connection and retry |
| `TOMTOM PROBLEM` | TomTom reported a temporary problem (HTTP 5xx) | retry later |
| `NO TRAFFIC DATA NEAR THE POINT` | **observed**: HTTP 400 with "Point too far from nearest existing segment." (no road near the point) | choose a point on or near a main road |
| `UNEXPECTED ANSWER` | TomTom answered with a status this application does not understand (for example HTTP 404 for an unknown service version, or any other 400); nothing is shown | report it; nothing is guessed |
| `UNREADABLE ANSWER` | the response was not in the expected format | report it; nothing is guessed or filled in |

Only what TomTom was **observed** to say is stated as fact (the 401 and the "no road" 400 above) or **documented** to mean (HTTP 403: "The supplied API Key is not valid for this
request.", HTTP 429: too many requests). How a real 403 body is worded has never been seen, so a cause taken from it (quota, not enabled) is shown only as what the answer
*suggests*, and an answer that says nothing useful gets the neutral message listing the possible reasons. The response body is only used to choose a message; it is never shown or logged.

### Restart the dashboard (a running dashboard never sees a key set after it started)

```powershell
# in the window that runs it: press Ctrl+C. If you cannot find that window, list the dashboards (the port is in the command line):
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit run app.py' } |
    Select-Object ProcessId, @{n='Port';e={ if ($_.CommandLine -match '--server.port (\d+)') { $Matches[1] } }}, CreationDate

# stop the ones you want (stops the process and its children; this stops EVERY Streamlit app named app.py):
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit run app.py' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F }

# start it again and note the URL it prints (if the port is busy it uses the next one, and says so):
.\run_dashboard.ps1
```

If the launcher says "Port 8501 is busy", an **old dashboard is still running** on 8501 and your browser tab may still be showing it; open the
URL the launcher prints, or stop the old one first.

### Confirm that real traffic works

1. In the dashboard sidebar open **Diagnostics (files, models, data)**. Under *TomTom key configuration* it must say
   **"TomTom key: configured (source: ...)"**. It also shows when this dashboard process started and the `LIVE_CACHE_TTL_S` / `LIVE_MAX_REQUESTS`
   values it is really using (and warns if they differ from the ones saved for your Windows user).
2. The sidebar chip then reads **Live traffic: key found, not yet tested** (amber; a key alone proves nothing).
3. Open **Live Traffic Map**, search `Silk Board Junction`, press **Search**, choose the result, press **Get traffic for this place**.
4. **Real traffic is working only when** a coloured segment appears, the cards show **Current speed** and **Last fetched**, the caption shows
   "Traffic data: © TomTom", and the sidebar chip changes to **Live traffic: last response OK hh:mm:ss**. (TomTom gives no measurement time:
   "Last fetched" is this computer's clock.)
5. If it fails, the page shows the reason from the table above and the chip says **Live traffic: provider problem**; no traffic is drawn.

Until step 4 has happened, live traffic is **unverified**. Readings stay in memory only.

### Verification status of the TomTom integration

Checked on **2026-10-30** with a real key, through the project's own client and against TomTom's official documentation.

* **Verified against the real service.** Single-point Flow Segment Data requests returned a valid segment for **16 of 16 main-road points** spread over Bengaluru
  (Silk Board, MG Road, Majestic, Hebbal, KR Puram, Electronic City, Yeshwanthpur, Marathahalli, Whitefield, Banashankari, Jayanagar, Peenya, Bellandur,
  Kengeri, Hosur Road, Outer Ring Road) and for **4 of 4 points at the edges of the map scope**; no error occurred. The response has **no measurement
  timestamp** and its `Cache-Control` is `public, max-age=0`.
* **Recorded real error responses** (the shapes the tests use, labelled as recorded): a point with no road near it (tested with points in the sea, outside the map
  scope) gives HTTP 400 `{"error": "Point too far from nearest existing segment.", "httpStatusCode": 400, "detailedError": {"code": "INVALID_REQUEST", ...}}`; a wrong
  or missing key gives HTTP 401 `{"detailedError": {"code": "Unauthorized", "message": "You are missing valid authentication credentials"}}`; a malformed or missing
  point gives other HTTP 400s ("Invalid point format.", "Missing point parameter."); an unknown service version gives HTTP 404 in a different JSON shape.
* **Documented but never observed here:** HTTP 403 ("The supplied API Key is not valid for this request."), HTTP 429 ("Too many requests were sent in a given amount of
  time for the supplied API Key."), 500, 503 and 596 ("Unknown version of the service."). **Unknown:** how a real 403 body is worded, and whether an exhausted
  quota answers 403 or 429. The classifier (`src/live/tomtom.py`) therefore never presents a guessed 403 cause as fact: see "Check the setup" above.
* **Not verified:** the binding developer terms, the quota of *your* account, and coverage beyond the points listed.

**Coverage, honestly.** TomTom answers for the **nearest road segment** to the point, so a valid reading describes that road, which the page draws on the map. When
that road is more than 150 m from the point you chose, the page says so ("TomTom answered for the nearest road, about N m from the point you chose"). The 20 points
above are a probe, not a survey: **no claim of city-wide or lane-level coverage is made**, and a place with no road near it is reported as "no traffic data near the
point". The probe stored nothing (only the outcome per point was looked at); readings are never stored.

### TomTom terms, quota and attribution: what has and has not been verified

Official sources read on **2026-10-30**:

* Flow Segment Data reference (page last edited 2025-04-17): <https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/traffic-flow/flow-segment-data>. Request format,
  status codes (200, 400, 403, 405, 429, 500, 503, 596), JSON error body (`error`, `httpStatusCode`, `detailedError.code` / `.message`), `Cache-Control: public, max-age=0`.
* Pricing page: <https://docs.tomtom.com/pricing>. It lists **"Traffic Flow API (Segment Data): Free 20K monthly"** (a free monthly allowance, no credit card). It does **not** state
  a requests-per-second limit, what happens when the allowance is exceeded, or per-request prices. What your account is actually entitled to is **account-specific and not known here**.
* Traffic API introduction (page last edited 2026-09-15): <https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/product-information/introduction>. Describes real-time observed
  speeds and travel times; states **no** refresh frequency, usage limit, caching rule or attribution rule for this endpoint.
* Attribution mechanism for the Maps APIs: Copyrights API, <https://docs.tomtom.com/map-display-api/documentation/tomtom-maps/v2/copyrights/copyrights> (its caption example is
  "©TomTom"). A TomTom community answer surfaced by search (not re-opened, now redirected) says direct Traffic API users should display notices obtained from that API in the
  application's legal-notices ("about") section; that is **secondary and unconfirmed**.

**Not verified:** the TomTom developer terms that govern caching, storage, derived data, machine-learning use and attribution **could not be read**: the legal page
(<https://docs.tomtom.com/legal/terms-and-conditions>) is navigation only for the tool used, and TomTom's website terms say nothing about API data. Consequences, unchanged:

* Readings are **not stored** and **not used to build datasets or train or evaluate models**. Do not change this until the terms are read and explicitly allow it.
* Whether reusing a reading for up to 5 minutes is allowed is **unconfirmed**; `max-age=0` suggests clients should not rely on cached copies. The strictest behaviour is
  `LIVE_CACHE_TTL_S=0`, which is the setting in use (see above).
* The request cap (300 per run) and the 30 s refresh limit are **this project's own safeguards**, not TomTom limits. They stay.
* The page shows "Traffic data: © TomTom" beside the reading and OpenStreetMap's attribution on the map. **Whether TomTom requires more notices for direct use of this endpoint is
  unresolved**; before publishing the application or widening its use, read TomTom's current developer terms and attribution rules (or ask TomTom) and follow them.

### Colour bands are display conventions

Segments are coloured by current speed as a share of free-flow speed: **80% or more = free-flowing (green), 50% to under
80% = slow (amber), under 50% = congested (red)**; a reported road closure is red and a stale reading is grey. These
bands and thresholds are defined by this app for display only. They are not categories returned by TomTom and **are not
validated traffic thresholds**; they have not been checked against any ground truth.

### Attribution and usage limits

* Map data: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (Open Database Licence). Place search
  uses OpenStreetMap's [Nominatim](https://operations.osmfoundation.org/policies/nominatim/) service, shown on the page as
  "via Nominatim". Traffic data, when shown, is attributed to TomTom.
* Nominatim's public service allows at most 1 request per second, requires an identifying User-Agent, and forbids
  autocomplete-style searching. The app follows this (searches only on button press, a wait of at least 1.1 s between
  requests, cached results).
* **Basemap tiles come from OpenStreetMap's standard tile server (`tile.openstreetmap.org`).** Its usage policy does not
  allow heavy use, and it offers no service guarantee. That is acceptable for a local demo or coursework, but **not for a
  production deployment or many users**: use a commercial or self-hosted tile provider (and check its attribution
  rules) before any wider use.
* The map is drawn with Leaflet 1.9.4 (BSD 2-Clause licence), bundled in `src/ui/vendor/leaflet/` and checked against its
  published integrity hashes each time it loads, so the browser loads no script from a CDN. It is not updated
  automatically.

## System status and Diagnostics

The sidebar **System status** chips and the homepage **At a glance** cards come from real checks (`src/ui/health.py`), not fixed text.
Colours: green = checked and fine, amber = works but with a limitation or not yet tested, red = a check failed, grey = unknown.

| Item | What is actually checked | Typical text today |
|---|---|---|
| Live traffic (TomTom, **external**) | whether a key is configured, and what the shared provider service has seen in this run (last valid response, last failure, rate-limit pause). **No test request is sent** | "Live traffic: not configured" (red), then "key found, not yet tested", then "last response OK hh:mm:ss" or "provider problem" |
| Historical data | the latest observation date in the loaded dataset and its age in days | "Data: historical, 2022-2024 (latest 2024-08-09)" (amber while older than 30 days) |
| Models | the trained files exist (loading is tested on demand) | "Models: not validated for 2026" (amber; red if a file is missing) |
| Community reports | the local report database can be opened | "Reports: working" |

Application health and provider health are kept apart: a red "Live traffic" chip with everything else green means the app works and
only the external provider (or its key) is the problem. Messages never contain keys, absolute paths or stack traces.

**Loading state.** While a run is busy (a place search, a traffic request, a prediction, loading the models, the Diagnostics checks) a spinner names what is happening, for example
"Loading the trained models (the first time can take 10 to 20 seconds)...". The sidebar shows "Updating... the status below is from the previous action." above the chips until the run
has finished, so an older status is never mistaken for the current one; the line is removed when the run ends, also if the page crashes.

**Diagnostics** (sidebar; it replaces the old "Files / Status" list and keeps every entry): the same items with what was checked, the
files the dashboard reads (present / MISSING / optional), and two on-demand checks: **Check models load** (really loads the LSTM and
XGBoost) and **Run preflight** (the same fast preflight the launcher runs). A failure is shown in red with its reason; nothing is
shown as OK that was not checked.

**Homepage** (the Live Traffic Map page): the setup message when no key is configured, the map and search, then **At a glance**: live
traffic, data freshness, a one-sentence model summary quoted from the saved held-out evaluation, and the number of community reports,
with links to Traffic Forecasts, Model Insights and Community Reports. **Model Insights** (its own page) documents the dataset, the
held-out split, the baselines and what MAE, RMSE and R² mean. None of this is a live measurement or a forecast.

## Community Reports

Short, **anonymous, unverified** text reports from road users (category, road or place, description), stored **on this computer**
in `data/community/community.db` (SQLite; set the `COMMUNITY_DB_PATH` environment variable to use another file). The folder is
git-ignored. Everyone using the same running dashboard sees the same reports. They are not traffic measurements and are never used
by the models.

* **Sending:** fill in the form and press **Send report**. "Saved" is shown only after the report was stored and read back from the
  database; otherwise you get an error that says what to fix (too short or long, a missing field, a duplicate, a rate limit, a
  database problem). Nothing is dropped silently.
* **Listing:** the newest 50 reports, read from the database on every page load (so they survive a refresh), each with its age;
  reports older than a day are marked "possibly outdated". With no reports the page says so; sample reports are never shown.
* **Safeguards (no accounts):** text only, 3-120 characters for the place and 10-500 for the description; control characters
  refused; at most 5 reports per browser session per 10 minutes, 60 per hour overall, 10,000 in total; an identical report within 10
  minutes is refused; a double click cannot create two rows. All user text is shown as escaped plain text. There is **no edit,
  delete or photo upload** in the application. Please do not enter names, phone numbers or number plates.
* **Viewing never creates the database**; the first saved report does. To remove all reports, close the dashboard and delete
  `data\community\community.db` yourself.

## Forecast data coverage

The dashboard has **three different kinds of output, and they are labelled as such**:

| Kind | What it is | Dates |
|---|---|---|
| **Historical observed traffic** | the real daily records of the Kaggle dataset (provenance and licence unverified); observed, but **not live** | 2022-01-01 to **2024-08-09** |
| **Historical replay and what-if** | the models re-run on real past days, or on past days with the last day's conditions edited. Neither is a forecast of the future | past days only |
| **Model-generated future forecasts** | one day after the last observed day (the models' only horizon) | **not available**: the latest date this dataset can reach is 2024-08-10; nothing newer has been observed |

**Why forecasts do not reach October 2026:** the dataset is a static file whose last observation is 2024-08-09 (810 days before
2026-10-28). Nothing in the code limits the year; there is simply no later data, no source of it in this project, and none is
invented. The models also were never validated for 2026 and showed no skill beyond the per-road mean even on 2024 data.

**Sources searched for newer data (2026-10-30).** No legitimate, usable extension was found: the Kaggle dataset's licence and any newer version could not be verified; the Bengaluru
open-data listings found (OpenCity, India Data Platform) are traffic *violations*, one-way roads and planning PDFs, not a congestion index, and licences vary per listing; a
volunteer-built open road-speed dataset on Datameet has a different schema and is not an official feed. Nothing was downloaded, merged or invented. If you obtain a dataset you may
legally use, follow the steps below.

**If you obtain real newer observations** (in the same columns as the dataset, and with the right to use them), check them first with
the read-only validator. It changes nothing:

```powershell
python -m src.dataProcessing.check_new_data path\to\new_observations.csv --target 2026-10-28
```

It applies the project's own validation rules, and reports the date range, overlap with the current data, gaps, unknown roads and
whether the file reaches the target date. Using such data for forecasts would additionally require merging it deliberately,
retraining and re-evaluating the models, and then a new held-out evaluation: that needs separate approval and is not done here.

## Big-data component

`src/bigdata/spark_analysis.py` uses PySpark (local mode, one machine) to validate/clean the CSV (FAILFAST schema, null,
duplicate and range checks) and aggregate congestion by road, area, weekday, month and weather into `reports/spark/`.
It is not distributed and not streaming; the dataset is small and has no hour-of-day field (no rush-hour analysis).

## Evaluation report

`model/evaluation_report.md` / `.json` (written by `comparePerformancePlots.py`): dataset source, split dates, target,
horizon, features, metrics, artifact hashes, software versions and limitations.

### Independent audit of the saved evaluation (2026-10-30)

`tests/test_evaluation_audit.py` re-checks the saved results by a route that does not use the evaluation scripts. It trains nothing, writes nothing and changes no model file. It found **no
defect**, so nothing was retrained and no number changed:

* all four models (LSTM, XGBoost, persistence, road mean) are scored on the **identical 856 test rows** (same dates, roads, actual values and order);
* `metrics.json` equals the metrics recomputed from the saved predictions (largest difference below 1e-7);
* every "actual" value, and every persistence prediction (the value of the day before the target), equals the raw dataset;
* the split is chronological (training targets to 2023-10-30, validation to 2024-03-20, test from 2024-03-21) and the road-mean baseline uses **training targets only**;
* every input window ends the day before its target; the LSTM scaler's range is the **training windows'** range (it differs from the all-data range);
* hyperparameters are fixed in the training scripts and early stopping uses the validation split. The scripts print the test RMSE, so some unconscious influence of the test score on past
  choices cannot be excluded; the test set was **not** used for tuning in this audit.

The measured result is unchanged: road-mean baseline RMSE 19.85, LSTM 19.93, XGBoost 20.07, persistence 28.23. The road mean is at least as good as both learned models, and every RMSE
difference has a 95% bootstrap interval that includes zero. **No model is claimed to win.** The test asserts this too: it fails if a learned model ever appeared to beat the baseline without a
supported interval, which would force the wording to change.

**What more data would be needed.** The Stage 5A benchmark found no within-road signal beyond each road's usual level. To show a learned model can do better you would need *real* observations
that are newer (the file ends on 2024-08-09), more frequent than daily, and richer (hourly counts or speeds per road), with a right to use them. None can be generated or assumed here.

## Stage 5A signal-gate benchmark

`src/analysis/signal_check.py` asks whether the data contains within-road, time-dependent signal that a learned model could
use beyond each road's usual level. It **trains no model**, uses no network and writes only to its own output directory
(`experiments/stage5a/`; it refuses to write into `model/`, `data/`, the documents, `src/` or `tests/`). Saved prediction files
in `model/` are only read.

```powershell
python -m src.analysis.signal_check            # writes experiments/stage5a/signal_check.json and .md (about 10 seconds)
python -m unittest tests.test_signal_check -v  # synthetic no-signal and known-signal tests, plus real-data checks
```

* **Target and split:** the project's own: the same road on the next calendar day, chronological 70/15/15 by target date.
* **Baselines (no tuning, no labels leaked):** expanding per-road mean (the reference; uses only observations up to day t),
  per-road training mean, per-road weekday mean, last value, last-7 mean, global mean.
* **Evaluation:** the official test split, and a rolling-origin evaluation (5 chronological folds after a warm-up of the first
  half of the dates). Reported: MAE, RMSE, R², within-road correlation of prediction and actual, and the RMSE difference to
  the reference with a date-cluster bootstrap confidence interval.
* **Signal diagnostics (development period only, so test labels play no part):** within-road autocorrelation at lags
  1, 2, 3, 7 and 14 days, and weekday, month-of-year, shared-date and road-by-month effects, with Bonferroni-adjusted tests.
* **Result on this dataset:** `NO_USABLE_SIGNAL`. No baseline is distinguishably better than the expanding road mean, the
  within-road lag-1 correlation is about 0, and no calendar effect was found. The saved LSTM and XGBoost differ from the
  reference by +0.12 and +0.26 RMSE (confidence intervals include 0). Training a new learned model on this dataset is not
  justified by this evidence.
* **Limits:** it describes this dataset only (its provenance is unverified), small effects cannot be excluded, and it does
  not say how a model would perform on different data.

## Project structure

    ├── app.py                         Streamlit dashboard
    ├── data/raw/                      Banglore_traffic_Dataset.csv (you must supply it)
    ├── data/community/                community.db, created by the first saved report (git-ignored)
    ├── model/
    │   ├── model.h5, scaler.pkl, lstm_features.json            LSTM + its scaler/feature order
    │   ├── xgboost_model.json, xgboost_features.json           XGBoost + its feature order
    │   ├── xgboost_/lstm_/baseline_predictions.csv             aligned test predictions
    │   ├── metrics.json, saved_metrics.txt, evaluation_report.md/json, *.png   evaluation outputs
    │   └── original_backup/           the ORIGINAL (pre-fix) artifacts, kept for reference only
    ├── reports/spark/                 PySpark result tables
    ├── DEMO_SCRIPT.md                 2-3 minute judge demo script
    ├── src/
    │   ├── app/demo.py, realtime.py          demo logic and real-time interface
    │   ├── bigdata/spark_analysis.py         PySpark workflow
    │   ├── dataProcessing/forecast_data.py   shared pipeline (features, windows, split)
    │   ├── analysis/signal_check.py          Stage 5A signal-gate benchmark (trains nothing)
    │   ├── community/store.py                Community Reports store (SQLite, local, anonymous)
    │   ├── dataProcessing/check_new_data.py  read-only validator for newer observations
    │   ├── ui/health.py, diagnostics.py      real status checks, the System status chips and the Diagnostics panel
    │   ├── live/geo.py, geocode.py, tomtom.py    Live Traffic Map: location checks, Nominatim search, TomTom client
    │   ├── ui/                               dashboard pages/navigation (live_map.py, leaflet_map.py, vendor/leaflet/)
    │   ├── dataProcessing/preprocess2.py, preprocess_xgboost.py   LEGACY, not used for training
    │   └── modelTraining/  trainXgboost.py, trainLstm2.py, comparePerformancePlots.py,
    │                       predictXgboost.py, predictLstm.py
    ├── run_dashboard.ps1 / .bat       launchers (use ..\.venv directly; run the preflight first)
    ├── src/preflight.py               preflight check (required vs optional components)
    ├── experiments/stage5a/           outputs of the signal-gate benchmark (separate from model/)
    ├── tests/                         core software tests + failure_matrix.py + browser_check.py
    ├── docs_tests/                    separate read-only tests of the paper/slides (skip when doucments/ is absent)
    ├── notebooks/                     exploration
    └── doucments/ (local only)        slides and research paper: NOT in the repository, absent from a fresh clone
    ├── requirements.txt               core dependencies
    └── requirements-bigdata.txt       optional: PySpark

## Limitations

* No within-road signal found: beyond each road's usual level, next-day congestion could not be predicted from this data (see
  "Stage 5A signal-gate benchmark"). Do not present the models as accurate forecasters. They are experimental, and the
  evaluation does not show them to be better than the per-road historical mean.
* Recent real-world performance has not been established: the dataset ends on 2024-08-09 and nothing was validated against later conditions.
* Real TomTom readings were retrieved for 20 probe points in the Bengaluru area (no city-wide coverage is claimed; TomTom answers for the nearest road, and the page says when that road is far
  from your point). TomTom's binding terms, caching/storage/ML rules, attribution requirements and your account's quota are **unresolved** (see "TomTom terms, quota and attribution"), and the wording of
  a real HTTP 403 has never been observed. On your computer, live traffic is verified only when you see a real success (see "Confirm that real traffic works").
* No newer historical observations exist in this project; forecasts cannot reach 2025-2026 (see "Forecast data coverage").
* Community Reports are local, anonymous and unverified; they have no moderation, accounts, editing or photos.
* Daily, gappy data, so "next day" samples exist only where both days were observed.
* One chronological split (no cross-validation); the test period is about 4.5 months.
* The dataset looks synthetic/aggregated rather than raw sensor data (about 43.5% of congestion values are exactly 100); this is
  an inference, not a verified fact.
* An input window is the last 7 **observations**, not 7 calendar days (57% of windows span more than 7 days; the gap is a
  feature). Samples exist only when day t+1 was observed, and days whose next day is missing have lower congestion
  (75.1 vs 83.9), so the evaluation is conditional on that. Results are similar in every window-span bucket.
* LSTM numbers come from one training run; differences between LSTM, XGBoost and the road-mean baseline are inside the
  bootstrap noise (see `model/evaluation_report.md`), so **no ranking among them is claimed**.
* (Local files, not in the repository:) the research paper (`doucments/Research Paper/`) and the slides (`doucments/ppt/`) were rewritten to match this
  implementation and the verified results. The originals, which quoted leaky results, hour-of-day / "rush hour"
  claims, model settings the code never used, and several unverifiable or mis-dated references, are kept untouched in
  `doucments/original_backup/` for reference only and must not be presented. Their figures come from
  `python -m src.modelTraining.make_document_figures`; `tests/test_documents.py` checks that the documents still match
  `model/metrics.json`. (The old code and `model/original_backup/model.h5` used 48-step windows; the documents never stated 48.)
* References in the paper were checked against Crossref (DOIs listed). The TomTom documentation and the Kaggle dataset
  page were only consulted as web pages; the dataset's source and licence remain unverified.
* See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the full methodology audit.

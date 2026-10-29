# 🚦 Traffic Congestion Forecasting: LSTM vs XGBoost

Forecasts **next-day congestion level per road** in Bangalore with an LSTM and an XGBoost model,
compares them with simple baselines on the same held-out data, and shows the results in a Streamlit dashboard.

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
| Road-mean baseline | 15.338 | 19.846 | 0.207 |

**Interpretation:** both models beat persistence but do **not** beat the road-mean baseline, i.e. they capture which
roads are congested on average but essentially no day-to-day signal. In this dataset day-to-day autocorrelation is
only about 0.2, and congestion is mostly explained by same-day measurements (volume, capacity use) that are not
available the day before. Earlier versions of this project reported XGBoost R² = 0.9995; that came from target
leakage (features computed from the target itself) and is not a valid forecasting result.
Re-run `python -m src.modelTraining.comparePerformancePlots` to regenerate these numbers
(`model/saved_metrics.txt`, `model/metrics.json`).

## Quick start (Windows)

The project's virtual environment sits next to this folder (`..\.venv`). The launchers call it directly, so **you do not
need to activate anything**.

```powershell
cd C:\Users\User\Downloads\Traffic-Congestion-Forecasting-main\Traffic-Congestion-Forecasting-main
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

```powershell
cd C:\Users\User\Downloads\Traffic-Congestion-Forecasting-main
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Traffic-Congestion-Forecasting-main\requirements.txt
# optional, only to re-run the PySpark script (also needs Java 17+):
.\.venv\Scripts\python.exe -m pip install -r Traffic-Congestion-Forecasting-main\requirements-bigdata.txt
```

You also need the dataset `data/raw/Banglore_traffic_Dataset.csv` (git-ignored; see "Dataset" above) and the validated
model files in `model/`. Neither is created by the launchers or the preflight.

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
python tests/failure_matrix.py                        # 26 broken-project scenarios + Spark-without-Java, on scratch copies (about 7 min)
# real-browser check (start the dashboard first, e.g. .\run_dashboard.ps1 -Port 8599 -NoBrowser; needs Chrome or Edge):
python tests/browser_check.py shots 1536 700 8599

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
| TomTom mode shows "No API key configured" | normal; the key is optional | see "Live prediction demo" below |

## Live prediction demo (dashboard tab)

* **Mode 1 – Interactive demo:** pick a road and a day from the held-out test period; both trained models and both
  baselines forecast the next day from the real preceding observations ("historical replay"), or from a **what-if
  scenario** where you edit the day-t conditions (presets use the road's own training-period percentiles).
  The output is always labelled as historical/what-if, never as live traffic. No congestion category and no
  uncertainty interval is shown (no validated method/thresholds exist).
* **Mode 2 – Real-time API:** an interface for the TomTom Traffic Flow API (`src/app/realtime.py`). The API only gives
  a current speed/travel-time snapshot, so it **cannot** supply the models' inputs (volume, capacity, incidents,
  weather, history); the app displays the compatibility gap and blocks prediction. It is **unverified against the live
  service** (no key was available). Key: environment variable `TOMTOM_API_KEY` or `.streamlit/secrets.toml`
  (see `.streamlit/secrets.toml.example`).
* A 2–3 minute presentation script is in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## Big-data component

`src/bigdata/spark_analysis.py` uses PySpark (local mode, one machine) to validate/clean the CSV (FAILFAST schema, null,
duplicate and range checks) and aggregate congestion by road, area, weekday, month and weather into `reports/spark/`.
It is not distributed and not streaming; the dataset is small and has no hour-of-day field (no rush-hour analysis).

## Evaluation report

`model/evaluation_report.md` / `.json` (written by `comparePerformancePlots.py`): dataset source, split dates, target,
horizon, features, metrics, artifact hashes, software versions and limitations.

## Project structure

    ├── app.py                         Streamlit dashboard
    ├── data/raw/                      Banglore_traffic_Dataset.csv (you must supply it)
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
    │   ├── dataProcessing/preprocess2.py, preprocess_xgboost.py   LEGACY, not used for training
    │   └── modelTraining/  trainXgboost.py, trainLstm2.py, comparePerformancePlots.py,
    │                       predictXgboost.py, predictLstm.py
    ├── run_dashboard.ps1 / .bat       launchers (use ..\.venv directly; run the preflight first)
    ├── src/preflight.py               preflight check (required vs optional components)
    ├── tests/                         core software tests + failure_matrix.py + browser_check.py
    ├── docs_tests/                    separate read-only tests of the paper/slides
    ├── notebooks/, doucments/         exploration, slides, research paper
    ├── requirements.txt               core dependencies
    └── requirements-bigdata.txt       optional: PySpark

## Limitations

* Weak signal: next-day congestion is barely predictable from this data; do not present the models as accurate forecasters.
* Daily, gappy data, so "next day" samples exist only where both days were observed.
* One chronological split (no cross-validation); the test period is about 4.5 months.
* The dataset looks synthetic/aggregated rather than raw sensor data (43% of congestion values are exactly 100); this is
  an inference, not a verified fact.
* An input window is the last 7 **observations**, not 7 calendar days (57% of windows span more than 7 days; the gap is a
  feature). Samples exist only when day t+1 was observed, and days whose next day is missing have lower congestion
  (75.1 vs 83.9), so the evaluation is conditional on that. Results are similar in every window-span bucket.
* LSTM numbers come from one training run; differences between LSTM, XGBoost and the road-mean baseline are inside the
  bootstrap noise (see `model/evaluation_report.md`), so **no ranking among them is claimed**.
* The research paper (`doucments/Research Paper/`) and the slides (`doucments/ppt/`) were rewritten to match this
  implementation and the verified results. The originals, which quoted leaky results, hour-of-day / "rush hour"
  claims, model settings the code never used, and several unverifiable or mis-dated references, are kept untouched in
  `doucments/original_backup/` for reference only and must not be presented. Their figures come from
  `python -m src.modelTraining.make_document_figures`; `tests/test_documents.py` checks that the documents still match
  `model/metrics.json`. (The old code and `model/original_backup/model.h5` used 48-step windows; the documents never stated 48.)
* References in the paper were checked against Crossref (DOIs listed). The TomTom documentation and the Kaggle dataset
  page were only consulted as web pages; the dataset's source and licence remain unverified.
* See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the full methodology audit.

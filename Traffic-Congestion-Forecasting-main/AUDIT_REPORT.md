# Methodology audit

Scope: the split, baselines, LSTM windows, XGBoost features, scaler, metrics, documentation and provenance claims.
Method: code inspection plus an **independent recomputation from the raw CSV that does not import the project's pipeline**
(scratch scripts, read-only; model artifacts were hash-checked before/after the audit and are unchanged).

## Files inspected
`src/dataProcessing/forecast_data.py`, `src/modelTraining/{trainXgboost,trainLstm2,comparePerformancePlots,predictLstm,predictXgboost}.py`,
`src/app/{demo,realtime}.py`, `app.py`, `tests/*`, `model/*` (predictions, metrics, report, original_backup), `README.md`,
`notebooks/test.ipynb`, the research paper (.docx) and `Traffic-Flow-OptimizationLSTM-and-XGBoost.pptx` text.

## Verified correct
| Check | Result |
|---|---|
| Road-mean baseline uses **training targets only** (`comparePerformancePlots.py:48`, `demo.py:61`) | Confirmed; recomputed independently and equal to the file. Using all-data means would give RMSE 19.80 instead of 19.85 (the leak would be small, but it is not present) |
| Persistence baseline = congestion on day t | Confirmed (matches file); uses only day-t data |
| Chronological split by target date (`forecast_data.py:145-153`) | Independently rebuilt: 5,754 samples, 3,997 / 901 / 856, same boundary dates |
| Target features | No feature contains the target of day t+1. Old leaky features (`Congestion Level_rolling_mean3`, `congestion_to_travel_ratio`) exist in the ORIGINAL backup model and are absent from the current one |
| Windows never cross roads; window ends at day t | Confirmed by existing tests and the independent rebuild |
| Scaler fitted on training windows only | Confirmed (test compares scaler min/max with train-only min/max) |
| Same test rows for all models/baselines | Confirmed independently: identical Date, road and actual values in all three prediction files |
| Metrics | Recomputed from the CSVs with separate code: all 12 numbers equal `metrics.json` |
| Dataset identity | The downloaded CSV's min/max for all 6 numeric features equal the ORIGINAL scaler's (`original_backup/scaler.pkl`), and the 5 weather categories match: strong evidence it is the dataset the project was built on |
| Test target distribution | Similar to train (mean 83.4 vs 83.5; 49.6% vs 48.3% at the cap of 100): no obvious distribution shift |

## Confirmed issues
| # | Issue | Location | Severity |
|---|---|---|---|
| C1 | Provenance overstated: "Government of Karnataka open data, per Kaggle" was never read from the Kaggle page (the page fetch returned only a title); it comes from search-result summaries of third-party write-ups. Licence is unknown | `README.md:10`, `app.py:182,670`, `comparePerformancePlots.py:68`, `model/evaluation_report.*` | Documentation |
| C2 | Evaluation report contains hard-coded values: `rows: 8936`, `identical_test_rows_for_all_models: True` (a literal), "43%" and "~0.2" in the limitations | `comparePerformancePlots.py` (`write_report`) | Reproducibility |
| C3 | The ranking LSTM vs XGBoost vs road-mean is **not statistically supported**: cluster-bootstrap (by date, 4,000 resamples) 95% CIs of RMSE difference vs road-mean: LSTM +0.09 [-0.04, +0.21], XGBoost +0.23 [-0.10, +0.56]; LSTM minus XGBoost -0.14 [-0.50, +0.23]. The dashboard nevertheless announced a "Lowest test RMSE" winner | `app.py` (KPI block) | Evaluation reporting |
| C4 | My earlier README note (and summary) said the paper/slides describe a "48-step LSTM". They do not: 48 steps was in the old code and the old `model.h5` (`(None, 48, 11)`). What the documents really say that is wrong: R² 0.9995 / -0.0287 (leaky), "rush hour cycles" and an hour feature (data is daily, hour is always 0), LSTM layers 64/32 and `max_depth=6` (old code used 128/64 and depth 8) | `README.md` limitations; paper; slides 4 and 6 | Documentation |
| C5 | `notebooks/test.ipynb` cells 8-9 call the removed `predictLstm.predict_next_congestion(last_6)` API (now takes a road-history DataFrame) and would fail | notebook | Breakage |

## Potential issues (not errors in the evaluation; need a decision, not a fix)
| # | Issue | Evidence |
|---|---|---|
| P1 | An input window is the last **7 observations**, not 7 days. 56.8% of samples span > 7 days, 9.5% span > 14, 23.4% contain an internal gap > 3 days (the `days_since_prev` feature exposes gaps). Stricter reading of "don't cross time gaps" would need a different sample definition and retraining | Test RMSE by span with the existing predictions: ≤7 days 19.63/19.70/19.57 (LSTM/XGB/road-mean), 8-14 days 19.71/19.88/19.58, >14 days 21.89/22.16/21.87. The conclusion (models ≈ road mean) is the same in every stratum |
| P2 | Selection effect: a sample exists only if day t+1 was observed. Days whose next day is missing have lower congestion at t (75.1 vs 83.9) so evaluation is conditional on a future observation existing | Cannot be corrected with this data; should be disclosed |
| P3 | LSTM results come from one seed and TensorFlow is not bit-deterministic; a retrain may change the LSTM numbers slightly (not tested: no training in the audit) | Differences are inside the bootstrap noise anyway |
| P4 | XGBoost file holds 83 trees but predictions use the first 53 (`best_iteration + 1`); the app's feature-importance chart counts all 83. Predictions are consistent between training and inference | Minor; fixing would change the artifact |
| P5 | Only one chronological split (about 4.5 months of test data), no cross-validation | — |
| P6 | Real-time TomTom integration: untested against the live service, and cannot supply the model inputs | — |

## Which reported metrics remain trustworthy
All four rows of the results table are **valid, reproducible numbers** for the defined task (next-day congestion, chronological
test of 856 rows). What is *not* supported is any ranking among them: LSTM, XGBoost and the road-mean baseline are
statistically indistinguishable, and the persistence baseline is clearly worse. The old R² of 0.9995 / -0.0287 are invalid.

## Fixes applied (originals kept in `audit_backup/`)
* C1: provenance/licence wording corrected in README, dashboard, evaluation script and report.
* C2: report values are computed from the data (rows, % at cap, lag-1 correlation, window span, selection effect); the "identical rows" flag comes from the alignment check.
* C3: seeded cluster bootstrap CIs added to `metrics.json` / `evaluation_report.*`; the dashboard now says "No winner" when the CI of the LSTM-XGBoost RMSE difference includes 0 and shows the CI table.
* C4: README corrected (the documents do not state 48 steps; the old code/model did) and lists what the documents do get wrong.
* C5: notebook cell 9 updated to the current API.
* Tests: `tests/test_audit.py` (independent rebuild of samples/split/baselines/metrics, report content, bootstrap, notebook, dashboard message).
* No retraining was necessary and none was done: model files and the prediction CSVs are byte-identical (md5) before and after. Metrics are unchanged; the evaluation script was re-run only to add CIs and correct the report.

## Verification
* 38 automated tests pass (`python -m unittest discover -s tests`).
* Real-browser check in headless Chrome (`tests/browser_check.py`): 31/31 checks passed: all 6 tabs, historical replay, what-if with the Heavy-congestion preset, prediction log, Reset, and the unconfigured TomTom mode. Screenshots were inspected.
* Not covered by the browser run: input *validation* errors (the UI controls are range-limited, so invalid input cannot be typed; validation is covered by unit tests and AppTest), a configured TomTom key (none available), and other browsers/devices.

## Follow-up: research paper and slides updated
The paper and slides were rewritten after this audit (originals preserved in `doucments/original_backup/`):
* Removed the leaky R² values, all five old figures, the hour-of-day / rush-hour / peak-hour claims, and LSTM/XGBoost settings the code never used.
* References re-checked against Crossref: Lv et al. is 2015 and is a stacked-autoencoder paper (not LSTM), Zhao et al. is 2017; "Zheng 2020", "Li 2021" and "Xu 2022" could not be found and were removed together with the claims attributed to them; an unverifiable portal reference and an uncited Brownlee entry were removed. A real LSTM-XGBoost traffic paper (Zhang & Zhang, 2020) is cited instead, described only as far as its title supports.
* Both documents state that the road-mean baseline is best on the reported metrics and that neither model shows a supported improvement; the dataset provenance and licence, the TomTom interface and the local-mode PySpark are labelled as unverified / unverified / single-machine.
* `tests/test_documents.py` guards the documents against regressions (no leaky numbers, metrics equal `metrics.json`, citation/DOI consistency, embedded figures are the current ones).

# Evaluation report

Generated (UTC): 2026-10-28T13:51:37+00:00

* Dataset: Kaggle dataset preethamgouda/banglore-city-traffic-dataset (downloaded from Kaggle). Kaggle/third-party descriptions say it derives from Government of Karnataka open traffic data; this was NOT verified from an authoritative source. Licence not verified. 8936 rows, 16 roads, daily (no hour-of-day)
* Target: Congestion Level (0-100) of the same road on day t+1  |  horizon: 1 day  |  inputs: last 7 observations of the road (not 7 calendar days)
* Split (chronological, by target date): train to 2023-10-30, validation to 2024-03-20, test 2024-03-21 .. 2024-08-09 (856 samples, identical for every model: True)

| Model | MAE | RMSE | RMSE 95% CI | R2 |
|---|---|---|---|---|
| LSTM | 15.463 | 19.933 | [18.80, 21.07] | 0.200 |
| XGBoost | 15.931 | 20.074 | [18.96, 21.18] | 0.189 |
| Persistence baseline | 19.105 | 28.232 | [26.70, 29.70] | -0.605 |
| Road-mean baseline | 15.338 | 19.846 | [18.71, 20.98] | 0.207 |

RMSE differences (observed [95% CI]; cluster bootstrap by test date, 4000 resamples, seed 0):
* LSTM - Road-mean baseline: +0.09 [-0.04, +0.21]
* XGBoost - Road-mean baseline: +0.23 [-0.10, +0.56]
* LSTM - XGBoost: -0.14 [-0.50, +0.23]

No pairwise RMSE difference among LSTM, XGBoost and the road-mean baseline is statistically distinguishable from zero, so no ranking among them is supported.

A learned model is only useful here if it beats the road-mean baseline; it does not.

## Software
python 3.11.9, numpy 1.26.4, pandas 2.1.1, scikit-learn 1.6.1, tensorflow 2.16.1, keras 3.15.1, xgboost 3.2.0

## Artifacts (sha256 prefix)
model.h5: c635e002b88d, scaler.pkl: 06de6e2e9b93, xgboost_model.json: 965a03e5e6cd

## Limitations
* Single chronological split; no cross-validation; test period about 4.5 months.
* Daily, gappy data: samples exist only where the road was observed on days t and t+1.
* Selection effect: mean congestion on day t is 83.9 when day t+1 is observed vs 75.1 when it is missing, so evaluation is conditional on the next day being observed.
* 57% of input windows span more than 7 calendar days (a window is the last 7 observations; days_since_prev is a feature).
* Next-day congestion is only weakly predictable (lag-1 correlation 0.21).
* 43% of congestion values are exactly 100 (a cap); the data looks synthetic/aggregated, which is an inference, not a verified fact.
* LSTM numbers come from one training run (one seed); TensorFlow is not bit-deterministic.
* Predictions are clipped to the physical range [0, 100].
* No prediction intervals are produced; the bootstrap CIs above describe test-set RMSE uncertainty only.

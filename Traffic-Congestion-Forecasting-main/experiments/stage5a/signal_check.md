# Stage 5A signal-gate benchmark

Verdict: **NO_USABLE_SIGNAL**. No evidence of within-road, time-dependent signal beyond each road's usual level was found on this data. Training a new learned model on it is not justified; any learned model should be expected to match, not beat, the per-road mean.

Data: 8936 observations, 16 roads, 2022-01-01 to 2024-08-09. Samples: 5754 (the same road on the next calendar day). Seed 0, 2000 bootstrap resamples, 1000 permutations.

## Project split (target dates)

| split | first | last | samples |
|---|---|---|---|
| train | 2022-01-09 | 2023-10-30 | 3997 |
| val | 2023-10-31 | 2024-03-20 | 901 |
| test | 2024-03-21 | 2024-08-09 | 856 |

## Official test split (856 samples, 142 dates)

| candidate | MAE | RMSE | R2 | within-road corr | RMSE minus reference (95% CI) |
|---|---|---|---|---|---|
| global_train_mean | 18.591 | 22.285 | -0.000 | n/a | +2.469 [+1.736, +3.204] * |
| road_train_mean | 15.338 | 19.846 | 0.207 | n/a | +0.030 [-0.052, +0.111] |
| expanding_road_mean | 15.222 | 19.816 | 0.209 | -0.106 | reference |
| persistence | 19.105 | 28.232 | -0.605 | -0.036 | +8.416 [+7.074, +9.754] * |
| mean_last_7_obs | 15.656 | 20.951 | 0.116 | -0.007 | +1.135 [+0.641, +1.592] * |
| road_weekday_train_mean | 15.566 | 20.238 | 0.175 | -0.029 | +0.422 [+0.195, +0.696] * |
| saved_LSTM | 15.463 | 19.933 | 0.200 | -0.008 | +0.117 [-0.017, +0.246] |
| saved_XGBoost | 15.931 | 20.074 | 0.189 | 0.028 | +0.258 [-0.067, +0.572] |

Reference = expanding per-road mean. Positive difference = worse than the reference. `*` = CI excludes 0 after Bonferroni adjustment.

## Rolling-origin folds

| fold | train samples | test target dates | test samples |
|---|---|---|---|
| 1 | 2830 | 2023-04-26 to 2023-07-29 | 561 |
| 2 | 3391 | 2023-07-30 to 2023-11-01 | 614 |
| 3 | 4005 | 2023-11-02 to 2024-02-03 | 603 |
| 4 | 4608 | 2024-02-04 to 2024-05-07 | 572 |
| 5 | 5180 | 2024-05-08 to 2024-08-09 | 574 |

### Pooled over folds (2924 samples)

| candidate | MAE | RMSE | R2 | within-road corr | RMSE minus reference (95% CI) |
|---|---|---|---|---|---|
| global_train_mean | 18.302 | 22.067 | -0.000 | -0.001 | +2.236 [+1.854, +2.611] * |
| road_train_mean | 15.311 | 19.853 | 0.190 | -0.064 | +0.023 [-0.019, +0.065] |
| expanding_road_mean | 15.230 | 19.831 | 0.192 | -0.065 | reference |
| persistence | 18.803 | 27.822 | -0.590 | 0.012 | +7.992 [+7.319, +8.707] * |
| mean_last_7_obs | 15.783 | 21.172 | 0.079 | -0.018 | +1.342 [+1.074, +1.606] * |
| road_weekday_train_mean | 15.519 | 20.189 | 0.163 | -0.026 | +0.359 [+0.209, +0.502] * |

Reference = expanding per-road mean. Positive difference = worse than the reference. `*` = CI excludes 0 after Bonferroni adjustment.

## Signal diagnostics, development period (used for the gate)

Road-standardised values; exact calendar lags within the same road.

| lag (days) | pairs | r | 95% CI | useful |
|---|---|---|---|---|
| 1 | 4959 | 0.007 | [-0.020, 0.036] | False |
| 2 | 4932 | -0.016 | [-0.041, 0.009] | False |
| 3 | 4902 | -0.022 | [-0.049, 0.005] | False |
| 7 | 4896 | -0.037 | [-0.062, -0.011] | False |
| 14 | 4875 | -0.001 | [-0.027, 0.026] | False |

| calendar effect | groups | adjusted R2 | permutation p | useful |
|---|---|---|---|---|
| weekday | 7 | -0.0006 | 0.945 | False |
| month_of_year | 12 | 0.0002 | 0.315 | False |
| calendar_date | 810 | -0.0097 | 0.973 | False |
| road_month | 432 | -0.0088 | 0.987 | False |

## Signal diagnostics, all data (descriptive only)

Road-standardised values; exact calendar lags within the same road.

| lag (days) | pairs | r | 95% CI | useful |
|---|---|---|---|---|
| 1 | 5815 | 0.003 | [-0.021, 0.028] | False |
| 2 | 5785 | -0.018 | [-0.043, 0.009] | False |
| 3 | 5750 | -0.021 | [-0.046, 0.004] | False |
| 7 | 5743 | -0.033 | [-0.057, -0.007] | False |
| 14 | 5720 | -0.006 | [-0.030, 0.018] | False |

| calendar effect | groups | adjusted R2 | permutation p | useful |
|---|---|---|---|---|
| weekday | 7 | -0.0003 | 0.799 | False |
| month_of_year | 12 | -0.0002 | 0.604 | False |
| calendar_date | 952 | -0.0059 | 0.865 | False |
| road_month | 512 | -0.0086 | 0.993 | False |

## Methodology

* **reference**: expanding_road_mean (mean of every observation of the road up to and including day t)
* **no_tuning**: no parameter is fitted or selected on any labels
* **rolling_origin**: expanding window by target date; first 50% of dates are training only; remaining dates split into contiguous blocks; at most 5 folds, fewer when fewer than 100 test samples per fold are available
* **diagnostics_period**: observations dated up to 2024-03-20 (end of the validation split); test labels are not used for the gate
* **bonferroni**: 9 diagnostic tests; 5 baseline comparisons (+ saved models on the official split)

## Limitations

* Results describe THIS dataset only. Its provenance and measurement process are unverified, so nothing here is a statement about real-world traffic.
* About 43% of target values are exactly 100 (a cap); RMSE and MAE rank methods differently because of it.
* Each calendar day has only a few roads observed (about 9 of 16), and 35% of road days are missing; samples need the next day to be observed.
* The rolling-origin evaluation uses simple fixed baselines, not learned models: it tests whether signal exists, not how well a model could use it.
* Tests have limited power: effects smaller than the thresholds above cannot be excluded.
* Saved LSTM/XGBoost predictions are only read, and only the official test split is available for them.

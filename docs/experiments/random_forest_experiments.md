# Random Forest — Experiment Log

## Objective

Develop and improve a one-hour-ahead household electricity demand forecasting model using Random Forest. Evaluate performance against hourly persistence, daily naïve, weekly naïve, and the selected Ridge regression benchmark.

## Experimental Setup

- Training period: Through December 31, 2008 (16,626 observations).
- Validation period: January–December 2009 (8,124 observations).
- Held-out test period: 2010 (6,861 observations; evaluation pending).
- Evaluation metrics: MAE and RMSE (kW).
- Initial model: 100 trees, random_state=42, default max_depth and min_samples_leaf.
- Features: 57 encoded features inherited from Ridge, including one-hot hours, weekend interactions, cyclical day/month encoding, weekend indicator, lag_1, lag_24, lag_168 and rolling_mean_24.
- No StandardScaler required. Forecasts use actual readings available through the previous hour.
- Source: saved random_forest.ipynb results and ridge_regression.md; no new training performed.

## Experiment Summary

| Experiment | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| E00 — Initial Random Forest (100 trees) | 0.368301 | 0.534288 |
| E00 — Tree count tuning (250 trees) | 0.367176 | 0.532018 |
| E01 — Maximum depth tuning (10) | 0.363694 | 0.529217 |
| E02 — Minimum leaf size tuning (2; 500 trees) | 0.363130 | 0.528875 |
| E03 — Selected configuration (250 trees) | 0.363227 | 0.528939 |
| E04 — Peak and residual analysis (no model change) | 0.363227 | 0.528939 |
| E05-A — Original rolling mean (matched control) | 0.364402 | 0.530837 |
| E05-B — Recent rolling mean (replacement) | 0.363663 | 0.530372 |
| E05-C — Both rolling means | 0.363424 | 0.529835 |
| E05 — Decision: retain E03 | 0.363227 | 0.528939 |
| E06 — Three-hour rolling mean | 0.359180 | 0.523998 |
| E07 — Six-hour rolling mean | Pending | Pending |


## E00 — Initial Random Forest Baseline and Tree Count Tuning

**Objective:** Establish a nonlinear forecasting benchmark using the final Ridge feature set.

**Hypothesis:** Averaging multiple trees improves prediction stability, with diminishing returns as tree count increases.

**Change:** Replace Ridge and StandardScaler with RandomForestRegressor; vary n_estimators.

Fixed seed 42; default depth and leaf settings. Validation results:

| Trees | MAE | RMSE |
|---:|---:|---:|
| 25 | 0.372839 | 0.541400 |
| 50 | 0.369204 | 0.536283 |
| 100 | 0.368301 | 0.534288 |
| 250 | 0.367176 | 0.532018 |
| 500 | 0.366970 | 0.531900 |
| 750 | 0.366550 | 0.531448 |
| 900 | 0.366575 | 0.531362 |

Decision: 250 trees; diminishing returns beyond this point. 1000 was discussed but not tested in saved notebook. Training times were not recorded. Seed sensitivity was discussed but no saved sweep was found.

## E01 — Maximum Depth Tuning

**Hypothesis:** Limiting tree depth improves generalization by reducing overly specific splits.

**Change:** Vary max_depth while holding tree count and seed fixed.

250 trees, seed 42, default leaf size 1.

| Max depth | MAE | RMSE |
|---:|---:|---:|
| 5 | 0.380949 | 0.547710 |
| 10 | 0.363694 | 0.529217 |
| 15 | 0.362247 | 0.529392 |
| 20 | 0.364184 | 0.530883 |

Decision: depth 10 for lowest RMSE; depth 15 has slightly lower MAE. Duplicate fit call in loop should be removed.

## E02 — Minimum Leaf Size Tuning

**Hypothesis:** Requiring more observations per leaf reduces noisy predictions.

**Change:** Vary min_samples_leaf.

500 trees, depth 10, seed 42 (note mismatch with selected 250 trees).

| Min samples leaf | MAE | RMSE |
|---:|---:|---:|
| 1 | 0.363572 | 0.529215 |
| 2 | 0.363130 | 0.528875 |
| 5 | 0.363826 | 0.529209 |
| 10 | 0.365676 | 0.530597 |

Decision: leaf size 2; recheck at 250 trees. Remove duplicate fit call.

## E03 — Selected Random Forest Configuration

**Objective:** Evaluate the chosen combination of hyperparameters against naive baselines.

```python
RandomForestRegressor(n_estimators=250, random_state=42,
                      max_depth=10, min_samples_leaf=2)
```

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Random Forest | 0.363227 | 0.528939 |
| Hourly Naive | 0.431716 | 0.658933 |
| Weekly Naive | 0.565109 | 0.840827 |
| Daily Naive | 0.609805 | 0.899521 |

RF wins on 52.04% of individual observations versus hourly naive. No negative predictions; clipping has no effect. Prediction range 0.225250–4.308702; actual range 0.179733–5.233800. Overall mean residual −0.026330.

## E04 — Peak Demand and Residual Analysis

**Objective:** Identify remaining forecasting weaknesses using hourly and high-demand validation slices.

### Hourly and peak diagnostics

| Slice | RF MAE | Hourly naive MAE |
|---|---:|---:|
| 7 a.m. weekday | 0.379523 | 0.917641 |
| 7 a.m. weekend | 0.480389 | 0.614627 |
| 7 a.m. overall | 0.408384 | 0.830938 |
| 7 p.m. overall | 0.617045 | 0.684564 |
| 8 p.m. overall | 0.607482 | 0.687243 |

Top 5% validation demand: threshold 2.73709 kW, 407 observations. This threshold is descriptive, not a training threshold.

| Model | MAE | RMSE | Mean residual | Underprediction % |
|---|---:|---:|---:|---:|
| RF | 1.112740 | 1.317517 | 1.048849 | 90.663391 |
| Hourly naive | 1.139414 | 1.379434 | 0.871905 | 78.869779 |
| Weekly naive | 1.560551 | 1.842291 | 1.384659 | 88.452088 |
| Daily naive | 1.634815 | 1.904740 | 1.511210 | 90.909091 |

Top 10 actual peaks: RF MAE 2.394196, RMSE 2.492217; hourly naive MAE 1.830477, RMSE 2.178000. All 10 were underpredicted by RF. Worst actual peak 2009-02-11 18:00: actual 5.233800, RF 1.272213, lag_1 1.013400 kW. The preceding observed hour suggests an abrupt increase if lag alignment is confirmed. Adjacent peaks can belong to the same event.

Actual-demand quintile mean residuals (very low to very high): −0.193289, −0.259227, −0.159661, −0.009005, +0.489758 kW. Overall bias masks underprediction in high-demand observations. Conditioning on actual demand can itself create residual patterns, so interpret alongside peak metrics and residuals vs predictions.

### Comparison with Ridge

| Metric | Ridge | Random Forest |
|---|---:|---:|
| Overall MAE | 0.370864 | 0.363227 |
| Overall RMSE | 0.532377 | 0.528939 |
| Weekday 7 AM MAE | 0.472196 | 0.379523 |
| Weekend 7 AM MAE | 0.562860 | 0.480389 |
| Top 5% MAE | 1.105884 | 1.112740 |
| Top 5% RMSE | 1.308388 | 1.317517 |
| Top 10 MAE | 2.225080 | 2.394196 |
| Top 10 RMSE | 2.355818 | 2.492217 |

**Conclusion:** Random Forest improves overall and morning accuracy relative to the reported Ridge scores, but Ridge performs better on extreme demand. Confirm identical evaluation timestamps before attributing differences solely to model choice.

**Decision:** Retain E03 as the current control; prioritize feature engineering before further tuning. Final test evaluation remains pending.

## E05 — Recent Rolling Average

**Objective:** Determine whether more recent consumption context improves one-hour-ahead Random Forest forecasts, particularly during high demand.

**Hypothesis:** The most recent 24 hours may provide more useful context than the older 24-hour window.

**Change:** Added `rolling_mean_24_recent = power.shift(1).rolling(24).mean()` while preserving `rolling_mean_24 = power.shift(24).rolling(24).mean()`. The original window spans t-47 through t-24; the recent window spans t-24 through t-1. Both use only historical readings.

**Experimental setup:**

- E05-A: original rolling mean only (control).
- E05-B: recent rolling mean replaces original.
- E05-C: both rolling means.
- Fixed configuration: 250 trees, depth 10, minimum leaf size 2, seed 42; existing calendar encoding retained.
- Shared complete-case dataset across all three variants; same chronological training/validation boundaries as E03. Original dataset: 31,611 rows, 9 columns including target; matched dataset: 30,316 rows, 10 columns including target. Loss: 1,295 rows (4.1%).
- Validation: 7,787 observations (337 fewer than E03). Ridge predictions were aligned by timestamp from the existing benchmark, which was trained on the larger original training set.
- 2010 test evaluation remains pending.

**Overall results:**

| Variant | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| E05-A | 0.364402 | 0.530837 |
| E05-B | 0.363663 | 0.530372 |
| E05-C | 0.363424 | 0.529835 |

### High-demand analysis

Top 5% of actual demand on the matched validation sample: threshold 2.73758 kW, 390 observations. Residual = actual minus predicted.

| Model | MAE (kW) | RMSE (kW) | Mean residual (kW) | Underprediction (%) |
|---|---:|---:|---:|---:|
| E05-A | 1.110474 | 1.319125 | 1.048854 | 91.025641 |
| E05-B | 1.103409 | 1.316183 | 1.045677 | 91.282051 |
| E05-C | 1.108216 | 1.317870 | 1.054768 | 91.282051 |
| Ridge | 1.105974 | 1.311942 | 1.024707 | 89.743590 |
| Hourly Naive | 1.136544 | 1.377570 | 0.861632 | 78.717949 |
| Weekly Naive | 1.580572 | 1.860126 | 1.406933 | 88.717949 |
| Daily Naive | 1.638781 | 1.913986 | 1.534504 | 91.282051 |

**Key findings:**

- E05-C achieved the lowest overall MAE and RMSE, reducing them by approximately 0.27% and 0.19% versus E05-A.
- E05-B achieved the lowest high-demand MAE among all evaluated models; Ridge achieved the lowest high-demand RMSE.
- E05-B and E05-C underpredicted 356 of 390 high-demand observations, versus 355 for E05-A and 350 for Ridge. None meaningfully resolved underprediction.
- All three Random Forest variants performed similarly. No uncertainty analysis or repeated-seed evaluation was conducted, so the small differences do not establish a robust improvement.
- Requiring both rolling windows reduced data availability. This is a property of the matched complete-case sample; the recent-only model could have different coverage if evaluated independently.
- This experiment compares rolling-window choices; it does not establish whether rolling averages are useful versus removing them entirely.

**Decision:** Retain E03 with the original rolling mean as the selected Random Forest configuration. Preserve the recent rolling feature and E05 results for future reference. Do not adopt both windows solely on these marginal gains.

**Conclusion:** More recent 24-hour context produced only modest improvements and did not resolve peak forecasting errors. Shorter windows and directional change features may capture information that full-day averages smooth away. This is a hypothesis for future experiments, not a demonstrated result.

**Limitations:** E05 top-10 peak and weekday/weekend hourly metrics were not run or recorded. Ridge is an aligned evaluation benchmark, not a matched-training ablation. Peak subsets are selected retrospectively and do not measure prospective peak detection.

## Next Experiments

- E06 — Three-hour rolling mean: test `power.shift(1).rolling(3).mean()` against a matched E03 control.
- E07 — Six-hour rolling mean: test `power.shift(1).rolling(6).mean()`; assess whether a longer short-term window provides useful context.
- Final model selection and held-out 2010 test evaluation.

## Limitations and Reproducibility

- All reported results are saved validation results, not newly executed runs. Training times were not recorded.
- Missing hourly readings are masked before lag construction. Confirm the hourly index is regular; row-based shifts require this. New features can change eligible timestamps, so use matched samples for comparisons.
- Remove duplicate fit calls in tuning loops and correct the stale Ridge plot title.
- High-demand subsets are selected retrospectively; they do not measure prospective peak detection or false alarm rates.
- The dataset represents household demand, not transformer-level load. One-hour-ahead forecasts update throughout the day; fixed 24-hour forecasts are a separate future task.
- Repeated tuning on 2009 can overfit validation. Freeze model choices before held-out testing.

## E06 — Three-Hour Rolling Average

**Objective:** Determine whether recent household electricity consumption improves one-hour-ahead forecasting accuracy.

**Hypothesis:** A three-hour rolling average provides useful information about recent demand patterns beyond the most recent hourly reading.

**Change:** Added `rolling_mean_3`, calculated using the three most recent completed hourly observations.

```python
features["rolling_mean_3"] = (
    power.shift(1).rolling(3).mean()
)
```

**Experimental setup:**

- E06-A: Original Random Forest features (control).
- E06-B: Original features plus `rolling_mean_3`.
- Both models trained and evaluated on identical eligible timestamps.
- Random Forest configuration: 250 trees, maximum depth 10, minimum leaf size 2, random state 42.
- Validation period: 2009.
- High-demand threshold: 2.733877 kW (top 5%, 405 observations).

**Overall validation results:**

| Model | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| E06-A — Control | 0.362005 | 0.527512 |
| E06-B — Three-hour mean | **0.359180** | **0.523998** |

**High-demand results:**

| Model | MAE (kW) | RMSE (kW) | Underprediction rate |
|---|---:|---:|---:|
| E06-A | 1.102023 | 1.310357 | 90.12% |
| E06-B | **1.096234** | **1.305319** | 90.86% |
| Hourly Naive | 1.135787 | 1.375485 | 78.52% |

**Key findings:**

- Adding the three-hour rolling mean reduced overall MAE by approximately 0.78% and RMSE by 0.67%.
- High-demand MAE improved by approximately 0.53%.
- High-demand underprediction frequency increased slightly, despite smaller average prediction errors.
- The shorter rolling window provided more useful predictive information than the additional 24-hour rolling window tested in E05, based on the respective within-experiment comparisons.

**Decision:** Provisionally retain `rolling_mean_3` for Random Forest. Evaluate its contribution alongside the upcoming demand-change experiment before final model selection.

**Next experiment:** E07 — Hourly Demand Change.

**Test-set status:** The 2010 test set remains held out from model selection.
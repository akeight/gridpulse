# Ridge Regression — Experiment Log

## Objective

Develop and improve a one-hour-ahead household electricity demand forecasting model using Ridge regression. Evaluate performance against hourly persistence, daily naïve, and weekly naïve baselines.

## Experimental Setup

- Training period: Through December 31, 2008
- Validation period: January–December 2009
- Held-out test period: 2010
- Evaluation metrics: MAE and RMSE (kW)
- Initial regularization strength: alpha = 1.0

## Experiment Summary

| Experiment                          |      MAE |          RMSE |
|-------------------------------------|---------:|--------------:|
| E00 — Initial Ridge Baseline        |   0.5196 |        0.6879 |
| E01 — Ridge with hourly lag         | 0.396234 |      0.564386 |
| E02 — Cyclical weekend interactions | 0.395167 |      0.563180 |
| E03 — One-hot hour encoding         | 0.375341 |      0.536316 |
| E04 — One-hot weekend interactions  | 0.370890 |      0.532393 |
| E05 — Nonnegative clipping          | 0.370864 |      0.532377 |

## E00 — Initial Ridge Regression Baseline

**Objective:** Establish an initial machine learning forecasting benchmark and evaluate whether Ridge regression improves upon daily and weekly naïve forecasting methods.

**Features:**

- Cyclical hour, day-of-week, and month encoding
- Weekend indicator
- lag_24
- lag_168
- rolling_mean_24

**Model configuration:**

- StandardScaler
- Ridge regression
- alpha = 1.0
- Validation observations: 8,140

**Results:**

| Model | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Ridge | 0.5196 | 0.6879 |
| Weekly Naïve | 0.5657 | 0.8420 |
| Daily Naïve | 0.6111 | 0.9015 |

**Key findings:**

- Ridge outperformed both seasonal naïve baselines.
- The model struggled to predict extreme demand peaks.
- Ridge underpredicted all ten highest-demand validation observations.
- On February 11, 2009, at 6 PM, actual demand reached 5.23 kW while Ridge predicted 1.83 kW.

**Conclusion:** The initial Ridge model demonstrated that combining historical demand and calendar features improved forecasting accuracy. However, the absence of recent hourly consumption information limited its ability to respond to rapid demand changes.

**Next experiment:** Introduce lag_1 to evaluate whether the most recent hourly demand observation improves one-hour-ahead forecasting.

**Comparison note:** Adding lag_1 changes which observations have complete features. Subsequent experiments may use a different validation sample. A controlled feature ablation on identical timestamps is required to isolate the exact improvement attributable to lag_1.

## E01 — Hourly Lag Feature

**Hypothesis:** Recent electricity consumption provides useful information for next-hour demand forecasting.

**Change:** Added lag_1 to the historical demand features.

**Result:** Ridge outperformed hourly persistence on overall MAE and RMSE but struggled with weekday morning demand peaks.

## E02 — Cyclical Weekend Interactions

**Hypothesis:** Electricity demand follows different hourly patterns on weekdays and weekends.

**Change:** Added interactions between cyclical hour features and weekend status.

**Result:** Marginal overall improvement. Weekday 7 AM MAE remained relatively high at 0.732490 kW.

## E03 — One-Hot Hour Encoding

**Hypothesis:** Independent hourly indicators may better capture sharp demand transitions than smooth cyclical hour features.

**Change:** Replaced cyclical hour encoding and its weekend interactions with 24 binary hour indicators.

**Result:** Substantial improvement in overall accuracy. Weekday 7 AM MAE decreased to 0.524111 kW, but weekend 7 AM MAE increased to 0.660555 kW.

## E04 — One-Hot Weekend Interactions

**Hypothesis:** Separate weekday and weekend hourly effects may address the morning performance trade-off.

**Change:** Added 24 interactions between hourly indicators and weekend status.

**Result:** Overall accuracy improved. Weekday 7 AM MAE decreased to 0.472196 kW, and weekend 7 AM MAE decreased to 0.562860 kW.

## E05 — Nonnegative Prediction Clipping

**Objective:** Enforce physically valid, nonnegative electricity demand forecasts.

**Change:** Applied np.maximum(predictions, 0) while retaining raw predictions for diagnostics.

**Result:** Eliminated seven negative predictions. The effect on overall MAE and RMSE was negligible.

## Next Experiments

- Ridge regularization strength (alpha)
- Peak demand error analysis
- Final model selection and held-out test evaluation
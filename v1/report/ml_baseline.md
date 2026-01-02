# ML Baseline Report

## Target
- age_standardized_suicide_rate_2021

## Features
- gbd_depression_dalys_rate_both (DALYs rate)
- gbd_addiction_death_rate_both (Deaths rate)
- gbd_selfharm_death_rate_both (Deaths rate; mean of male/female)
- region_name, income_group, data_quality (one-hot)

## Data preparation
- Aggregated to one row per country (mean across age groups) to align with age-standardized target.
- Rows used: 183 countries

## Holdout Results (25% test)
| model | mae | r2 |
| --- | --- | --- |
| Ridge | 2.4822 | 0.7709 |
| RandomForest | 2.9645 | 0.6217 |

## Cross-validation (5-fold)
| model | mae_mean | mae_std | r2_mean | r2_std |
| --- | --- | --- | --- | --- |
| Ridge | 2.1472 | 0.3163 | 0.7538 | 0.0441 |
| RandomForest | 2.3260 | 0.4623 | 0.6756 | 0.0503 |

## Outputs
- v1/report/ml_baseline_results.csv
- v1/report/ml_baseline_cv.csv
- v1/data_clean/ml_baseline_predictions.csv
- v1/data_clean/ml_baseline_features.csv
- v1/report/ml_feature_importance.csv
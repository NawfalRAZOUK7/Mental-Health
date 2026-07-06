# Advanced ML (real data): tuned LightGBM + SHAP + conformal

Countries: 183. Target: WHO age-standardized suicide rate.

## Nested cross-validation (honest, tuning inside folds)
- LightGBM MAE = 4.18 (+/- 0.54)
- LightGBM R2  = 0.105 (+/- 0.209)

## SHAP — top drivers
- life_expectancy_years, alcohol_litres_per_capita, addiction_death_rate, gdp_per_capita_usd, urban_population_pct

## Conformal prediction intervals (target 90%)
- Interval half-width: +/- 8.49 per 100k
- Empirical coverage: 95.6%

## Outputs
- ml_advanced_metrics.csv, ml_advanced_shap.csv, ml_advanced_conformal.csv
- report_latex/figures/fig_v1_shap_summary.png
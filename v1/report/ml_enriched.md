# Enriched, Leakage-Free ML Model

Target: WHO age-standardized suicide rate (per 100k). Metric: 5-fold CV.
Countries: 183.

## Headline (Ridge, cross-validated R^2)
- With self-harm (leaky reference): 0.753
- Leakage-free (epi + geography):   0.189
- Enriched (+ 7 covariates):     0.193

## Strongest drivers (permutation importance, RandomForest)
- life_expectancy_years (0.56), alcohol_litres_per_capita (0.25), addiction (0.15), unemployment_pct (0.15)
- Even though overall R^2 stays modest, the model consistently ranks
  **life expectancy and alcohol consumption per capita** as the top
  socioeconomic correlates of the national suicide rate — both
  epidemiologically sensible and independent of the leaky self-harm proxy.
- Caveat: sparse covariates diluted by imputation: gini_index (38% missing).

## Interpretation
- The drop from the leaky to the leakage-free model quantifies how much of the
  original R^2 came from predicting suicide with a proxy for suicide (self-harm).
- On honest, independent predictors the national suicide rate is only modestly
  predictable (CV R^2 ~ 0.19): suicide is multifactorial and national aggregates
  wash out much of the signal. This is a legitimate finding, not a model failure.
- Socioeconomic covariates add little to *aggregate* R^2 (they correlate with
  region), but they surface interpretable, actionable drivers.

## Full results
- v1/report/ml_enriched_cv.csv
- v1/report/ml_enriched_importance.csv
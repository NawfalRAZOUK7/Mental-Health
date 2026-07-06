# Hierarchical (Mixed-Effects) Model

Countries: 183. Grouping: WHO region (partial pooling).
Predictors standardized (z-scores); coefficients are per 1 SD.

## Variance decomposition (null model)
- Between-region variance: 8.65
- Within-region (residual) variance: 30.47
- **ICC = 0.22** — 22% of the variance in national suicide rates is
  *between* WHO regions (a moderate but clear geographic-clustering effect),
  which is why flat models lean on region as a predictor.

## Fixed effects — strongest standardized associations
- life_expectancy_years (-2.64), health_exp_per_capita_usd (+1.54), unemployment_pct (+1.18), alcohol_litres_per_capita (+1.15)
- Signs are epidemiologically interpretable (e.g. alcohol positive, life
  expectancy negative), consistent with the SHAP analysis.

## Why this matters
- A mixed model is the statistically correct treatment of nested (country-in-region)
  data and separates 'where' (region) from 'what' (covariates) instead of conflating them.

## Outputs
- v1/report/hierarchical_model.csv
# Spatial autocorrelation (feature-space similarity graph)

**Neighbours are defined in feature space** (the k-NN country-similarity network), not by geography — this is network autocorrelation.

- Countries analysed: 183; permutations: 999.
- Moran's I (suicide rate): 0.2089 (p=0.001). Significant positive autocorrelation in suicide_rate (I=0.209, p=0.001): countries with similar profiles have similar values — they cluster.
- Moran's I (model residual): 0.0027 (p=0.415). No significant network autocorrelation in model_residual (p=0.415) — similar-profile countries are not more alike than chance.
- Significant LISA clusters: 5 (High-High: 2, Low-Low: 3).
- Spatial-lag regression: OLS R²=0.272 vs pseudo-R²=0.2577 (rho=0.281).

## High-High clusters (high residual, high-residual neighbours)
- Belarus (BLR): residual +0.6
- Somalia (SOM): residual +2.7

## Low-Low clusters (low residual, low-residual neighbours)
- Guatemala (GTM): residual -1.8
- Lebanon (LBN): residual -3.3
- Portugal (PRT): residual -1.7

## Outputs
- spatial_model.csv (per-country residual + LISA quadrant)
- spatial_model.json (Moran's I + regression summary)
- report_latex/figures/fig_v1_spatial_lisa.png
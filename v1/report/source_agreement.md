# Source Agreement: WHO vs IHME

Two independent global bodies measure suicide/self-harm. This checks how closely
the WHO age-standardized suicide rate and the IHME self-harm death rate agree,
and documents why self-harm must not be used as a 'predictor' of suicide.

- Countries compared: 183
- Pearson r: 0.844 (R² = 0.712)
- Spearman rho: 0.864
- Mean difference (WHO − IHME): 0.49 per 100k
- Limits of agreement: [-6.02, 7.00] per 100k
- Median absolute difference: 1.45 per 100k

**Interpretation.** The two sources agree strongly and with little systematic bias,
confirming they measure essentially the same phenomenon. Consequently, the enriched
ML model (see 11_ml_enriched.py) excludes self-harm as a feature and predicts the
suicide rate from *independent* drivers (depression burden, addiction burden,
geography, and socioeconomic covariates).

## Outputs
- v1/report/source_agreement_metrics.csv
- report_latex/figures/fig_v1_source_agreement.png
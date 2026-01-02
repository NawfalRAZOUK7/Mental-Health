# Synthetic Generation Notes

This dataset is synthetic and intended for demonstration only.

- Seed: 42
- Generator version: 1.0
- Years: 2000-2023
- Base distributions from v1 merged_ml_country.csv (WHO + GBD).
- Missing base values are filled with median before synthesis.
- Trends include region bias, latent risk, and Gaussian noise.
- Age/sex multipliers and income-scaled population growth add variation.
- Both-sex rows are population-weighted averages of male/female.
- Country/region outputs are population-weighted aggregates from the long table.
- Missingness injected: ~3% per feature column.
- risk_index is the mean z-score of the core feature columns.

Outputs:
- v2/data_clean/synth_long.csv
- v2/data_clean/synth_country_year.csv
- v2/data_clean/synth_region_year.csv

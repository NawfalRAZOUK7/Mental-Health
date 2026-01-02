# v1 Clean Data + Dashboard

Validated pipeline using WHO 2021 + GBD 2023.

## Contents
- `v1/data_clean`: cleaned tables + merged ML dataset
- `v1/report`: data model, dictionary, ML baseline, quality outputs
- `v1/notebooks`: analysis notebooks
- `v1/assets`: optional exported figures

## Rebuild (run in order)
```bash
MHP_VERSION=v1 python src/01_country_mapping.py
MHP_VERSION=v1 python src/02_clean_who.py
MHP_VERSION=v1 python src/03_clean_gbd.py
MHP_VERSION=v1 python src/04_merge_ml.py
MHP_VERSION=v1 python src/05_merge_context.py
MHP_VERSION=v1 python src/06_ml_baseline.py
MHP_VERSION=v1 python src/07_data_quality_scorecard.py
MHP_VERSION=v1 python src/08_segmentation_outliers.py
```

## Run
```bash
python scripts/run_app.py --version v1
```

# v2 Synthetic Version

This version uses synthetic data generated from v1 distributions for advanced analytics demos.

**Disclaimer:** Synthetic data for demonstration only. Do not use for real-world inference.

## Contents
- `v2/data_clean`: synthetic tables + cluster outputs
- `v2/report`: analytics outputs, notes, validation, forecasts
- `v2/notebooks`: v2 analysis + modeling notebooks
- `v2/assets`: optional exported figures

## Rebuild
```bash
python scripts/run_v2_pipeline.py
```

## Run
```bash
python scripts/run_app.py --version v2
```

## Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-v2.txt
```

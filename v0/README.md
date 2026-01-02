# v0 Static Visuals

This version is a static visual gallery. Charts are exported as PNG/HTML into `v0/assets` and surfaced in the v0 gallery page.

## Contents
- `v0/assets`: exported figures + `manifest.csv`
- `v0/notebooks`: v0 notebooks
- `v0/report`: notes

## Rebuild
```bash
MHP_VERSION=v0 python src/v0_visuals.py
```

## Run
```bash
python scripts/run_app.py --version v0
```

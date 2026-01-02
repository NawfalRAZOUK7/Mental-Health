# Project Versions

This repo supports multiple output versions while sharing `data_raw/` and `src/app.py`.

## Versions
- v0: simple static visuals (no web app)
- v1: validated pipeline (current)
- v2: synthetic/advanced demo (generated data)
- v3: interactive risk estimator (v1/v2 selectable)

## Layout
- `v0/`, `v1/`, `v2/` are version roots.
- `v*/data_clean/` and `v*/report/` store versioned outputs.
- `data_raw/` stays shared across all versions.

## How to run
| Version | Build outputs | Run dashboard |
| --- | --- | --- |
| v0 | `MHP_VERSION=v0 python src/v0_visuals.py` | `python scripts/run_app.py --version v0` |
| v1 | `python scripts/run_v1_pipeline.py` | `python scripts/run_app.py --version v1` |
| v2 | `python scripts/run_v2_pipeline.py` | `python scripts/run_app.py --version v2` |
| v3 | `python scripts/run_v3_pipeline.py` | `python scripts/run_app.py --version v3` |

## Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-v2.txt  # v2 advanced features
```

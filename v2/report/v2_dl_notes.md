# v2 DL Forecast Notes

Model: single-layer GRU/LSTM (PyTorch) for regional suicide-rate trends.
- Model type: GRU
- Lookback window: 5 years
- Hidden size: 32
- Layers: 1
- Epochs: 200
- Batch size: 64
- Learning rate: 0.001
- Train split: target years <= 2020
- Test split: target years 2021-2023

Outputs:
- v2/report/v2_dl_forecast_region.csv
- v2/report/v2_dl_metrics.csv

Notes:
- Model uses only suicide_rate time series (univariate).
- Inputs are z-scored using the training window mean/std.
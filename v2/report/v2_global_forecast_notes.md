# Global Forecasting (methods demo, synthetic data)

Series: 183 country suicide-rate trajectories (2000-2023). Test horizon: last 3 years. Lags: 5.

## Results (MAE, lower is better)
- Global N-BEATS (deep, PyTorch): 3.287
- darts N-BEATS (deep): 3.531
- darts LightGBM (global): 4.643
- Global LightGBM (all series): 4.81
- Naive drift (per series): 6.265

## Why this is the right upgrade
- A single global model learns shared temporal structure across all countries,
  instead of fitting a separate tiny model per series (the previous GRU approach).
- The naive-drift baseline is kept so the added complexity has to earn its place.
- N-BEATS global MAE = 3.287 (pure PyTorch, no darts).
- darts models: darts LightGBM (global)=4.643, darts N-BEATS (deep)=3.531

Synthetic data -- methodology demonstration only.
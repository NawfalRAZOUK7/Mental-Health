#!/usr/bin/env python3
"""Global forecasting across all country series (no heavy dependencies required).

Upgrade over the per-region toy GRU (v2_dl_forecast.py): instead of fitting one
tiny model per series, train a SINGLE global model jointly across all ~180 country
series using pooled lag features. Global models are the modern standard for
many-related-series forecasting (M4/M5) and are the right tool when each individual
series is short.

- Global model: one LightGBM trained on pooled (lag-window -> next-step) examples,
  then rolled out recursively per country. Needs only lightgbm (already a dep).
- Baseline: naive drift per series (must be beaten to justify the model).
- Optional deep models (opt-in, see flags below): a compact generic **N-BEATS in
  pure PyTorch** (needs only `torch`), plus library-backed **darts N-BEATS** and
  **darts TFT** (need `pip install -r requirements-deep.txt`). All skip gracefully
  if their dependencies are absent.

Flags:
- MHV_DEEP=1                 include the deep models (otherwise only baselines run).
- MHV_DEEP_BACKEND=torch|darts|all   which deep backend(s) to run (default "all").
    torch  -> pure-PyTorch N-BEATS only
    darts  -> darts N-BEATS + darts TFT only
    all    -> both (cross-check the two implementations)
  (The darts *LightGBM* global model runs whenever darts is installed, since it is
   cheap and needs no Lightning.)

NOTE: runs on the v2 SYNTHETIC panel -- the only longitudinal data available -- so it
is a methods demo, not a validated forecast. See v2/README.md.

Outputs (v2/report/): v2_global_forecast_metrics.csv, v2_global_forecast_notes.md
"""
from __future__ import annotations

import os

# LightGBM and PyTorch each bundle their own OpenMP runtime; loading both in one
# process on macOS can deadlock the first parallel torch op. Set BEFORE importing
# either library so torch/darts N-BEATS can train after LightGBM has run.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs

warnings.filterwarnings("ignore")
TEST_YEARS = 3
LAGS = 5
RNG = 42


def series_by_country(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for iso, g in df.groupby("iso3"):
        g = g.sort_values("year")
        vals = pd.to_numeric(g["suicide_rate"], errors="coerce").to_numpy(dtype=float)
        if not np.isnan(vals).any() and len(vals) >= LAGS + TEST_YEARS + 2:
            out[iso] = vals
    return out


def make_supervised(series: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (last LAGS values -> next value) examples, excluding the test horizon."""
    X, y = [], []
    for vals in series.values():
        last_train_target = len(vals) - TEST_YEARS  # exclusive
        for t in range(LAGS, last_train_target):
            X.append(vals[t - LAGS:t])
            y.append(vals[t])
    return np.asarray(X), np.asarray(y)


def recursive_forecast(model, history: np.ndarray, steps: int) -> np.ndarray:
    window = list(history[-LAGS:])
    preds = []
    for _ in range(steps):
        yhat = float(model.predict(np.asarray(window[-LAGS:]).reshape(1, -1))[0])
        preds.append(yhat)
        window.append(yhat)
    return np.asarray(preds)


def run_darts(series: dict[str, np.ndarray], lags: int, horizon: int, seed: int,
              run_deep: bool = False) -> dict[str, float]:
    """Forecast with darts. Returns {model_name: mae}.

    - darts LightGBMModel: global ML forecaster, needs only lightgbm (no Lightning).
    - darts NBEATSModel + TFTModel: deep models (only if run_deep), need
      torch + pytorch-lightning (pip install -r requirements-deep.txt).
    Raises ModuleNotFoundError if darts itself is not installed (caller treats as skip).
    """
    from darts import TimeSeries  # ModuleNotFoundError here => darts not installed

    train = [TimeSeries.from_values(v[:-horizon]) for v in series.values()]
    truth = [v[-horizon:] for v in series.values()]

    def mae_of(preds) -> float:
        return float(np.mean([mean_absolute_error(t, p.values().ravel()) for t, p in zip(truth, preds)]))

    results: dict[str, float] = {}

    try:  # darts global LightGBM (no Lightning needed)
        from darts.models import LightGBMModel

        m = LightGBMModel(lags=lags, output_chunk_length=horizon)
        m.fit(train)
        results["darts LightGBM (global)"] = mae_of(m.predict(n=horizon, series=train))
    except Exception:  # noqa: BLE001
        pass

    if not run_deep:
        return results  # skip the slow deep models unless explicitly requested

    try:  # darts N-BEATS (needs torch + pytorch-lightning)
        from darts.models import NBEATSModel

        nb = NBEATSModel(input_chunk_length=lags, output_chunk_length=horizon,
                         n_epochs=50, random_state=seed,
                         pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": True})
        nb.fit(train)
        results["darts N-BEATS (deep)"] = mae_of(nb.predict(n=horizon, series=train))
    except Exception:  # noqa: BLE001 - usually missing pytorch-lightning; skip quietly
        pass

    try:  # darts Temporal Fusion Transformer (needs torch + pytorch-lightning)
        import torch as _torch
        from darts.models import TFTModel

        # Univariate series with no covariates -> add_relative_index gives the model a
        # positional signal; likelihood=None + MSE keeps the forecast deterministic so
        # mae_of can read point values directly.
        tft = TFTModel(input_chunk_length=lags, output_chunk_length=horizon,
                       n_epochs=50, random_state=seed, add_relative_index=True,
                       likelihood=None, loss_fn=_torch.nn.MSELoss(),
                       pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": True})
        tft.fit(train)
        results["darts TFT (deep)"] = mae_of(tft.predict(n=horizon, series=train))
    except Exception:  # noqa: BLE001 - version/config or missing Lightning; skip quietly
        pass

    return results


def run_nbeats_torch(series: dict[str, np.ndarray], lags: int, horizon: int, seed: int) -> float:
    """Compact generic N-BEATS trained globally across all series. Needs only torch.

    Raises ModuleNotFoundError if torch is absent (caller treats that as 'skip').
    """
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)  # avoid OpenMP deadlock after LightGBM in same process
    torch.manual_seed(seed)

    # Global standardization from training portions (exclude the test horizon).
    train_vals = np.concatenate([v[:-horizon] for v in series.values()])
    mu, sd = float(train_vals.mean()), float(train_vals.std() + 1e-8)

    xs, ys = [], []
    for v in series.values():
        z = (v - mu) / sd
        train_part = z[:-horizon]
        for t in range(lags, len(train_part) - horizon + 1):
            xs.append(train_part[t - lags:t])
            ys.append(train_part[t:t + horizon])
    if len(xs) < 50:
        raise RuntimeError("not enough training windows for N-BEATS")

    x_t = torch.tensor(np.asarray(xs), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(ys), dtype=torch.float32)

    class Block(nn.Module):
        def __init__(self, size: int, hidden: int, forecast: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(size, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.backcast = nn.Linear(hidden, size)
            self.forecast = nn.Linear(hidden, forecast)

        def forward(self, x):
            h = self.body(x)
            return self.backcast(h), self.forecast(h)

    class NBeats(nn.Module):
        def __init__(self, size: int, forecast: int, n_blocks: int = 3, hidden: int = 128) -> None:
            super().__init__()
            self.forecast = forecast
            self.blocks = nn.ModuleList([Block(size, hidden, forecast) for _ in range(n_blocks)])

        def forward(self, x):
            residual = x
            total = x.new_zeros((x.size(0), self.forecast))
            for blk in self.blocks:
                bc, fc = blk(residual)
                residual = residual - bc
                total = total + fc
            return total

    model = NBeats(lags, horizon)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    epochs = 200
    for e in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
        if e == 1 or e % 40 == 0:
            print(f"    [N-BEATS/torch] epoch {e}/{epochs}  loss={loss.item():.4f}", flush=True)

    model.eval()
    errs = []
    with torch.no_grad():
        for v in series.values():
            z = (v - mu) / sd
            ctx = z[-horizon - lags:-horizon]  # the `lags` values just before the test window
            pred_z = model(torch.tensor(ctx.reshape(1, -1), dtype=torch.float32)).numpy().ravel()
            pred = pred_z * sd + mu
            errs.append(mean_absolute_error(v[-horizon:], pred))
    return float(np.mean(errs))


def main() -> None:
    ensure_dirs()
    path = DATA_CLEAN / "synth_country_year.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run src/v2_generate_synth.py first.")
    df = pd.read_csv(path)
    df = df[df["sex_name"] == "Both"].copy()
    series = series_by_country(df)
    if len(series) < 20:
        raise SystemExit("Not enough complete series for global forecasting.")

    X, y = make_supervised(series)
    gmodel = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, random_state=RNG, verbose=-1)
    gmodel.fit(X, y)

    global_err, naive_err = [], []
    for vals in series.values():
        history, actual = vals[:-TEST_YEARS], vals[-TEST_YEARS:]
        # Global model recursive forecast.
        global_err.append(mean_absolute_error(actual, recursive_forecast(gmodel, history, TEST_YEARS)))
        # Naive drift baseline.
        slope = (history[-1] - history[0]) / (len(history) - 1)
        naive = history[-1] + slope * np.arange(1, TEST_YEARS + 1)
        naive_err.append(mean_absolute_error(actual, naive))

    global_mae, naive_mae = float(np.mean(global_err)), float(np.mean(naive_err))
    rows = [
        {"model": "Global LightGBM (all series)", "mae": round(global_mae, 3), "n_series": len(series)},
        {"model": "Naive drift (per series)", "mae": round(naive_mae, 3), "n_series": len(series)},
    ]

    run_deep = bool(os.getenv("MHV_DEEP"))
    backend = os.getenv("MHV_DEEP_BACKEND", "all").strip().lower()
    if backend not in {"torch", "darts", "all"}:
        backend = "all"
    run_torch_deep = run_deep and backend in {"torch", "all"}
    run_darts_deep = run_deep and backend in {"darts", "all"}
    print(f"[{VERSION}] baselines done (global LightGBM={global_mae:.3f}, naive={naive_mae:.3f}).",
          flush=True)
    if run_deep:
        print(f"[{VERSION}] deep models ON (backend={backend}).", flush=True)

    # ---- Optional deep model: pure-PyTorch N-BEATS (MHV_DEEP=1 and backend in {torch,all}) ----
    nbeats_note = "N-BEATS (PyTorch) skipped; set MHV_DEEP=1 to include deep models."
    if run_torch_deep:
        print(f"[{VERSION}] training PyTorch N-BEATS (importing torch, may take ~20s first time)...",
              flush=True)
        try:
            nbeats_mae = run_nbeats_torch(series, LAGS, TEST_YEARS, RNG)
            rows.append({"model": "Global N-BEATS (deep, PyTorch)", "mae": round(nbeats_mae, 3),
                         "n_series": len(series)})
            nbeats_note = f"N-BEATS global MAE = {nbeats_mae:.3f} (pure PyTorch, no darts)."
        except ModuleNotFoundError:
            nbeats_note = "N-BEATS skipped (PyTorch not installed)."
        except Exception as exc:  # noqa: BLE001
            nbeats_note = f"N-BEATS skipped ({type(exc).__name__})."
    elif run_deep:
        nbeats_note = f"PyTorch N-BEATS skipped (backend={backend})."

    print(f"[{VERSION}] {nbeats_note}", flush=True)
    print(f"[{VERSION}] running darts models...", flush=True)

    # ---- Optional: darts models (global LightGBM always; darts N-BEATS/TFT only if run_darts_deep) ----
    darts_note = "darts skipped (not installed)."
    try:
        darts_results = run_darts(series, LAGS, TEST_YEARS, RNG, run_deep=run_darts_deep)
        for name, mae in darts_results.items():
            rows.append({"model": name, "mae": round(mae, 3), "n_series": len(series)})
        if darts_results:
            darts_note = "darts models: " + ", ".join(f"{k}={v:.3f}" for k, v in darts_results.items())
            if run_darts_deep and not any("deep" in k for k in darts_results):
                darts_note += " (darts N-BEATS/TFT need `pip install -r requirements-deep.txt`)."
        else:
            darts_note = "darts ran no model (check lightgbm install)."
    except ModuleNotFoundError:
        pass  # darts not installed
    except Exception as exc:  # noqa: BLE001
        darts_note = f"darts skipped ({type(exc).__name__})."

    metrics = pd.DataFrame(rows).sort_values("mae")
    metrics.to_csv(REPORT_DIR / "v2_global_forecast_metrics.csv", index=False)

    lines = [
        "# Global Forecasting (methods demo, synthetic data)",
        "",
        f"Series: {len(series)} country suicide-rate trajectories (2000-2023). "
        f"Test horizon: last {TEST_YEARS} years. Lags: {LAGS}.",
        "",
        "## Results (MAE, lower is better)",
    ]
    for r in metrics.itertuples():
        lines.append(f"- {r.model}: {r.mae}")
    lines += [
        "",
        "## Why this is the right upgrade",
        "- A single global model learns shared temporal structure across all countries,",
        "  instead of fitting a separate tiny model per series (the previous GRU approach).",
        "- The naive-drift baseline is kept so the added complexity has to earn its place.",
        f"- {nbeats_note}",
        f"- {darts_note}",
        "",
        "Synthetic data -- methodology demonstration only.",
    ]
    (REPORT_DIR / "v2_global_forecast_notes.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] global LightGBM MAE={global_mae:.3f} | naive MAE={naive_mae:.3f}")
    print(f"[{VERSION}] {nbeats_note}")
    print(f"[{VERSION}] {darts_note}")


if __name__ == "__main__":
    main()

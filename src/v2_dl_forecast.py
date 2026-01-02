#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "torch is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


LOOKBACK = 5
EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
MODEL_TYPE = "GRU"  # "GRU" or "LSTM"
FORECAST_YEARS = list(range(2024, 2031))
SEED = 42


@dataclass
class SeriesBundle:
    region_name: str
    years: list[int]
    values: np.ndarray


class RNNForecast(nn.Module):
    def __init__(self, model_type: str, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if model_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


def build_sequences(series: np.ndarray, years: list[int], lookback: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X, y, y_years = [], [], []
    for idx in range(lookback, len(series)):
        X.append(series[idx - lookback : idx])
        y.append(series[idx])
        y_years.append(int(years[idx]))
    return np.array(X, dtype=float), np.array(y, dtype=float), y_years


def compute_scale(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0:
        std = 1.0
    return mean, std


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_path = DATA_CLEAN / "synth_region_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["suicide_rate"] = pd.to_numeric(df["suicide_rate"], errors="coerce")
    df = df[(df["sex_name"] == "Both") & df["year"].notna()]

    bundles = []
    for region, group in df.groupby("region_name"):
        group = group.sort_values("year")
        years = group["year"].astype(int).tolist()
        values = group["suicide_rate"].to_numpy(dtype=float)
        if not years:
            continue
        full_years = list(range(min(years), max(years) + 1))
        series = pd.Series(values, index=years).reindex(full_years)
        series = series.interpolate(limit_direction="both").bfill().ffill()
        if series.isna().all():
            continue
        series = series.fillna(series.median())
        bundles.append(
            SeriesBundle(region_name=region, years=full_years, values=series.to_numpy(dtype=float))
        )

    if not bundles:
        raise SystemExit("No regional series available for DL forecasting.")

    X_list, y_list, year_list, region_list = [], [], [], []
    for bundle in bundles:
        X, y, years = build_sequences(bundle.values, bundle.years, LOOKBACK)
        if len(X) == 0:
            continue
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        if not np.any(valid_mask):
            continue
        X = X[valid_mask]
        y = y[valid_mask]
        years = [year for year, keep in zip(years, valid_mask) if keep]
        X_list.append(X)
        y_list.append(y)
        year_list.extend(years)
        region_list.extend([bundle.region_name] * len(years))

    if not X_list:
        raise SystemExit("No usable sequences after filtering missing values.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    year_arr = np.array(year_list)
    region_arr = np.array(region_list)

    train_mask = year_arr <= 2020
    test_mask = (year_arr >= 2021) & (year_arr <= 2023)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    if len(X_train) == 0:
        raise SystemExit("No training rows available after filtering.")

    scale_mean, scale_std = compute_scale(X_train.reshape(-1))
    X_train = (X_train - scale_mean) / scale_std
    y_train = (y_train - scale_mean) / scale_std
    X_test = (X_test - scale_mean) / scale_std

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RNNForecast(MODEL_TYPE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(xb)
        if epoch % 50 == 0:
            avg_loss = epoch_loss / max(1, len(train_ds))
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

    metrics_rows = []
    model.eval()
    with torch.no_grad():
        if len(X_test) > 0:
            test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
            preds_scaled = model(test_tensor).cpu().numpy()
            preds = preds_scaled * scale_std + scale_mean
            mae = float(np.mean(np.abs(preds - y_test)))
            rmse = float(math.sqrt(np.mean((preds - y_test) ** 2)))
        else:
            preds = np.array([])
            mae = float("nan")
            rmse = float("nan")

    metrics_rows.append(
        {"region_name": "Overall", "mae": mae, "rmse": rmse, "rows": int(len(y_test))}
    )

    if len(preds) > 0:
        for region in sorted(set(region_arr[test_mask])):
            mask = region_arr[test_mask] == region
            if not np.any(mask):
                continue
            region_true = y_test[mask]
            region_pred = preds[mask]
            metrics_rows.append(
                {
                    "region_name": region,
                    "mae": float(np.mean(np.abs(region_pred - region_true))),
                    "rmse": float(math.sqrt(np.mean((region_pred - region_true) ** 2))),
                    "rows": int(len(region_true)),
                }
            )

    metrics_path = REPORT_DIR / "v2_dl_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    forecast_rows = []
    for bundle in bundles:
        actual_years = bundle.years
        actual_values = bundle.values
        for year, value in zip(actual_years, actual_values):
            forecast_rows.append(
                {
                    "region_name": bundle.region_name,
                    "year": int(year),
                    "suicide_rate": float(value),
                    "type": "actual",
                    "model": "DL",
                }
            )

        history_scaled = (actual_values - scale_mean) / scale_std
        history = history_scaled.tolist()
        for year in FORECAST_YEARS:
            if len(history) < LOOKBACK:
                break
            window = np.array(history[-LOOKBACK:], dtype=float).reshape(1, LOOKBACK, 1)
            with torch.no_grad():
                pred_scaled = float(model(torch.tensor(window, dtype=torch.float32)).item())
            history.append(pred_scaled)
            pred = pred_scaled * scale_std + scale_mean
            forecast_rows.append(
                {
                    "region_name": bundle.region_name,
                    "year": int(year),
                    "suicide_rate": float(max(0.0, pred)),
                    "type": "forecast",
                    "model": "DL",
                }
            )

    forecast_path = REPORT_DIR / "v2_dl_forecast_region.csv"
    pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)

    notes = [
        "# v2 DL Forecast Notes",
        "",
        "Model: single-layer GRU/LSTM (PyTorch) for regional suicide-rate trends.",
        f"- Model type: {MODEL_TYPE}",
        f"- Lookback window: {LOOKBACK} years",
        f"- Hidden size: {HIDDEN_SIZE}",
        f"- Layers: {NUM_LAYERS}",
        f"- Epochs: {EPOCHS}",
        f"- Batch size: {BATCH_SIZE}",
        f"- Learning rate: {LEARNING_RATE}",
        "- Train split: target years <= 2020",
        "- Test split: target years 2021-2023",
        "",
        "Outputs:",
        f"- {forecast_path.relative_to(REPO_ROOT)}",
        f"- {metrics_path.relative_to(REPO_ROOT)}",
        "",
        "Notes:",
        "- Model uses only suicide_rate time series (univariate).",
        "- Inputs are z-scored using the training window mean/std.",
    ]
    notes_path = REPORT_DIR / "v2_dl_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"[{VERSION}] Wrote DL forecast to {forecast_path}")
    print(f"[{VERSION}] Wrote DL metrics to {metrics_path}")


if __name__ == "__main__":
    main()

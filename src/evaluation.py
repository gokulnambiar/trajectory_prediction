from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | int | str]] = []

    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        horizon_true = y_true[mask]
        horizon_pred = y_pred[mask]
        rows.append(
            {
                "model": model_name,
                "horizon": int(horizon),
                "rmse": float(np.sqrt(mean_squared_error(horizon_true, horizon_pred))),
                "mae": float(mean_absolute_error(horizon_true, horizon_pred)),
                "samples": int(mask.sum()),
            }
        )

    metrics_frame = pd.DataFrame(rows)
    summary = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    return metrics_frame, summary


def save_metrics(
    output_dir: Path,
    combined_metrics: pd.DataFrame,
    summary_metrics: dict[str, dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_metrics.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_metrics, handle, indent=2)

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.download_ngsim import download_ngsim_subset
from src.data_pipeline import prepare_datasets
from src.evaluation import save_metrics
from src.models import train_gradient_boosting, train_lstm
from src.plotting import save_error_plot, save_model_comparison_plot, save_trajectory_plot


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
DATA_PATH = PROJECT_ROOT / "data" / "ngsim_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
CONFIG_PATH = OUTPUT_DIR / "run_config.json"


def build_trajectory_preview(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    preview = metadata.copy()
    preview["true_x"] = y_true[:, 0]
    preview["true_y"] = y_true[:, 1]
    preview["pred_x"] = y_pred[:, 0]
    preview["pred_y"] = y_pred[:, 1]

    sample = (
        preview.sort_values(["vehicle_id", "history_end_time", "horizon"])
        .groupby("vehicle_id", as_index=False)
        .head(3)
        .head(12)
        .copy()
    )

    true_points = sample[["vehicle_id", "history_end_time", "horizon", "true_x", "true_y"]].rename(
        columns={"true_x": "x", "true_y": "y"}
    )
    true_points["series"] = "true"

    pred_points = sample[["vehicle_id", "history_end_time", "horizon", "pred_x", "pred_y"]].rename(
        columns={"pred_x": "x", "pred_y": "y"}
    )
    pred_points["series"] = f"predicted_{model_name.lower()}"

    trajectory_frame = pd.concat([true_points, pred_points], ignore_index=True)
    return trajectory_frame.sort_values(["vehicle_id", "history_end_time", "series", "horizon"]).reset_index(drop=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    return {"rmse": rmse, "mae": mae}


def main() -> None:
    history_window = 8
    horizons = [1, 3, 5]
    random_state = 7

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = download_ngsim_subset(DATA_PATH)
    split_dataset = prepare_datasets(dataset_path, history_window=history_window, horizons=horizons)

    metrics_rows: list[pd.DataFrame] = []
    model_summaries: dict[str, dict[str, float]] = {"GradientBoosting": {}, "LSTM": {}}
    prediction_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        train_mask = split_dataset.horizon_train == horizon
        test_mask = split_dataset.horizon_test == horizon

        x_tabular_train = split_dataset.x_tabular_train[train_mask]
        x_tabular_test = split_dataset.x_tabular_test[test_mask]
        x_sequence_train = split_dataset.x_sequence_train[train_mask]
        x_sequence_test = split_dataset.x_sequence_test[test_mask]
        y_train = split_dataset.y_train[train_mask]
        y_test = split_dataset.y_test[test_mask]
        metadata_test = split_dataset.metadata_test.loc[test_mask].reset_index(drop=True)

        gbr_outputs = train_gradient_boosting(
            x_train=x_tabular_train,
            y_train=y_train,
            x_test=x_tabular_test,
            random_state=random_state,
        )
        lstm_outputs = train_lstm(
            x_train=x_sequence_train,
            y_train=y_train,
            x_test=x_sequence_test,
            random_state=random_state,
        )

        metrics_rows.append(
            pd.DataFrame(
                [
                    {
                        "model": "GradientBoosting",
                        "horizon": horizon,
                        "samples": int(len(y_test)),
                        **regression_metrics(y_test, gbr_outputs.predictions),
                    },
                    {
                        "model": "LSTM",
                        "horizon": horizon,
                        "samples": int(len(y_test)),
                        **regression_metrics(y_test, lstm_outputs.predictions),
                    },
                ]
            )
        )

        prediction_frame = metadata_test.copy()
        prediction_frame["true_x"] = y_test[:, 0]
        prediction_frame["true_y"] = y_test[:, 1]
        prediction_frame["gbr_pred_x"] = gbr_outputs.predictions[:, 0]
        prediction_frame["gbr_pred_y"] = gbr_outputs.predictions[:, 1]
        prediction_frame["lstm_pred_x"] = lstm_outputs.predictions[:, 0]
        prediction_frame["lstm_pred_y"] = lstm_outputs.predictions[:, 1]
        prediction_frames.append(prediction_frame)

        model_summaries["GradientBoosting"][f"horizon_{horizon}_train_rows"] = float(len(y_train))
        model_summaries["LSTM"][f"horizon_{horizon}_final_train_loss"] = lstm_outputs.train_summary["final_train_loss"]

    metrics = pd.concat(metrics_rows, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["vehicle_id", "history_end_time", "horizon"]
    ).reset_index(drop=True)

    summary_gbr = regression_metrics(
        predictions[["true_x", "true_y"]].to_numpy(dtype=np.float32),
        predictions[["gbr_pred_x", "gbr_pred_y"]].to_numpy(dtype=np.float32),
    )
    summary_lstm = regression_metrics(
        predictions[["true_x", "true_y"]].to_numpy(dtype=np.float32),
        predictions[["lstm_pred_x", "lstm_pred_y"]].to_numpy(dtype=np.float32),
    )

    save_metrics(
        output_dir=OUTPUT_DIR,
        combined_metrics=metrics,
        summary_metrics={
            "GradientBoosting": {**summary_gbr, **model_summaries["GradientBoosting"]},
            "LSTM": {**summary_lstm, **model_summaries["LSTM"]},
        },
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    save_error_plot(metrics, FIGURES_DIR)
    save_model_comparison_plot(metrics, FIGURES_DIR)

    best_model_name = min(
        {"GradientBoosting": summary_gbr["rmse"], "LSTM": summary_lstm["rmse"]},
        key=lambda name: {"GradientBoosting": summary_gbr["rmse"], "LSTM": summary_lstm["rmse"]}[name],
    )
    best_predictions = (
        predictions[["gbr_pred_x", "gbr_pred_y"]].to_numpy(dtype=np.float32)
        if best_model_name == "GradientBoosting"
        else predictions[["lstm_pred_x", "lstm_pred_y"]].to_numpy(dtype=np.float32)
    )
    trajectory_preview = build_trajectory_preview(
        metadata=predictions[["vehicle_id", "history_end_time", "horizon"]],
        y_true=predictions[["true_x", "true_y"]].to_numpy(dtype=np.float32),
        y_pred=best_predictions,
        model_name=best_model_name,
    )
    save_trajectory_plot(trajectory_preview, FIGURES_DIR)

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history_window": history_window,
                "horizons": horizons,
                "random_state": random_state,
                "data_path": str(dataset_path),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

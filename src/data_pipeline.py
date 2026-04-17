from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["local_x", "local_y", "velocity_x", "velocity_y"]
TARGET_COLUMNS = ["target_x", "target_y"]


@dataclass
class SplitDataset:
    x_tabular_train: np.ndarray
    x_tabular_test: np.ndarray
    x_sequence_train: np.ndarray
    x_sequence_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    horizon_train: np.ndarray
    horizon_test: np.ndarray
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame


def load_subset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    required = {"vehicle_id", "global_time", "local_x", "local_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def add_velocity_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["vehicle_id", "global_time"]).copy()
    grouped = frame.groupby("vehicle_id", group_keys=False)

    delta_time_ms = grouped["global_time"].diff().replace(0, np.nan)
    delta_time_s = delta_time_ms.div(1000.0)
    frame["velocity_x"] = grouped["local_x"].diff().div(delta_time_s)
    frame["velocity_y"] = grouped["local_y"].diff().div(delta_time_s)

    frame["velocity_x"] = grouped["velocity_x"].transform(lambda series: series.interpolate(limit_direction="both"))
    frame["velocity_y"] = grouped["velocity_y"].transform(lambda series: series.interpolate(limit_direction="both"))
    frame["velocity_x"] = frame["velocity_x"].fillna(0.0)
    frame["velocity_y"] = frame["velocity_y"].fillna(0.0)
    return frame


def build_supervised_dataset(
    frame: pd.DataFrame,
    history_window: int,
    horizons: list[int],
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []

    for vehicle_id, group in frame.groupby("vehicle_id"):
        group = group.sort_values("global_time").reset_index(drop=True)
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        positions = group[["local_x", "local_y"]].to_numpy(dtype=np.float32)
        timestamps = group["global_time"].to_numpy(dtype=np.int64)

        max_horizon = max(horizons)
        if len(group) <= history_window + max_horizon:
            continue

        for end_idx in range(history_window - 1, len(group) - max_horizon):
            history = values[end_idx - history_window + 1 : end_idx + 1]
            base_time = timestamps[end_idx]

            for horizon in horizons:
                target_idx = end_idx + horizon
                target = positions[target_idx]
                record: dict[str, float | int] = {
                    "vehicle_id": int(vehicle_id),
                    "history_end_time": int(base_time),
                    "horizon": int(horizon),
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                }

                for step_idx, step_values in enumerate(history):
                    for feature_name, feature_value in zip(FEATURE_COLUMNS, step_values):
                        record[f"{feature_name}_t{step_idx}"] = float(feature_value)

                records.append(record)

    if not records:
        raise RuntimeError("No training examples were produced from the downloaded subset.")

    return pd.DataFrame.from_records(records)


def split_dataset(samples: pd.DataFrame, history_window: int, train_fraction: float = 0.8) -> SplitDataset:
    samples = samples.sort_values("history_end_time").reset_index(drop=True)
    cutoff = int(len(samples) * train_fraction)
    train_frame = samples.iloc[:cutoff].reset_index(drop=True)
    test_frame = samples.iloc[cutoff:].reset_index(drop=True)

    feature_names = [
        f"{feature_name}_t{step_idx}"
        for step_idx in range(history_window)
        for feature_name in FEATURE_COLUMNS
    ]

    x_tabular_train = train_frame[feature_names].to_numpy(dtype=np.float32)
    x_tabular_test = test_frame[feature_names].to_numpy(dtype=np.float32)
    x_sequence_train = x_tabular_train.reshape(len(train_frame), history_window, len(FEATURE_COLUMNS))
    x_sequence_test = x_tabular_test.reshape(len(test_frame), history_window, len(FEATURE_COLUMNS))
    y_train = train_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    return SplitDataset(
        x_tabular_train=x_tabular_train,
        x_tabular_test=x_tabular_test,
        x_sequence_train=x_sequence_train,
        x_sequence_test=x_sequence_test,
        y_train=y_train,
        y_test=y_test,
        horizon_train=train_frame["horizon"].to_numpy(dtype=np.int64),
        horizon_test=test_frame["horizon"].to_numpy(dtype=np.int64),
        metadata_train=train_frame[["vehicle_id", "history_end_time", "horizon"]].copy(),
        metadata_test=test_frame[["vehicle_id", "history_end_time", "horizon"]].copy(),
    )


def prepare_datasets(
    data_path: Path,
    history_window: int,
    horizons: list[int],
) -> SplitDataset:
    frame = load_subset(data_path)
    frame = add_velocity_features(frame)
    samples = build_supervised_dataset(frame, history_window=history_window, horizons=horizons)
    return split_dataset(samples, history_window=history_window)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_error_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=metrics, x="horizon", y="rmse", hue="model", marker="o", ax=axis)
    axis.set_title("RMSE by Prediction Horizon")
    axis.set_xlabel("Prediction Horizon (steps)")
    axis.set_ylabel("RMSE")
    figure.tight_layout()
    figure.savefig(output_dir / "error_vs_horizon.png", dpi=200)
    plt.close(figure)


def save_model_comparison_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics, x="horizon", y="mae", hue="model", ax=axis)
    axis.set_title("MAE Comparison by Horizon")
    axis.set_xlabel("Prediction Horizon (steps)")
    axis.set_ylabel("MAE")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=200)
    plt.close(figure)


def save_trajectory_plot(
    trajectory_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=trajectory_frame, x="x", y="y", hue="series", style="series", markers=True, dashes=False, ax=axis)
    axis.set_title("True vs Predicted Trajectory")
    axis.set_xlabel("Local X")
    axis.set_ylabel("Local Y")
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_true_vs_predicted.png", dpi=200)
    plt.close(figure)

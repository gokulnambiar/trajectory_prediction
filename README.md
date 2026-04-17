# Trajectory Prediction

This project predicts short-term vehicle positions from recent motion history using a small subset of the NGSIM vehicle trajectory dataset. It compares a gradient boosting baseline against a compact LSTM on the same prediction task and saves metrics, predictions, and plots to disk.

## Problem

Given recent vehicle motion, predict where the vehicle will be over the next few timesteps. The goal is short-horizon trajectory forecasting with a setup that is easy to run locally and easy to explain.

## Dataset

The project uses the public NGSIM trajectory release hosted on the U.S. Department of Transportation data portal. The downloader streams the source CSV and writes a manageable subset to `data/ngsim_subset.csv`.

- Source URL: `https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD`
- Local subset size: up to 20,000 rows
- Vehicle cap: first 120 vehicles encountered in the stream

## Approach

The pipeline:

1. Downloads or reuses the local NGSIM subset
2. Sorts records by vehicle and time
3. Derives longitudinal and lateral velocities from position deltas
4. Builds supervised examples from an 8-step history window
5. Trains separate models for 1, 3, and 5-step prediction horizons
6. Evaluates each model with RMSE and MAE
7. Writes plots and predictions to `outputs/`

Each training example uses:

- `local_x`
- `local_y`
- `velocity_x`
- `velocity_y`

## Models

### Gradient Boosting

The gradient boosting model flattens the 8-step history window into tabular features and fits a `GradientBoostingRegressor` wrapped with `MultiOutputRegressor` to predict the next `(x, y)` target.

### LSTM

The LSTM model reads the same 8-step history as a sequence of `(x, y, vx, vy)` values and predicts the future `(x, y)` position with a single-layer recurrent model and a linear output head.

## Evaluation

Examples are sorted by timestamp and split by time order, with the earlier 80% used for training and the later 20% used for testing. Metrics are reported per horizon and also aggregated across all test predictions.

Saved metrics:

- `outputs/metrics_by_horizon.csv`
- `outputs/summary_metrics.json`

Saved predictions:

- `outputs/predictions.csv`

Saved figures:

- `outputs/figures/error_vs_horizon.png`
- `outputs/figures/model_comparison.png`
- `outputs/figures/trajectory_true_vs_predicted.png`

## Project Structure

```text
trajectory_prediction/
├── data/
│   ├── download_ngsim.py
│   └── ngsim_subset.csv
├── outputs/
│   └── figures/
├── src/
│   ├── data_pipeline.py
│   ├── evaluation.py
│   ├── models.py
│   └── plotting.py
├── main.py
├── README.md
└── requirements.txt
```

## Notes

- The first run may take longer because it downloads the dataset subset and builds the local matplotlib cache.
- If `data/ngsim_subset.csv` already exists, the downloader reuses it.

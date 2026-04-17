from __future__ import annotations

from pathlib import Path

import pandas as pd


NGSIM_URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
OUTPUT_PATH = Path(__file__).resolve().parent / "ngsim_subset.csv"
MAX_ROWS = 20000
MAX_VEHICLES = 120
CHUNK_SIZE = 50000


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    return frame


def build_subset() -> pd.DataFrame:
    selected_chunks: list[pd.DataFrame] = []
    selected_vehicle_ids: set[int] = set()

    for chunk in pd.read_csv(NGSIM_URL, chunksize=CHUNK_SIZE):
        chunk = normalize_columns(chunk)
        required_columns = {"vehicle_id", "global_time", "local_x", "local_y"}
        if not required_columns.issubset(chunk.columns):
            missing = sorted(required_columns.difference(chunk.columns))
            raise ValueError(f"NGSIM columns missing from downloaded data: {missing}")

        chunk = chunk.dropna(subset=["vehicle_id", "global_time", "local_x", "local_y"]).copy()
        chunk["vehicle_id"] = chunk["vehicle_id"].astype(int)
        chunk["global_time"] = chunk["global_time"].astype(int)

        if not selected_vehicle_ids:
            selected_vehicle_ids.update(chunk["vehicle_id"].drop_duplicates().head(MAX_VEHICLES).tolist())

        filtered = chunk[chunk["vehicle_id"].isin(selected_vehicle_ids)].copy()
        if filtered.empty:
            continue

        selected_chunks.append(filtered)
        if sum(len(part) for part in selected_chunks) >= MAX_ROWS:
            break

    if not selected_chunks:
        raise RuntimeError("No rows were downloaded from the NGSIM source.")

    subset = pd.concat(selected_chunks, ignore_index=True).head(MAX_ROWS).copy()
    subset = subset.sort_values(["vehicle_id", "global_time"]).reset_index(drop=True)
    return subset


def download_ngsim_subset(output_path: Path = OUTPUT_PATH, force: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = build_subset()
    subset.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = download_ngsim_subset()
    print(path)

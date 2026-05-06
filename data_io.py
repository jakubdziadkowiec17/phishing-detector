import os
from pathlib import Path

import pandas as pd

import kagglehub

from config import DATASET, FILE_PATH


def find_first_csv(root: Path) -> Path | None:
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".csv"):
                return Path(dirpath) / name
    return None


def load_df() -> pd.DataFrame:
    cache_dir = Path("data") / "kagglehub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_first_csv(cache_dir)
    if csv_path:
        return pd.read_csv(csv_path)

    if FILE_PATH:
        from kagglehub import KaggleDatasetAdapter

        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET,
            FILE_PATH,
        )

    local_dir = Path(kagglehub.dataset_download(DATASET, output_dir=str(cache_dir)))
    csv_path = find_first_csv(local_dir)
    if not csv_path:
        raise FileNotFoundError(f"No CSV found in: {local_dir}")
    return pd.read_csv(csv_path)


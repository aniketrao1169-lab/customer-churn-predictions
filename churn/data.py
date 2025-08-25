from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd


DEFAULT_TARGET: str = "Exited"
DEFAULT_DROP_COLS: List[str] = ["RowNumber", "CustomerId", "Surname"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Place Churn_Modelling.csv there.")
    df = pd.read_csv(csv_path)
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"Loaded dataset with {after} rows (dropped {before - after} duplicates).")
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    drop_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if drop_columns is None:
        drop_columns = DEFAULT_DROP_COLS
    missing = [c for c in drop_columns + [target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df.drop(columns=drop_columns + [target_column])
    y = df[target_column]
    return X, y

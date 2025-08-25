from __future__ import annotations

import argparse
import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

from .paths import ProjectPaths
from .data import load_dataset, split_features_target


def predict_cli(args: argparse.Namespace) -> None:
    paths = ProjectPaths()

    # Load trained pipeline
    if args.model is None:
        # Try to auto-detect model file
        candidates = [p for p in os.listdir(paths.models_dir) if p.startswith("best_model_") and p.endswith(".joblib")]
        if not candidates:
            raise FileNotFoundError("No saved model found in models/. Run training first.")
        model_path = os.path.join(paths.models_dir, sorted(candidates)[-1])
    else:
        model_path = args.model

    print("Loading model:", model_path)
    pipeline = joblib.load(model_path)

    # Load threshold
    threshold_path = os.path.join(paths.models_dir, "threshold.json")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError("Threshold file not found at models/threshold.json. Train to generate it.")
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    # Load input data
    df = pd.read_csv(args.input)
    if args.has_target:
        try:
            X, y = split_features_target(df)
        except Exception:
            # Fallback if target not present despite flag
            X = df
            y = None
    else:
        X = df
        y = None

    # Predict
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out_df = pd.DataFrame({"prediction": pred, "probability": proba})

    # Optionally join selected input columns to the output
    if args.join_cols:
        requested_cols = [c.strip() for c in args.join_cols.split(",") if c.strip()]
        existing_cols = [c for c in requested_cols if c in df.columns]
        if existing_cols:
            out_df = pd.concat([df[existing_cols].reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

    # Optionally export only the top-k highest risk rows
    if args.top_k and args.top_k > 0:
        out_df = out_df.sort_values("probability", ascending=False).head(args.top_k)

    out_path = args.output or os.path.join(paths.reports_dir, "predictions.csv")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch predict churn on a CSV")
    p.add_argument("--input", type=str, required=True, help="Path to input CSV to score")
    p.add_argument("--output", type=str, default=None, help="Path to save predictions CSV")
    p.add_argument("--model", type=str, default=None, help="Path to saved model .joblib (optional)")
    p.add_argument("--has-target", action="store_true", help="Set if the input CSV includes the target column Exited")
    p.add_argument("--join-cols", type=str, default=None, help="Comma-separated list of input columns to include in output (e.g., CustomerId,Geography,Age)")
    p.add_argument("--top-k", type=int, default=0, help="If >0, write only the top-K highest churn probability rows")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    predict_cli(args)

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .paths import ProjectPaths, ensure_directories
from .data import load_dataset, split_features_target
from .pipeline import build_preprocessor, build_models


def evaluate_predictions(y_true, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
    }


def pick_threshold_for_target_recall(y_true, y_scores: np.ndarray, target_recall: float = 0.80) -> float:
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold, best_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        if rec >= target_recall and f1 > best_f1:
            best_threshold, best_f1 = thr, f1
    return best_threshold


def train_cli(args: argparse.Namespace) -> None:
    paths = ProjectPaths()
    ensure_directories(paths.models_dir, paths.reports_dir)

    # Load data
    df = load_dataset(args.data or paths.data_path)
    X, y = split_features_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build models
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    # Optional: quick model selection via CV
    import pandas as pd
    from .pipeline import cross_validate_models

    print("Running cross-validation to compare models...")
    cv_table = cross_validate_models(models, X_train, y_train)
    print(cv_table)

    # Choose top-2 models by recall
    top_two = cv_table.head(2)["model"].tolist()
    print("Top models:", top_two)

    # Set search spaces
    rf_params = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 6, 8, 12],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    xgb_params = {
        "clf__n_estimators": [300, 400, 600],
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    }

    searches: Dict[str, Any] = {}
    for name in top_two:
        model = models[name]
        if name == "RandomForest":
            param_dist = rf_params
        elif name == "XGBoost":
            param_dist = xgb_params
        else:
            # For LogisticRegression, skip heavy search
            searches[name] = model.fit(X_train, y_train)
            continue

        print(f"Tuning {name}...")
        search = RandomizedSearchCV(
            model, param_dist, n_iter=12, scoring="recall", cv=3, n_jobs=-1, random_state=42, verbose=1
        )
        search.fit(X_train, y_train)
        searches[name] = search

    # Pick best by recall from searches (fallback to CV table if needed)
    best_name = None
    best_model = None
    best_recall = -1.0

    for name, obj in searches.items():
        if hasattr(obj, "best_score_"):
            score = float(obj.best_score_)
            est = obj.best_estimator_
        else:
            # Fallback for direct-fit models
            score = float(cv_table[cv_table.model == name].iloc[0]["recall_mean"])  # type: ignore[index]
            est = obj
        if score > best_recall:
            best_recall = score
            best_name = name
            best_model = est

    assert best_model is not None and best_name is not None
    print(f"Selected best model: {best_name} (CV recall={best_recall:.3f})")

    # Fit on full train and evaluate on test
    best_model.fit(X_train, y_train)
    y_scores = best_model.predict_proba(X_test)[:, 1]

    # Choose threshold for target recall
    threshold = pick_threshold_for_target_recall(y_test, y_scores, target_recall=args.target_recall)
    metrics = evaluate_predictions(y_test, y_scores, threshold)

    print("Test metrics at chosen threshold:", metrics)
    print(f"Chosen threshold: {threshold:.2f}")

    # Save artifacts
    model_filename = os.path.join(paths.models_dir, f"best_model_{best_name.lower()}.joblib")
    joblib.dump(best_model, model_filename)

    with open(os.path.join(paths.models_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    with open(os.path.join(paths.reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model": best_name, **metrics}, f, indent=2)

    print("Saved model:", model_filename)
    print("Saved threshold to models/threshold.json and metrics to reports/metrics.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train churn prediction models")
    p.add_argument("--data", type=str, default=None, help="Path to Churn_Modelling.csv")
    p.add_argument("--target-recall", type=float, default=0.80, help="Target recall for selecting decision threshold")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train_cli(args)

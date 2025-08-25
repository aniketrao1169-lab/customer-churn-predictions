from __future__ import annotations

import argparse
import os
import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .paths import ProjectPaths, ensure_directories
from .data import split_features_target


sns.set(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.figsize"] = (9, 6)


def _load_model_and_threshold(models_dir: str):
    candidates = [p for p in os.listdir(models_dir) if p.startswith("best_model_") and p.endswith(".joblib")]
    if not candidates:
        raise FileNotFoundError("No saved model found in models/. Run training first.")
    model_path = os.path.join(models_dir, sorted(candidates)[-1])
    pipeline = joblib.load(model_path)

    thr_path = os.path.join(models_dir, "threshold.json")
    if not os.path.exists(thr_path):
        raise FileNotFoundError("Threshold file not found at models/threshold.json. Train to generate it.")
    with open(thr_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    return pipeline, threshold, model_path


def _save_churn_distribution(y: pd.Series, reports_dir: str) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    counts.plot(kind="bar", ax=ax[0], color=["#2ca02c", "#d62728"])  # 0 non-churn, 1 churn
    ax[0].set_xticklabels(["Non-Churn", "Churn"], rotation=0)
    ax[0].set_title("Churn Count (Bar)")
    ax[0].set_ylabel("Customers")

    ax[1].pie(counts, labels=["Non-Churn", "Churn"], autopct="%1.1f%%", colors=["#2ca02c", "#d62728"], startangle=90)
    ax[1].set_title("Churn Distribution (Pie)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "churn_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _save_probability_histogram(y_true: Optional[pd.Series], proba: np.ndarray, reports_dir: str) -> None:
    plt.figure()
    if y_true is not None:
        sns.kdeplot(x=proba[y_true == 0], fill=True, color="#2ca02c", label="Non-Churn")
        sns.kdeplot(x=proba[y_true == 1], fill=True, color="#d62728", label="Churn")
        plt.legend()
        plt.title("Predicted Probability by Class")
    else:
        sns.histplot(proba, bins=30, color="#1f77b4")
        plt.title("Predicted Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "probability_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _maybe_save_feature_importance(pipeline, reports_dir: str) -> None:
    try:
        preprocess = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["clf"]
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["onehot"]
        num_feature_names = preprocess.transformers_[0][2]
        cat_feature_names = cat_encoder.get_feature_names_out(preprocess.transformers_[1][2])
        feature_names = np.concatenate([num_feature_names, cat_feature_names])

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        elif hasattr(clf, "coef_"):
            coef = np.ravel(clf.coef_)
            imp_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)}).sort_values("importance", ascending=False)
        else:
            return

        top = imp_df.head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=top, y="feature", x="importance", palette="mako")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "feature_importances.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        # Keep CLI robust even if feature name extraction fails
        pass


def report_cli(args: argparse.Namespace) -> None:
    paths = ProjectPaths()
    ensure_directories(paths.models_dir, paths.reports_dir)

    df = pd.read_csv(args.input)

    # Identify features/target if present
    has_target = "Exited" in df.columns
    if has_target:
        X, y = split_features_target(df)
    else:
        X, y = df, None

    # Load model and threshold
    pipeline, threshold, model_path = _load_model_and_threshold(paths.models_dir)
    print("Using model:", model_path)
    print("Using threshold:", threshold)

    # Predict
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    # Save predictions joined with requested columns
    out_df = pd.DataFrame({"prediction": pred, "probability": proba})
    if args.join_cols:
        req = [c.strip() for c in args.join_cols.split(",") if c.strip()]
        keep = [c for c in req if c in df.columns]
        if keep:
            out_df = pd.concat([df[keep].reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

    if args.top_k and args.top_k > 0:
        out_df = out_df.sort_values("probability", ascending=False).head(args.top_k)

    out_path = args.output or os.path.join(paths.reports_dir, "predictions_with_ids.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved predictions to", out_path)

    # Charts that do not require ground truth
    _save_probability_histogram(y, proba, paths.reports_dir)
    _maybe_save_feature_importance(pipeline, paths.reports_dir)

    # Charts requiring ground truth
    if y is not None:
        _save_churn_distribution(y, paths.reports_dir)

        # Metrics and curves
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred)
        rec = recall_score(y, pred)
        f1 = f1_score(y, pred)
        auc = roc_auc_score(y, proba)
        print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})

        cm = confusion_matrix(y, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Non-Churn", "Churn"]).plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(paths.reports_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
        plt.close()

        RocCurveDisplay.from_predictions(y, proba)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(paths.reports_dir, "roc_curve.png"), dpi=200, bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y, proba)
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(paths.reports_dir, "precision_recall_curve.png"), dpi=200, bbox_inches="tight")
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate charts and a report from a dataset and saved model")
    p.add_argument("--input", type=str, required=True, help="Path to CSV (ideally with Exited column)")
    p.add_argument("--output", type=str, default=None, help="Where to write predictions-with-ids CSV")
    p.add_argument("--join-cols", type=str, default="CustomerId,Geography,Gender,Age,Tenure,Balance,NumOfProducts,IsActiveMember", help="Comma-separated input columns to include in output")
    p.add_argument("--top-k", type=int, default=0, help="If >0, save only the top-K highest-risk rows")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    report_cli(args)

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    HAS_XGB = False


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> Dict[str, ImbPipeline]:
    models: Dict[str, ImbPipeline] = {}

    log_reg = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    models["LogisticRegression"] = log_reg

    rf = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    models["RandomForest"] = rf

    if HAS_XGB:
        xgb = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
            )),
        ])
        models["XGBoost"] = xgb

    return models


def cross_validate_models(models: Dict[str, ImbPipeline], X: pd.DataFrame, y) -> pd.DataFrame:
    import pandas as pd

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        auc = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        recall = cross_val_score(model, X, y, scoring="recall", cv=cv, n_jobs=-1)
        results.append({
            "model": name,
            "roc_auc_mean": float(auc.mean()),
            "roc_auc_std": float(auc.std()),
            "recall_mean": float(recall.mean()),
            "recall_std": float(recall.std()),
        })
    return pd.DataFrame(results).sort_values(by=["recall_mean", "roc_auc_mean"], ascending=False)

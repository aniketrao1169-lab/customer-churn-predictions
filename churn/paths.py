import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPaths:
    data_path: str = "data/Churn_Modelling.csv"
    models_dir: str = "models"
    reports_dir: str = "reports"


def ensure_directories(models_dir: str, reports_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

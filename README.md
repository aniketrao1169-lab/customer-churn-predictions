## Customer Churn Prediction (CLI Version)

Command-line project to train and use a churn prediction model on the Kaggle bank churn dataset (`data/Churn_Modelling.csv`). No notebooks required.

### Dataset
- Download `Churn_Modelling.csv` from Kaggle and place it at `data/Churn_Modelling.csv`.

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train
Trains multiple models, performs quick model selection, tunes the top ones, evaluates on a holdout set, then saves:
- Model pipeline to `models/best_model_*.joblib`
- Decision threshold to `models/threshold.json`
- Metrics to `reports/metrics.json`

```bash
python -m churn.train --data data/Churn_Modelling.csv --target-recall 0.80
```

### Predict (Batch)
Scores an input CSV and writes predictions.
```bash
python -m churn.predict --input data/Churn_Modelling.csv --has-target --output reports/predictions.csv
```

- If your input has no target column, omit `--has-target`.
- You can specify a model path with `--model`, otherwise it auto-selects the latest `best_model_*.joblib`.

### Project Structure
- `churn/`: Python package with training and prediction CLIs
  - `data.py`: Data loading and splitting
  - `pipeline.py`: Preprocessing and model definitions
  - `train.py`: Training CLI
  - `predict.py`: Prediction CLI
- `data/`: Place `Churn_Modelling.csv` here (ignored by git)
- `models/`: Saved models and threshold (ignored by git)
- `reports/`: Metrics and predictions (ignored by git)

### Notes
- Default objective emphasizes Recall to avoid missing churners. Adjust `--target-recall` as needed.
- Exact library versions may vary; if you hit environment issues, pin and freeze versions after a successful run.

# mlops-heart-disease

Small end-to-end MLOps project for the UCI Heart Disease dataset. Includes data ingestion, preprocessing pipeline, model training (multiple classifiers), evaluation, plotting, MLflow experiment tracking, and a simple FastAPI prediction endpoint.

## Quick start

1. Create environment (recommended):

```bash
conda env create -f environment.yml
conda activate mlops-heart-disease
```

Or install with pip into a venv:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Download raw data:

```bash
python scripts/download_data.py
```

3. Preprocess and split:

```bash
python -c "from src.data_preprocessing import load_csv, clean_df, preprocess_and_split; df=clean_df(load_csv('data/raw/heart.csv')); preprocess_and_split(df)"
```

4. Train (with optional tuning):

Grid search (default param grids):

```bash
python -c "from src.train import train_from_csv; train_from_csv('data/processed/train.csv', out_dir='models', tuning_method='grid')"
```

Randomized search (faster):

```bash
python -c "from src.train import train_from_csv; train_from_csv('data/processed/train.csv', out_dir='models', tuning_method='random', n_iter=30)"
```

5. Generate evaluation plots (ROC, PR, confusion matrices):

```bash
python scripts/generate_eval_plots.py
```

6. Run tests:

```bash
.venv\\Scripts\\python.exe -m pytest -q
```

7. Start API (loads `models/best_model.joblib`):

```bash
uvicorn src.api:app --reload --port 8000
```

## Important files

- `scripts/download_data.py` — downloads UCI Cleveland dataset to `data/raw/heart.csv`.
- `src/preprocessing_pipeline.py` — `build_preprocessing()` returns a scikit-learn ColumnTransformer.
- `src/data_preprocessing.py` — cleaning, `preprocess_and_split()` saves processed CSVs and writes `models/preprocessor.joblib`.
- `src/train.py` — training, supports `GridSearchCV`, `RandomizedSearchCV`, and optional Optuna; saves models to `models/` and logs to MLflow when available.
- `scripts/generate_eval_plots.py` — creates evaluation plots and logs artifacts to MLflow when available.
- `notebooks/` — EDA and evaluation notebooks (executed copies and screenshots included).

## Monitoring

- The FastAPI app now exposes Prometheus metrics at `/metrics`.
- Prometheus scrape config is available at `monitoring/prometheus.yml`.
- Grafana dashboard JSON is available at `monitoring/grafana/dashboard.json`.
- Metrics tracked: request count, request latency, and error count.

## Reproducibility

- A fitted preprocessor is saved at `models/preprocessor.joblib` during preprocessing.
- A pinned environment is provided in `environment.yml` for reproducible CI and local environments.
- MLflow tracking is integrated and run artifacts are in `mlruns/` when used.

## Monitoring

- The FastAPI app now exposes a Prometheus metrics endpoint at `/metrics`.
- A Prometheus scrape config is available at `monitoring/prometheus.yml`.
- A Grafana dashboard definition is available at `monitoring/grafana/dashboard.json` for visualizing request count, latency, and error rate.

## CI

- GitHub Actions workflow is in `.github/workflows/ci.yml`: installs deps, runs lint, executes tests, and uploads `models/` as an artifact.

## License

This repository is provided as course/assignment material. Modify and reuse as needed.

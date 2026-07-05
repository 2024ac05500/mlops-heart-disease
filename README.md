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

## Understanding the repository

Follow this ordered walkthrough to understand how the code executes end to end.

1. Data ingestion and cleaning
   - `scripts/download_data.py`
     - downloads the UCI Cleveland heart disease dataset into `data/raw/heart.csv`
   - `src/data_preprocessing.py`
     - loads the raw CSV
     - cleans missing values and duplicate rows
     - fills numeric and categorical missing values
     - saves preprocessed datasets and the preprocessing artifact
   - `src/preprocessing_pipeline.py`
     - builds the scikit-learn preprocessing pipeline
     - applies median imputation and scaling for numeric features
     - applies one-hot encoding for categorical features

2. Model training and selection
   - `src/train.py`
     - loads processed train data
     - trains multiple candidate models
     - optionally performs hyperparameter search (grid or random)
     - evaluates performance using cross-validation
     - saves model artifacts to `models/`
     - logs metrics and artifacts to MLflow when available

3. Model evaluation
   - `scripts/generate_eval_plots.py`
     - loads saved models and test data
     - generates ROC, precision-recall, and confusion matrix plots
     - saves evaluation artifacts to `screenshots/`
     - optionally uploads artifacts to MLflow

4. Service deployment
   - `src/api.py`
     - loads `models/best_model.joblib` and `models/preprocessor.joblib`
     - exposes `POST /predict` for inference
     - exposes `GET /metrics` for Prometheus monitoring
     - records request count, latency, and error metrics
   - `Dockerfile`
     - packages the API into a Docker image
   - `k8s/deployment.yaml` and `k8s/service.yaml`
     - define deployment and service manifests for Kubernetes

5. Monitoring and observability
   - `monitoring/prometheus.yml`
     - configures Prometheus to scrape `GET /metrics`
   - `monitoring/grafana/dashboard.json`
     - dashboard definition for request rate, latency, and error monitoring

6. Continuous integration
   - `.github/workflows/ci.yml`
     - installs dependencies
     - runs flake8 linting and pytest tests
     - performs a quick smoke training run
     - uploads the `models/` directory as a CI artifact

## Important files

- `scripts/download_data.py` — downloads UCI Cleveland dataset to `data/raw/heart.csv`.
- `src/preprocessing_pipeline.py` — `build_preprocessing()` returns a scikit-learn ColumnTransformer.
- `src/data_preprocessing.py` — cleaning, `preprocess_and_split()` saves processed CSVs and writes `models/preprocessor.joblib`.
- `src/train.py` — training, supports `GridSearchCV`, `RandomizedSearchCV`, and optional Optuna; saves models to `models/` and logs to MLflow when available.
- `scripts/generate_eval_plots.py` — creates evaluation plots and logs artifacts to MLflow when available.
- `src/api.py` — FastAPI prediction and metrics service.
- `Dockerfile` — container packaging for the API.
- `k8s/deployment.yaml` / `k8s/service.yaml` — Kubernetes deployment and service manifests.
- `monitoring/prometheus.yml` — Prometheus scrape config.
- `monitoring/grafana/dashboard.json` — Grafana dashboard definition.
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

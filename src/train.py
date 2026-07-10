import os
import json
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from joblib import dump
from joblib import dump as _dump
# keep joblib.load available via direct import when needed
from pathlib import Path


# optional MLflow; fall back to no-op logger if not installed
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_ENABLED = True
except Exception:
    mlflow = None
    MLFLOW_ENABLED = False


def evaluate_model(model, X, y) -> dict:
    y_pred = cross_val_predict(model, X, y, cv=3)
    scores = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    try:
        y_prob = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]
        scores["roc_auc"] = float(roc_auc_score(y, y_prob))
    except Exception:
        try:
            y_prob = cross_val_predict(model, X, y, cv=3, method="decision_function")
            if y_prob.ndim > 1:
                y_prob = y_prob.ravel()
            scores["roc_auc"] = float(roc_auc_score(y, y_prob))
        except Exception:
            scores["roc_auc"] = float("nan")
    return scores


def _plot_and_log_artifacts(model, X, y, run, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        y_pred = model.predict(X)
    except Exception:
        return

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    cm_path = out_dir / f'confusion_{model_name}.png'
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    if MLFLOW_ENABLED and run is not None:
        mlflow.log_artifact(str(cm_path), artifact_path='plots')

    # ROC curve
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        try:
            y_prob = model.decision_function(X)
            if y_prob.ndim > 1:
                y_prob = y_prob.ravel()
        except Exception:
            y_prob = None

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y, y_prob):.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc='lower right')
        roc_path = out_dir / f'roc_{model_name}.png'
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        if MLFLOW_ENABLED and run is not None:
            mlflow.log_artifact(str(roc_path), artifact_path='plots')


def _log_run_params(name, quick, tuning_method, n_iter, cv, X, y):
    """Log core run parameters for reproducibility and comparison."""
    params = {
        "model_name": name,
        "quick_mode": bool(quick),
        "tuning_method": tuning_method or "none",
        "n_iter": int(n_iter),
        "cv": int(cv),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else 1,
    }
    mlflow.log_params(params)


def _log_estimator_params(model):
    """Log simple estimator params while skipping non-serializable values."""
    try:
        base_params = model.get_params(deep=False)
    except Exception:
        return

    serializable = {}
    for k, v in base_params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            serializable[f"est_{k}"] = v

    if serializable:
        mlflow.log_params(serializable)


def _get_models():
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
    }

    # try xgboost first, otherwise use sklearn's GradientBoostingClassifier
    try:
        import xgboost as xgb

        models["xgb"] = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=0,
        )
    except Exception:
        models["xgb"] = GradientBoostingClassifier(random_state=0)

    # add SVM (probabilities) but it's slower
    try:
        models["svc"] = SVC(probability=True, random_state=0)
    except Exception:
        pass

    return models


def _default_param_grids():
    return {
        "logreg": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "rf": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
        "xgb": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "svc": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    }


def tune_model(model, X, y, param_grid=None, method="grid", n_iter=20, cv=3, scoring="accuracy", random_state=0):
    """Tune `model` using GridSearchCV or RandomizedSearchCV.

    method: 'grid' | 'random' | 'optuna' (optuna optional)
    Returns: best_estimator_, best_params_
    """
    if param_grid is None:
        return model, {}

    if method == "grid":
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_

    if method == "random":
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_

    # optional: try optuna if available
    if method == "optuna":
        try:
            import optuna

            def _objective(trial):
                params = {}
                # sample from provided param_grid choosing either categorical or continuous
                for k, v in param_grid.items():
                    if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                        # treat as categorical for now
                        params[k] = trial.suggest_categorical(k, v)
                    else:
                        params[k] = trial.suggest_categorical(k, v)

                model.set_params(**params)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                return float(np.mean(scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials=n_iter)
            best_params = study.best_params
            model.set_params(**best_params)
            model.fit(X, y)
            return model, best_params
        except Exception:
            # optuna not available or failed; fall back
            return model, {}

    return model, {}


def train_and_log(
    X,
    y,
    out_dir: str = "models",
    quick: bool = False,
    tuning_method: str = None,
    param_grids: dict = None,
    n_iter: int = 20,
    cv: int = 3,
    preprocessor=None,
) -> Tuple[str, object, str]:
    os.makedirs(out_dir, exist_ok=True)
    if quick:
        X = X[:50]
        y = y[:50]

    models = _get_models()

    if MLFLOW_ENABLED:
        mlflow.set_experiment("heart_disease_experiment")

    best_score = -1.0
    best_model = None
    best_pipeline = None

    for name, model in models.items():
        if MLFLOW_ENABLED:
            run = mlflow.start_run(run_name=name)
            _log_run_params(name, quick, tuning_method, n_iter, cv, X, y)
            _log_estimator_params(model)
        else:
            run = None

        try:
            # perform hyperparameter tuning if requested and grid provided
            grid = None
            if param_grids is None:
                grid = _default_param_grids().get(name)
            else:
                grid = param_grids.get(name)

            tuned_params = None
            if tuning_method and grid:
                try:
                    best_est, best_params = tune_model(
                        model,
                        X,
                        y,
                        param_grid=grid,
                        method=tuning_method,
                        n_iter=n_iter,
                        cv=cv,
                    )
                    model = best_est
                    tuned_params = best_params
                except Exception as e:
                    print(f"Tuning failed for {name}: {e}")

            model.fit(X, y)
            scores = evaluate_model(model, X, y)

            if MLFLOW_ENABLED:
                for k, v in scores.items():
                    mlflow.log_metric(k, v)
                if tuned_params:
                    for pk, pv in tuned_params.items():
                        try:
                            mlflow.log_param(pk, pv)
                        except Exception:
                            pass
                mlflow.sklearn.log_model(model, f"model_{name}")
            else:
                print(f"Model {name} scores: {scores}")

            # save model artifact locally
            model_path = os.path.join(out_dir, f"model_{name}.joblib")
            dump(model, model_path)
            if MLFLOW_ENABLED and run is not None:
                mlflow.log_artifact(model_path, artifact_path="models")

            # save a wrapped pipeline for reproducible inference if preprocessing is available
            pipeline_path = None
            if preprocessor is not None:
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", model),
                ])
                pipeline_path = os.path.join(out_dir, f"pipeline_{name}.joblib")
                dump(pipeline, pipeline_path)
                if MLFLOW_ENABLED and run is not None:
                    mlflow.log_artifact(pipeline_path, artifact_path='pipelines')
                if scores.get("accuracy", 0) > best_score:
                    best_pipeline = pipeline

            if MLFLOW_ENABLED and run is not None:
                summary_path = os.path.join(out_dir, f"run_summary_{name}.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": name,
                            "scores": scores,
                            "tuned_params": tuned_params or {},
                            "model_artifact": model_path,
                            "pipeline_artifact": pipeline_path,
                        },
                        f,
                        indent=2,
                    )
                mlflow.log_artifact(summary_path, artifact_path="summaries")

            # plot and log artifacts to MLflow
            if MLFLOW_ENABLED and run is not None:
                _plot_and_log_artifacts(
                    model,
                    X,
                    y,
                    run,
                    name,
                    out_dir=os.path.join(out_dir, "plots"),
                )

            if scores.get("accuracy", 0) > best_score:
                best_score = scores["accuracy"]
                # record best model object
                best_model = model
        finally:
            if MLFLOW_ENABLED and run is not None:
                mlflow.end_run()

    out_path = os.path.join(out_dir, "best_model.joblib")
    dump(best_model, out_path)
    best_pipeline_path = None
    if best_pipeline is not None:
        best_pipeline_path = os.path.join(out_dir, "best_pipeline.joblib")
        dump(best_pipeline, best_pipeline_path)
    return out_path, best_model, best_pipeline_path


def train_from_csv(
    train_csv_path: str = "data/processed/train.csv",
    out_dir: str = "models",
    quick: bool = False,
    tuning_method: str = None,
    param_grids: dict = None,
    n_iter: int = 20,
    cv: int = 3,
):
    import pandas as pd

    df = pd.read_csv(train_csv_path)
    if "target" not in df.columns:
        raise ValueError("train CSV must contain 'target' column")
    y = df["target"].values
    X = df.drop(columns=["target"]).values

    preprocessor = None
    preproc_path = os.path.join("models", "preprocessor.joblib")
    if os.path.exists(preproc_path):
        try:
            from joblib import load

            preprocessor = load(preproc_path)
        except Exception:
            preprocessor = None

    return train_and_log(
        X,
        y,
        out_dir=out_dir,
        quick=quick,
        tuning_method=tuning_method,
        param_grids=param_grids,
        n_iter=n_iter,
        cv=cv,
        preprocessor=preprocessor,
    )


def train_model(X, y, out_path: str = "models/model.joblib"):
    """Compatibility wrapper for tests: train quickly and save model to `out_path`.

    Returns the trained model.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # use quick training to keep CI fast
    _, model, _ = train_and_log(
        X,
        y,
        out_dir=str(out_p.parent),
        quick=True,
    )

    try:
        # save specifically to requested path
        _dump(model, str(out_p))
    except Exception:
        pass

    return model


if __name__ == "__main__":
    # run training on processed train.csv
    train_from_csv("data/processed/train.csv", out_dir="models", quick=False)

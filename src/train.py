import os
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump
from joblib import dump as _dump
from joblib import load as _load
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
    scores = {}
    scores["accuracy"] = float(cross_val_score(model, X, y, cv=3).mean())
    return scores


def _get_models():
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
    }

    # try xgboost first, otherwise use sklearn's GradientBoostingClassifier
    try:
        import xgboost as xgb

        models["xgb"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=0)
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
        "logreg": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        "rf": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
        "xgb": {"n_estimators": [50, 100, 200], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.2]},
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
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, n_jobs=-1)
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


def train_and_log(X, y, out_dir: str = "models", quick: bool = False, tuning_method: str = None, param_grids: dict = None, n_iter: int = 20, cv: int = 3) -> Tuple[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    if quick:
        X = X[:50]
        y = y[:50]

    models = _get_models()

    if MLFLOW_ENABLED:
        mlflow.set_experiment("heart_disease_experiment")

    best_score = -1.0
    best_name = None
    best_model = None

    for name, model in models.items():
        if MLFLOW_ENABLED:
            run = mlflow.start_run(run_name=name)
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
                    best_est, best_params = tune_model(model, X, y, param_grid=grid, method=tuning_method, n_iter=n_iter, cv=cv)
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

            if scores.get("accuracy", 0) > best_score:
                best_score = scores["accuracy"]
                best_name = name
                best_model = model
        finally:
            if MLFLOW_ENABLED and run is not None:
                mlflow.end_run()

    out_path = os.path.join(out_dir, "best_model.joblib")
    dump(best_model, out_path)
    return out_path, best_model


def train_from_csv(train_csv_path: str = "data/processed/train.csv", out_dir: str = "models", quick: bool = False, tuning_method: str = None, param_grids: dict = None, n_iter: int = 20, cv: int = 3):
    import pandas as pd

    df = pd.read_csv(train_csv_path)
    if "target" not in df.columns:
        raise ValueError("train CSV must contain 'target' column")
    y = df["target"].values
    X = df.drop(columns=["target"]).values
    return train_and_log(X, y, out_dir=out_dir, quick=quick, tuning_method=tuning_method, param_grids=param_grids, n_iter=n_iter, cv=cv)


def train_model(X, y, out_path: str = "models/model.joblib"):
    """Compatibility wrapper for tests: train quickly and save model to `out_path`.

    Returns the trained model.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # use quick training to keep CI fast
    _, model = train_and_log(X, y, out_dir=str(out_p.parent), quick=True)

    try:
        # save specifically to requested path
        _dump(model, str(out_p))
    except Exception:
        pass

    return model


if __name__ == "__main__":
    # run training on processed train.csv
    train_from_csv("data/processed/train.csv", out_dir="models", quick=False)

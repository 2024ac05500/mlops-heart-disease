import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

import src.train as train_mod


class _DummyMlflow:
    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.runs = []

        class _Sklearn:
            @staticmethod
            def log_model(*args, **kwargs):
                return None

        self.sklearn = _Sklearn()

    def set_experiment(self, *_args, **_kwargs):
        return None

    def start_run(self, run_name=None):
        self.runs.append(run_name)
        return object()

    def end_run(self):
        return None

    def log_params(self, params):
        self.params.update(params)

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


def test_default_param_grids_and_log_helpers(monkeypatch):
    grids = train_mod._default_param_grids()
    assert set(["logreg", "rf", "xgb", "svc"]).issubset(grids.keys())

    dummy_mlflow = _DummyMlflow()
    monkeypatch.setattr(train_mod, "mlflow", dummy_mlflow)

    X = np.ones((5, 3))
    y = np.array([0, 1, 0, 1, 1])
    train_mod._log_run_params("logreg", quick=True, tuning_method=None, n_iter=3, cv=2, X=X, y=y)
    assert dummy_mlflow.params["model_name"] == "logreg"
    assert dummy_mlflow.params["quick_mode"] is True

    class _WeirdEstimator:
        def get_params(self, deep=False):
            return {"a": 1, "b": "x", "c": object()}

    train_mod._log_estimator_params(_WeirdEstimator())
    assert "est_a" in dummy_mlflow.params
    assert "est_b" in dummy_mlflow.params
    assert "est_c" not in dummy_mlflow.params


def test_tune_model_none_and_unknown_method():
    X, y = make_classification(n_samples=40, n_features=5, random_state=0)
    model = LogisticRegression(max_iter=200)

    same_model, params = train_mod.tune_model(model, X, y, param_grid=None)
    assert same_model is model
    assert params == {}

    same_model2, params2 = train_mod.tune_model(
        model,
        X,
        y,
        param_grid={"C": [0.1, 1.0]},
        method="unknown",
    )
    assert same_model2 is model
    assert params2 == {}


def test_tune_model_grid_and_random_branches():
    X, y = make_classification(n_samples=60, n_features=6, random_state=11)

    model = LogisticRegression(max_iter=300)
    grid = {"C": [0.1, 1.0], "solver": ["lbfgs"], "penalty": ["l2"]}

    best_g, params_g = train_mod.tune_model(
        model,
        X,
        y,
        param_grid=grid,
        method="grid",
        cv=2,
    )
    assert best_g is not None
    assert "C" in params_g

    best_r, params_r = train_mod.tune_model(
        LogisticRegression(max_iter=300),
        X,
        y,
        param_grid=grid,
        method="random",
        n_iter=1,
        cv=2,
    )
    assert best_r is not None
    assert "C" in params_r


def test_plot_and_log_artifacts_paths(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=40, n_features=6, random_state=1)
    model = LogisticRegression(max_iter=300).fit(X, y)

    dummy_mlflow = _DummyMlflow()
    monkeypatch.setattr(train_mod, "mlflow", dummy_mlflow)
    monkeypatch.setattr(train_mod, "MLFLOW_ENABLED", True)

    train_mod._plot_and_log_artifacts(model, X, y, run=object(), model_name="logreg", out_dir=tmp_path)
    assert (tmp_path / "confusion_logreg.png").exists()
    assert (tmp_path / "roc_logreg.png").exists()
    assert any("confusion_logreg.png" in p for p, _ in dummy_mlflow.artifacts)


def test_train_and_log_mlflow_disabled_and_preprocessor_pipeline(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=70, n_features=7, random_state=2)

    monkeypatch.setattr(train_mod, "MLFLOW_ENABLED", False)
    monkeypatch.setattr(
        train_mod,
        "_get_models",
        lambda: {"logreg": LogisticRegression(max_iter=200)},
    )
    monkeypatch.setattr(
        train_mod,
        "evaluate_model",
        lambda model, Xv, yv: {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "roc_auc": 0.75},
    )

    # Use a picklable preprocessor to exercise pipeline artifact branch.
    preproc = FunctionTransformer(np.asarray, validate=False)

    out_path, best_model, best_pipeline_path = train_mod.train_and_log(
        X,
        y,
        out_dir=str(tmp_path),
        quick=True,
        preprocessor=preproc,
    )

    assert os.path.exists(out_path)
    assert best_model is not None
    assert best_pipeline_path is not None
    assert os.path.exists(best_pipeline_path)


def test_train_and_log_mlflow_enabled_and_tuning_fallback(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=60, n_features=5, random_state=3)

    dummy_mlflow = _DummyMlflow()
    monkeypatch.setattr(train_mod, "mlflow", dummy_mlflow)
    monkeypatch.setattr(train_mod, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(
        train_mod,
        "_get_models",
        lambda: {"logreg": LogisticRegression(max_iter=200)},
    )
    monkeypatch.setattr(
        train_mod,
        "evaluate_model",
        lambda model, Xv, yv: {"accuracy": 0.81, "precision": 0.8, "recall": 0.8, "f1": 0.8, "roc_auc": 0.82},
    )
    monkeypatch.setattr(train_mod, "_plot_and_log_artifacts", lambda *args, **kwargs: None)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("tune failed")

    monkeypatch.setattr(train_mod, "tune_model", _boom)

    out_path, best_model, _ = train_mod.train_and_log(
        X,
        y,
        out_dir=str(tmp_path),
        quick=False,
        tuning_method="grid",
        param_grids={"logreg": {"C": [0.1, 1.0]}},
    )

    assert os.path.exists(out_path)
    assert best_model is not None
    assert dummy_mlflow.runs == ["logreg"]
    assert "accuracy" in dummy_mlflow.metrics


def test_train_from_csv_missing_target_raises(tmp_path):
    csv_path = tmp_path / "bad_train.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    try:
        train_mod.train_from_csv(str(csv_path), out_dir=str(tmp_path), quick=True)
        raised = False
    except ValueError:
        raised = True
    assert raised

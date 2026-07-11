import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation import evaluate_model_cv, evaluate_models
from src.preprocessing_pipeline import build_preprocessing
from src.utils import save_model, load_model
from src.data_preprocessing import clean_df, preprocess_and_split


def test_evaluate_model_cv_with_predict_proba_and_decision_function():
    X, y = make_classification(n_samples=80, n_features=8, random_state=42)

    metrics_lr = evaluate_model_cv(LogisticRegression(max_iter=1000), X, y, cv=3)
    assert set(["accuracy", "precision", "recall", "f1", "roc_auc"]).issubset(metrics_lr.keys())
    assert 0.0 <= metrics_lr["accuracy"] <= 1.0

    metrics_svc = evaluate_model_cv(LinearSVC(random_state=42), X, y, cv=3)
    assert set(["accuracy", "precision", "recall", "f1", "roc_auc"]).issubset(metrics_svc.keys())
    assert 0.0 <= metrics_svc["accuracy"] <= 1.0


def test_evaluate_models_collects_errors_for_invalid_model():
    X, y = make_classification(n_samples=60, n_features=6, random_state=0)

    class BadModel:
        pass

    results = evaluate_models(
        {
            "ok": LogisticRegression(max_iter=300),
            "bad": BadModel(),
        },
        X,
        y,
        cv=3,
    )

    assert "ok" in results
    assert "accuracy" in results["ok"]
    assert "bad" in results
    assert "error" in results["bad"]


def test_build_preprocessing_and_utils_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "age": [50, 60, 55, 52],
            "cp": [1, 2, 3, 1],
            "thal": ["normal", "fixed", "reversible", "normal"],
        }
    )

    preproc = build_preprocessing(df)
    Xt = preproc.fit_transform(df)
    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] >= 3

    out_path = tmp_path / "preproc.joblib"
    save_model(preproc, str(out_path))
    loaded = load_model(str(out_path))
    Xt2 = loaded.transform(df)
    assert Xt2.shape == Xt.shape


def test_clean_df_and_preprocess_and_split_outputs(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "age": [52, 52, 58, 60, 61, 45],
            "sex": [1, 1, 0, 1, 0, 1],
            "cp": [1, 1, 2, 3, 2, 0],
            "fbs": [0, 0, 1, 0, 1, 0],
            "restecg": [0, 0, 1, 2, 1, 0],
            "exang": [0, 0, 1, 0, 1, 0],
            "slope": [1, 1, 2, 3, 2, 1],
            "ca": [0, 0, 1, 2, 1, 0],
            "thal": ["normal", "normal", "fixed", "reversible", "fixed", "normal"],
            "chol": ["240", "240", "?", "250", "241", "260"],
            "target": [0, 0, 2, 1, 2, 0],
        }
    )

    cleaned = clean_df(df)
    assert len(cleaned) < len(df)
    assert pd.isna(cleaned["chol"]).any()

    # Avoid side effects in repository-level models/ during test runs.
    monkeypatch.setattr("src.data_preprocessing.dump", lambda *args, **kwargs: None)

    out_dir = tmp_path / "processed"
    train_path, test_path = preprocess_and_split(
        cleaned,
        target_col="target",
        test_size=0.5,
        random_state=0,
        out_dir=str(out_dir),
    )

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    assert os.path.exists(out_dir / "heart_processed.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert "target" in train_df.columns
    assert "target" in test_df.columns
    assert set(train_df["target"].unique()).issubset({0, 1})
    assert set(test_df["target"].unique()).issubset({0, 1})

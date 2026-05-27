from typing import Dict
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _get_probs_from_cv(model, X, y, cv=5):
    """Try to obtain out-of-fold probability estimates for ROC-AUC.

    Falls back to decision_function when predict_proba is unavailable.
    """
    try:
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        return probs
    except Exception:
        try:
            probs = cross_val_predict(model, X, y, cv=cv, method="decision_function")
            # decision_function may return shape (n,) or (n,1)
            probs = np.asarray(probs)
            if probs.ndim > 1:
                probs = probs.ravel()
            return probs
        except Exception:
            return None


def evaluate_model_cv(model, X, y, cv=5) -> Dict[str, float]:
    """Compute cross-validated metrics for a single model."""
    y_pred = cross_val_predict(model, X, y, cv=cv)

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y, y_pred, zero_division=0))

    probs = _get_probs_from_cv(model, X, y, cv=cv)
    if probs is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, probs))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_models(models_dict, X, y, cv=5):
    results = {}
    for name, model in models_dict.items():
        try:
            metrics = evaluate_model_cv(model, X, y, cv=cv)
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

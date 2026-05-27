from joblib import load
import numpy as np


def predict(model_path: str, X):
    model = load(model_path)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    probs = model.predict_proba(X)[:, 1]
    return probs

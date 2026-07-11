# Purpose: Provide shared utility helpers.
from joblib import dump, load


def save_model(model, path: str):
    dump(model, path)


def load_model(path: str):
    return load(path)

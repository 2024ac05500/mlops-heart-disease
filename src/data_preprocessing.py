import os
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from .preprocessing_pipeline import build_preprocessing


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: replace unknowns, coerce numeric types where appropriate."""
    df = df.copy()
    df = df.replace("?", pd.NA)

    # attempt to convert columns to numeric when possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    # drop exact duplicate rows
    df = df.drop_duplicates()
    return df


def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str = "target",
    cat_cols: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: str = "data/processed",
) -> Tuple[str, str]:
    """Run preprocessing pipeline, split into train/test, and save CSVs.

    Returns tuple of (train_path, test_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()

    # ensure target is numeric and binary (0/1)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = (df[target_col] > 0).astype(int)

    if cat_cols is None:
        cat_cols = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]

    # cast categories when present
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    preproc = build_preprocessing(X_raw)
    X_transformed = preproc.fit_transform(X_raw)

    # build feature names: numeric + one-hot
    num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
    cat_feature_names = []
    try:
        cat_pipeline = preproc.named_transformers_.get("cat")
        if cat_pipeline is not None:
            ohe = cat_pipeline.named_steps.get("onehot")
            if ohe is not None:
                present_cat_cols = [c for c in cat_cols if c in X_raw.columns]
                cat_feature_names = list(ohe.get_feature_names_out(present_cat_cols))
    except Exception:
        cat_feature_names = []

    feature_names = list(num_cols) + list(cat_feature_names)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_df.index = y.index

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    processed_path = os.path.join(out_dir, "heart_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    X_df.to_csv(processed_path, index=False)

    dump(preproc, "models/preprocessor.joblib")

    return train_path, test_path


if __name__ == "__main__":
    # quick runner: load raw file, clean, preprocess, and save splits
    raw_path = "data/raw/heart.csv"
    df = load_csv(raw_path)
    df = clean_df(df)
    tp, vp = preprocess_and_split(df)
    print(f"Saved processed train/test to: {tp}, {vp}")

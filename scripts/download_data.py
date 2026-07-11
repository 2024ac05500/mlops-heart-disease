# Purpose: Download raw heart disease dataset to local storage.
import os
import io
import pandas as pd
import urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
HEADERS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download(output_path: str = "data/raw/heart.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        text = resp.read().decode("utf-8")

    # processed.cleveland.data is comma-separated without header and uses ? for missing
    df = pd.read_csv(io.StringIO(text), header=None, names=HEADERS)
    df = df.replace("?", pd.NA)
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    download()

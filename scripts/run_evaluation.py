import pandas as pd
import os
import sys
from pathlib import Path

# ensure repo root is on path
repo_root = next((p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

from src.train import _get_models
from src.evaluation import evaluate_models


def run(train_csv: str = "data/processed/train.csv", out_path: str = "models/evaluation_results.csv"):
    df = pd.read_csv(train_csv)
    if "target" not in df.columns:
        raise ValueError("train CSV must contain 'target' column")
    y = df["target"].values
    X = df.drop(columns=["target"]).values

    models = _get_models()
    results = evaluate_models(models, X, y, cv=5)

    rows = []
    for name, metrics in results.items():
        if "error" in metrics:
            rows.append({"model": name, "error": metrics["error"]})
            continue
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved evaluation results to {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    run()

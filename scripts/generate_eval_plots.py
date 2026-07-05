from pathlib import Path
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


# optional MLflow artifact logging
try:
    import mlflow
    MLFLOW_ENABLED = True
except Exception:
    mlflow = None
    MLFLOW_ENABLED = False


def get_probs(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            p = model.decision_function(X)
            p = np.asarray(p)
            if p.ndim > 1:
                p = p.ravel()
            return p
        except Exception:
            return None


def main():
    root = Path.cwd()
    test_path = root / 'data' / 'processed' / 'test.csv'
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    df = pd.read_csv(test_path)
    y = df['target'].values
    X = df.drop(columns=['target']).values

    models = {}
    for mf in sorted(glob.glob(str(root / 'models' / 'model_*.joblib'))):
        name = Path(mf).stem
        try:
            models[name] = joblib.load(mf)
        except Exception:
            pass
    # include best_model if present
    best = root / 'models' / 'best_model.joblib'
    if best.exists():
        try:
            models['best_model'] = joblib.load(best)
        except Exception:
            pass

    out_dir = root / 'screenshots'
    out_dir.mkdir(exist_ok=True)
    sns.set(style='whitegrid')

    # ROC
    plt.figure(figsize=(8, 6))
    any_plot = False
    for name, model in models.items():
        probs = get_probs(model, X)
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_val:.3f})')
        any_plot = True
    if any_plot:
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(out_dir / 'roc_curves.png')
        plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    any_plot = False
    for name, model in models.items():
        probs = get_probs(model, X)
        if probs is None:
            continue
        prec, rec, _ = precision_recall_curve(y, probs)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f'{name} (AUC={pr_auc:.3f})')
        any_plot = True
    if any_plot:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(out_dir / 'pr_curves.png')
        plt.close()

    # confusion matrices and reports
    for name, model in models.items():
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {name}')
        plt.tight_layout()
        plt.savefig(out_dir / f'confusion_{name}.png')
        plt.close()

    print('Saved plots to', out_dir)

    # log artifacts to MLflow if available
    if MLFLOW_ENABLED:
        try:
            # ensure we have an active run
            if mlflow.active_run() is None:
                with mlflow.start_run(run_name="generate_eval_plots"):
                    mlflow.log_artifacts(str(out_dir), artifact_path="plots")
            else:
                mlflow.log_artifacts(str(out_dir), artifact_path="plots")

            # also log evaluation CSV if present
            eval_csv = Path('models') / 'evaluation_results.csv'
            if eval_csv.exists():
                if mlflow.active_run() is None:
                    with mlflow.start_run(run_name="generate_eval_plots"):
                        mlflow.log_artifact(str(eval_csv), artifact_path="metrics")
                else:
                    mlflow.log_artifact(str(eval_csv), artifact_path="metrics")
        except Exception as e:
            print('MLflow artifact logging failed:', e)


if __name__ == '__main__':
    main()

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return ROC-AUC, PR-AUC, balanced accuracy, F1, sensitivity, and specificity."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Return {metric: (mean, std)} across folds."""
    keys = fold_metrics[0].keys()
    return {
        k: (
            float(np.mean([m[k] for m in fold_metrics])),
            float(np.std([m[k] for m in fold_metrics])),
        )
        for k in keys
    }


def format_results_table(all_results: dict) -> str:
    """Format a results dict as a human-readable aligned table string."""
    header = f"{'Mode':<10} {'ROC-AUC':>14} {'PR-AUC':>14} {'Bal-Acc':>14} {'F1':>14} {'Sens':>14} {'Spec':>14}"
    lines = [header, '-' * len(header)]
    for mode, metrics in all_results.items():
        def fmt(k):
            m, s = metrics[k]
            return f"{m:.3f}+/-{s:.3f}"
        lines.append(
            f"{mode:<10} {fmt('roc_auc'):>14} {fmt('pr_auc'):>14} "
            f"{fmt('balanced_accuracy'):>14} {fmt('f1'):>14} "
            f"{fmt('sensitivity'):>14} {fmt('specificity'):>14}"
        )
    return '\n'.join(lines)

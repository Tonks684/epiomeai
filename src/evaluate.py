"""Load saved metrics and OOF predictions, produce all evaluation plots."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

METRIC_LABELS = {
    'roc_auc':           'ROC-AUC',
    'pr_auc':            'PR-AUC',
    'balanced_accuracy': 'Balanced Accuracy',
    'f1':                'F1',
    'sensitivity':       'Sensitivity',
    'specificity':       'Specificity',
}

MODEL_COLOURS = {
    'logreg': '#4e79a7',
    'gbm':    '#f28e2b',
    'mlp':    '#59a14f',
}

MODE_STYLES = {
    't0':     ('solid',   'o'),
    't1':     ('dashed',  's'),
    'concat': ('dashdot', '^'),
    'delta':  ('dotted',  'D'),
}

MODE_COLOURS = {
    't0':     '#4e79a7',
    't1':     '#f28e2b',
    'concat': '#59a14f',
    'delta':  '#e15759',
}

TASK_TITLES = {
    'task1': 'Task 1: CN -> MCI',
    'task2': 'Task 2: MCI -> Dementia',
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics(metrics_dir: Path, task: str) -> dict:
    """Return {model: {mode: {metric: (mean, std)}}}"""
    results = {}
    for path in sorted(metrics_dir.glob(f'{task}_*.json')):
        model = path.stem.replace(f'{task}_', '')
        with open(path) as f:
            results[model] = json.load(f)
    return results


def load_oof(pred_dir: Path, task: str, model: str, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load OOF predictions for a model/mode combination; return None if the file does not exist."""
    path = pred_dir / f'{task}_{model}_{mode}_oof.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df['y_true'].values, df['y_prob'].values


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def plot_metric_bars(results: dict, task: str, metric: str, out_path: Path) -> None:
    """Save a grouped bar chart of one metric across all models and temporal modes."""
    models = list(results.keys())
    modes  = list(next(iter(results.values())).keys())

    x       = np.arange(len(models))
    n_modes = len(modes)
    width   = 0.18
    offsets = np.linspace(-(n_modes - 1) / 2, (n_modes - 1) / 2, n_modes) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        means  = [results[m][mode][metric][0] for m in models]
        stds   = [results[m][mode][metric][1] for m in models]
        colour = MODE_COLOURS.get(mode, f'C{i}')
        ax.bar(x + offsets[i], means, width, yerr=stds, label=mode,
               color=colour, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f'{TASK_TITLES[task]} — {METRIC_LABELS[metric]} (mean +/- std, 5x3 CV)')
    ax.legend(title='Temporal mode')
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, pred_dir: Path, task: str, out_path: Path) -> None:
    """Save ROC curves computed from pooled OOF predictions for all model/mode combinations."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)

    legend_handles = []
    for model, modes in results.items():
        colour = MODEL_COLOURS.get(model, '#888888')
        for mode, _ in modes.items():
            oof = load_oof(pred_dir, task, model, mode)
            if oof is None:
                continue
            y_true, y_prob = oof
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ls, _ = MODE_STYLES.get(mode, ('solid', 'o'))
            ax.plot(fpr, tpr, color=colour, linestyle=ls, lw=1.5, alpha=0.9)
            legend_handles.append(
                mlines.Line2D([], [], color=colour, linestyle=ls, lw=1.5,
                              label=f'{model}/{mode}  AUC={roc_auc:.3f}')
            )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{TASK_TITLES[task]} — ROC curves (OOF)')
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PR curves
# ---------------------------------------------------------------------------

def plot_pr_curves(results: dict, pred_dir: Path, task: str, out_path: Path) -> None:
    """Save precision-recall curves computed from pooled OOF predictions for all model/mode combinations."""
    fig, ax = plt.subplots(figsize=(7, 6))

    legend_handles = []
    for model, modes in results.items():
        colour = MODEL_COLOURS.get(model, '#888888')
        for mode, _ in modes.items():
            oof = load_oof(pred_dir, task, model, mode)
            if oof is None:
                continue
            y_true, y_prob = oof
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ls, _ = MODE_STYLES.get(mode, ('solid', 'o'))
            ax.plot(recall, precision, color=colour, linestyle=ls, lw=1.5, alpha=0.9)
            legend_handles.append(
                mlines.Line2D([], [], color=colour, linestyle=ls, lw=1.5,
                              label=f'{model}/{mode}  AP={ap:.3f}')
            )

    baseline = (results[list(results.keys())[0]]['t0']['pr_auc'][0])
    ax.axhline(
        sum(1 for v in load_oof(pred_dir, task, list(results.keys())[0], 't0')[0] if v == 1)
        / len(load_oof(pred_dir, task, list(results.keys())[0], 't0')[0]),
        color='k', linestyle='--', lw=0.8, alpha=0.5, label='No-skill baseline'
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{TASK_TITLES[task]} — Precision-Recall curves (OOF)')
    ax.legend(handles=legend_handles + [
        mlines.Line2D([], [], color='k', linestyle='--', lw=0.8, label='No-skill baseline')
    ], fontsize=7, loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict, task: str) -> None:
    """Print mean ± std for all metrics across every model and temporal mode."""
    print(f"\n{'='*95}")
    print(f"  {TASK_TITLES[task]} — mean +/- std (5x3 repeated stratified CV)")
    print('='*95)
    header = (f"{'Model':<10} {'Mode':<8} {'ROC-AUC':>13} {'PR-AUC':>13} "
              f"{'Bal-Acc':>13} {'F1':>13} {'Sens':>13} {'Spec':>13}")
    print(header)
    print('-' * len(header))
    for model, modes in results.items():
        for mode, metrics in modes.items():
            def fmt(k):
                m, s = metrics[k]
                return f"{m:.3f}+/-{s:.3f}"
            print(f"{model:<10} {mode:<8} {fmt('roc_auc'):>13} {fmt('pr_auc'):>13} "
                  f"{fmt('balanced_accuracy'):>13} {fmt('f1'):>13} "
                  f"{fmt('sensitivity'):>13} {fmt('specificity'):>13}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments then load saved metrics and OOF predictions to produce all evaluation outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',        required=True, choices=['task1', 'task2'])
    parser.add_argument('--metrics_dir', default='outputs/metrics')
    parser.add_argument('--preds_dir',   default='outputs/predictions')
    parser.add_argument('--out',         default='outputs/plots')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    preds_dir   = Path(args.preds_dir)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_metrics(metrics_dir, args.task)
    if not results:
        print(f"No metrics found in {metrics_dir} for {args.task}. Run training first.")
        return

    print_summary_table(results, args.task)

    # Bar charts
    for metric in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'sensitivity'):
        path = out_dir / f'{args.task}_{metric}_bars.png'
        plot_metric_bars(results, args.task, metric, path)
        print(f"Saved -> {path}")

    # ROC and PR curves (only if OOF predictions exist)
    has_preds = any(
        (preds_dir / f'{args.task}_{m}_{mo}_oof.csv').exists()
        for m in results for mo in results[m]
    )
    if has_preds:
        path = out_dir / f'{args.task}_roc_curves.png'
        plot_roc_curves(results, preds_dir, args.task, path)
        print(f"Saved -> {path}")

        path = out_dir / f'{args.task}_pr_curves.png'
        plot_pr_curves(results, preds_dir, args.task, path)
        print(f"Saved -> {path}")
    else:
        print("No OOF predictions found — re-run training to generate ROC/PR curves.")


if __name__ == '__main__':
    main()

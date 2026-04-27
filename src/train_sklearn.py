"""Train logistic regression or gradient-boosted trees across all temporal modes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.datasets import get_dataset
from src.metrics import aggregate_fold_metrics, compute_metrics, format_results_table
from src.models.sklearn_models import build_model  # registers logreg + gbm as side-effect
from src.preprocessing import fit_transform, prepare_features
from src.splits import assert_no_leakage, get_cv
from src.wandb_utils import WandbLogger


def run(h5_path: str, task: str, config: dict, out_dir: Path,
        pred_dir: Path, use_wandb: bool) -> None:
    """Train a sklearn model across all temporal modes and save per-fold metrics and OOF predictions."""
    dataset    = get_dataset(config.get('dataset', 'adni'), h5_path, config)
    X_raw, y, _ = dataset.load_task(task)
    model_name  = config['model']

    cv = get_cv(
        n_splits=config['cv']['n_splits'],
        n_repeats=config['cv']['n_repeats'],
        random_state=config['cv']['random_state'],
    )

    all_results = {}

    for mode in config['temporal_modes']:
        X = prepare_features(X_raw, mode)
        fold_metrics = []
        oof_rows: list[dict] = []

        wandb_cfg = {**config, 'task': task, 'model': model_name, 'mode': mode}
        with WandbLogger(use_wandb, project='alzheimers-methylation',
                         name=f'{task}_{model_name}_{mode}', config=wandb_cfg) as logger:

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                assert_no_leakage(train_idx, val_idx)

                X_train_s, X_val_s, _ = fit_transform(X[train_idx], X[val_idx])
                y_train, y_val        = y[train_idx], y[val_idx]

                model  = build_model(config)
                model.fit(X_train_s, y_train)
                y_prob = model.predict_proba(X_val_s)[:, 1]

                metrics = compute_metrics(y_val, y_prob)
                fold_metrics.append(metrics)
                logger.log({f'fold/{k}': v for k, v in metrics.items()}, step=fold_idx)

                for yt, yp in zip(y_val.tolist(), y_prob.tolist()):
                    oof_rows.append({'fold': fold_idx, 'y_true': yt, 'y_prob': yp})

            aggregated = aggregate_fold_metrics(fold_metrics)
            all_results[mode] = aggregated
            logger.log_summary({k: v[0] for k, v in aggregated.items()})

        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(oof_rows).to_csv(
            pred_dir / f'{task}_{model_name}_{mode}_oof.csv', index=False
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{task}_{model_name}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  {task.upper()} | {model_name.upper()}")
    print('='*70)
    print(format_results_table(all_results))
    print(f"\nSaved -> {out_path}")


def main() -> None:
    """Parse CLI arguments and run sklearn model training for a single task."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',   required=True)
    parser.add_argument('--h5',     required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out',    default='outputs/metrics')
    parser.add_argument('--preds',  default='outputs/predictions')
    parser.add_argument('--wandb',  action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(args.h5, args.task, config, Path(args.out), Path(args.preds), args.wandb)


if __name__ == '__main__':
    main()

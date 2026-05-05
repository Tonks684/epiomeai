"""Train MLP across all temporal modes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.datasets import get_dataset
from src.metrics import aggregate_fold_metrics, compute_metrics, format_results_table
from src.models import get_model
from src.preprocessing import fit_transform, prepare_features
from src.splits import assert_no_leakage, get_cv
from src.wandb_utils import WandbLogger


def run(h5_path: str, task: str, config: dict, out_dir: Path,
        pred_dir: Path, use_wandb: bool) -> None:
    """Train the MLP across all temporal modes and save per-fold metrics and OOF predictions."""
    dataset      = get_dataset(config.get('dataset', 'adni'), h5_path, config)
    X_raw, y, _  = dataset.load_task(task)
    train_fold   = get_model('mlp', config)

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
        global_step = 0

        wandb_cfg = {**config, 'task': task, 'model': 'mlp', 'mode': mode}
        with WandbLogger(use_wandb, project='alzheimers-methylation',
                         name=f'{task}_mlp_{mode}', config=wandb_cfg) as logger:

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                assert_no_leakage(train_idx, val_idx)

                X_train_s, X_val_s, _ = fit_transform(X[train_idx], X[val_idx])
                y_train, y_val        = y[train_idx], y[val_idx]

                val_probs, history = train_fold(
                    X_train_s, y_train,
                    X_val_s,   y_val,
                    seed=config['cv']['random_state'] + fold_idx,
                )

                for entry in history:
                    logger.log({
                        'train_loss': entry['train_loss'],
                        'val_loss':   entry['val_loss'],
                        'fold':       fold_idx,
                    }, step=global_step)
                    global_step += 1

                metrics = compute_metrics(y_val, val_probs)
                fold_metrics.append(metrics)
                logger.log({f'fold/{k}': v for k, v in metrics.items()}, step=fold_idx)

                for yt, yp in zip(y_val.tolist(), val_probs.tolist()):
                    oof_rows.append({'fold': fold_idx, 'y_true': yt, 'y_prob': float(yp)})

            aggregated = aggregate_fold_metrics(fold_metrics)
            all_results[mode] = aggregated
            logger.log_summary({k: v[0] for k, v in aggregated.items()})

        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(oof_rows).to_csv(
            pred_dir / f'{task}_mlp_{mode}_oof.csv', index=False
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{task}_mlp.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  {task.upper()} | MLP")
    print('='*70)
    print(format_results_table(all_results))
    print(f"\nSaved -> {out_path}")


def main() -> None:
    """Parse CLI arguments and run MLP training for a single task."""
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

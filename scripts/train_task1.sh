#!/usr/bin/env bash
set -euo pipefail
H5=data/temporal_two_sets_n2000.h5
OUT=outputs/metrics
PREDS=outputs/predictions

python -m src.train_sklearn --task task1 --h5 $H5 --config config/task1_logreg.yaml --out $OUT --preds $PREDS ${WANDB:+--wandb}
python -m src.train_sklearn --task task1 --h5 $H5 --config config/task1_gbm.yaml    --out $OUT --preds $PREDS ${WANDB:+--wandb}
python -m src.train_torch   --task task1 --h5 $H5 --config config/task1_mlp.yaml    --out $OUT --preds $PREDS ${WANDB:+--wandb}

python -m src.evaluate --task task1 --metrics_dir $OUT --preds_dir $PREDS --out outputs/plots

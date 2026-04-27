#!/usr/bin/env bash
set -euo pipefail
H5=data/temporal_two_sets_n2000.h5
OUT=outputs/metrics
PREDS=outputs/predictions

python -m src.train_sklearn --task task2 --h5 $H5 --config config/task2_logreg.yaml --out $OUT --preds $PREDS ${WANDB:+--wandb}
python -m src.train_sklearn --task task2 --h5 $H5 --config config/task2_gbm.yaml    --out $OUT --preds $PREDS ${WANDB:+--wandb}
python -m src.train_torch   --task task2 --h5 $H5 --config config/task2_mlp.yaml    --out $OUT --preds $PREDS ${WANDB:+--wandb}

python -m src.evaluate --task task2 --metrics_dir $OUT --preds_dir $PREDS --out outputs/plots

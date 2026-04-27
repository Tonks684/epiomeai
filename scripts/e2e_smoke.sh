#!/usr/bin/env bash
set -euo pipefail

H5_PATH="${1:-data/temporal_two_sets_n2000.h5}"
OUT_ROOT="${2:-outputs/ci_smoke}"

METRICS_DIR="${OUT_ROOT}/metrics"
PREDS_DIR="${OUT_ROOT}/predictions"
PLOTS_DIR="${OUT_ROOT}/plots"

rm -rf "${OUT_ROOT}"
mkdir -p "${METRICS_DIR}" "${PREDS_DIR}" "${PLOTS_DIR}"

python -m src.train_sklearn \
  --task task1 \
  --h5 "${H5_PATH}" \
  --config config/ci_task1_logreg.yaml \
  --out "${METRICS_DIR}" \
  --preds "${PREDS_DIR}"

python -m src.train_torch \
  --task task1 \
  --h5 "${H5_PATH}" \
  --config config/ci_task1_mlp.yaml \
  --out "${METRICS_DIR}" \
  --preds "${PREDS_DIR}"

python -m src.evaluate \
  --task task1 \
  --metrics_dir "${METRICS_DIR}" \
  --preds_dir "${PREDS_DIR}" \
  --out "${PLOTS_DIR}"

test -f "${METRICS_DIR}/task1_logreg.json"
test -f "${METRICS_DIR}/task1_mlp.json"
test -f "${PREDS_DIR}/task1_logreg_t0_oof.csv"
test -f "${PREDS_DIR}/task1_mlp_t0_oof.csv"
test -f "${PLOTS_DIR}/task1_roc_curves.png"

echo "E2E smoke run completed successfully. Outputs in ${OUT_ROOT}."

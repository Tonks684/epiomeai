# Alzheimer's Disease Progression from DNA Methylation

Binary classification of Alzheimer's disease progression from peripheral blood DNA methylation data (ADNI). Two independent tasks:

- **Task 1:** CN → CN vs CN → MCI (190 individuals, 22.6% converters)
- **Task 2:** MCI → MCI vs MCI → Dementia (301 individuals, 36.5% converters)

Three model classes are evaluated across four temporal feature representations (t0, t1, concat, delta) using 5-fold × 3-repeat stratified cross-validation.

For full modelling rationale, results, and reflection see [`report.md`](report.md).

---

## Environment setup

### Option A — pip

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the project root on the Python path so `src.*` imports resolve:

```bash
export PYTHONPATH=$(pwd)          # Windows: set PYTHONPATH=%cd%
```

### Option B — Docker

```bash
# Run tests
docker compose run test

# Run training (mount your data file first — see Data section below)
docker compose run train bash scripts/train_task1.sh
```

---

## Data

Place the ADNI HDF5 file at:

```
data/temporal_two_sets_n2000.h5
```

The file is not included in this repository. To inspect its structure once placed:

```bash
python -m src.inspect_hdf5 --h5 data/temporal_two_sets_n2000.h5
```

---

## Running the tests

```bash
pytest tests/ -v
```

15 unit tests cover data loading correctness (shapes, beta-value range, NaN checks, label counts), split leakage, full-coverage cross-validation, and stratification preservation.

## End-to-end reproducibility check (local + CI)

Run a full smoke pipeline (train sklearn + train MLP + evaluate) with a lightweight deterministic config:

```bash
bash scripts/e2e_smoke.sh
```

This writes reproducibility artefacts to `outputs/ci_smoke/` and is the same command executed in CI.

## Continuous integration / continuous deployment

- **CI**: `.github/workflows/ci.yml` runs on every push and pull request, executes unit tests, runs the end-to-end smoke pipeline, and uploads smoke artefacts.
- **CD**: `.github/workflows/cd.yml` publishes a Docker image to GHCR on version tags (`v*`) or manual dispatch.

---

## Running model training

### One-shot scripts (all models, one task)

```bash
bash scripts/train_task1.sh    # logistic regression + GBM + MLP for Task 1
bash scripts/train_task2.sh    # same for Task 2
```

Each script trains all three model classes in sequence and then runs evaluation. Pass `WANDB=1` to enable Weights & Biases logging:

```bash
WANDB=1 bash scripts/train_task1.sh
```

### Individual model runs

```bash
# Sklearn models (logistic regression or GBM)
python -m src.train_sklearn \
    --task task1 \
    --h5   data/temporal_two_sets_n2000.h5 \
    --config config/task1_logreg.yaml \
    --out  outputs/metrics \
    --preds outputs/predictions

# MLP
python -m src.train_torch \
    --task task1 \
    --h5   data/temporal_two_sets_n2000.h5 \
    --config config/task1_mlp.yaml \
    --out  outputs/metrics \
    --preds outputs/predictions
```

Swap `--task task2` and the matching config file for Task 2.

### Config files

| File | Model | Task |
|---|---|---|
| `config/task1_logreg.yaml` | Logistic regression | Task 1 |
| `config/task1_gbm.yaml` | Gradient-boosted trees | Task 1 |
| `config/task1_mlp.yaml` | MLP | Task 1 |
| `config/task2_logreg.yaml` | Logistic regression | Task 2 |
| `config/task2_gbm.yaml` | Gradient-boosted trees | Task 2 |
| `config/task2_mlp.yaml` | MLP | Task 2 |

### Outputs

Training writes two artefacts per model run:

- `outputs/metrics/{task}_{model}.json` — mean and std of each metric across all 15 folds
- `outputs/predictions/{task}_{model}_{mode}_oof.csv` — out-of-fold predicted probabilities used for ROC and PR curves

---

## Modelling decisions and hyperparameter choices

### Temporal representation

Each individual has exactly two methylation timepoints (t0 = baseline, t1 = follow-up). Rather than using a sequence model, four explicit feature representations are compared:

| Mode | Description |
|---|---|
| `t0` | Baseline methylation profile (2,000 features) |
| `t1` | Follow-up methylation profile (2,000 features) |
| `concat` | t0 and t1 concatenated (4,000 features) |
| `delta` | t1 − t0 (2,000 features) |

The `temporal_modes` list in each config controls which modes are run.

### Model hyperparameters

**Logistic regression** (`config/task*_logreg.yaml`):
- `C: 1.0` — L2 regularisation strength; tuned conservatively for the high-dimensional low-sample regime
- `class_weight: balanced` — applied internally by sklearn

**Gradient-boosted trees** (`config/task*_gbm.yaml`):
- `max_iter: 200`, `learning_rate: 0.05`, `max_depth: 4`, `min_samples_leaf: 20`
- Shallow trees and high `min_samples_leaf` limit overfitting given the small sample size

**MLP** (`config/task*_mlp.yaml`):
- Architecture: input → 256 → 64 → 1 (2 hidden layers with BatchNorm, ReLU, Dropout 0.4)
- `lr: 0.001`, `weight_decay: 0.0001` (AdamW)
- `max_epochs: 200`, `patience: 15` (early stopping on validation loss)
- `pos_weight` computed per fold as n_negative / n_positive to handle class imbalance

### Validation strategy

5-fold repeated stratified cross-validation, 3 repeats (15 folds total). All metrics are reported as mean ± std across the 15 folds. A leakage assertion runs on every fold to confirm no individual appears in both train and validation sets. Preprocessing (StandardScaler) is fitted on training folds only.

---

## Running evaluation and interpreting outputs

```bash
python -m src.evaluate \
    --task task1 \
    --metrics_dir outputs/metrics \
    --preds_dir   outputs/predictions \
    --out         outputs/plots
```

This prints a summary table to stdout and saves the following plots to `outputs/plots/`:

| File | Description |
|---|---|
| `task1_roc_auc_bars.png` | ROC-AUC per model and temporal mode (mean ± std) |
| `task1_pr_auc_bars.png` | PR-AUC per model and temporal mode |
| `task1_balanced_accuracy_bars.png` | Balanced accuracy per model and mode |
| `task1_sensitivity_bars.png` | Sensitivity per model and mode |
| `task1_roc_curves.png` | ROC curves from pooled OOF predictions |
| `task1_pr_curves.png` | Precision-recall curves from pooled OOF predictions |

Replace `--task task1` with `--task task2` for Task 2 plots.

### Key results

The `delta` mode (methylation change between visits) is near-chance across all models, consistent with within-individual t0-t1 correlation of ~0.998. Baseline-level features (`t0`) carry the predictive signal.

Logistic regression is a strong baseline. The MLP improves on sensitivity (correctly identifying converters), particularly on Task 1. Gradient-boosted trees underperform both in this high-dimensional low-sample regime.

Full results tables, interpretation, and discussion of limitations are in [`report.md`](report.md).

---

## AI tool usage

Claude (Anthropic) was used throughout this project for code generation, architecture exploration, and the written report. Specifically:

- **Data exploration**: Claude helped interpret the HDF5 structure and identify that labels were pre-defined by group membership rather than requiring construction from trajectories.
- **Architecture decisions**: The switch from a temporal transformer to explicit temporal modes (t0, t1, concat, delta) was developed through discussion with Claude after examining the within-individual correlation structure.
- **Code generation**: Improved my training loops, cross-validation scaffolding, evaluation plots, and the dataset/model registry pattern were generated with Claude and reviewed/tested manually.
- **Report writing**: The analytical report was drafted collaboratively with Claude, with all scientific claims and results verified against the actual model outputs.

The model architecture, evaluation methodology, and all reported results are the my own.

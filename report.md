# Analytical Report: Alzheimer's Disease Progression from DNA Methylation


## Key terms

1. **DNA methylation** is a chemical modification where a methyl group (-CH3) is added to DNA, usually at CpG sites (regions where a cytosine is followed by a guanine). It changes with age, environment and disease and is part of epigenetics.
2. **Peripheral blood** means blood drawn from the body, rather than brain tissue. Studying the brain directly is hard in living patients so use blood as proxy.

## 1. Problem Framing

### 1.1 Task definition

The goal is to predict whether an individual will progress in their Alzheimer's disease status based on peripheral blood DNA methylation data. This is structured as two independent binary classification problems:

- **Task 1:** Given a methylation profile from a cognitively normal (CN) individual, predict whether they will later be diagnosed with Mild Cognitive Impairment (MCI).
- **Task 2:** Given a methylation profile from an MCI individual, predict whether they will later progress to dementia.

These are treated as independent tasks because the populations, feature panels, class distributions, and clinical questions differ between them. A single multi-state model would conflate these distinctions without obvious benefit at this sample size.

### 1.2 Prediction target

Each individual is assigned a binary label:

- `0` = non-converter (status is stable across the observation window)
- `1` = converter (individual progresses to the next diagnostic stage)

Labels are determined by group membership in the HDF5 file rather than being derived from raw trajectory data. The file partitions individuals into four pre-defined groups: `X_cn_to_cn`, `X_cn_to_mci`, `X_mci_to_mci`, `X_mci_to_dem`. This means the label construction decisions - which visits to use, how to handle reversion, how to define conversion - were made upstream by the data provider. This simplifies the modelling pipeline considerably and removes ambiguity around edge cases, but it also means that reported performance is conditional on the data provider's labelling choices.

### 1.3 Assumptions and constraints

| Assumption | Basis |
|---|---|
| Labels are correctly assigned by the HDF5 grouping | No ground-truth trajectory data available to verify |
| The two methylation panels (cpg_ids_cn, cpg_ids_mci) are appropriate for their respective tasks | Pre-selection was performed by the data provider using variance ratio between converters and non-converters |
| The 2,000 features per task are sufficient for the classification problem | Given as the starting point; full CpG space not available for comparison |
| Peripheral blood methylation reflects CNS-relevant biological variation | Standard assumption in ADNI-based methylation studies; acknowledged limitation |
| No batch or site correction is applied | No batch metadata available in the provided file |

---

## 2. Data Exploration

### 2.1 Dataset structure

The HDF5 file contains exactly four feature matrices, one per outcome group:

| Group | Shape (features, timepoints, individuals) | Task | Label |
|---|---|---|---|
| X_cn_to_cn | (2000, 2, 147) | Task 1 | 0 |
| X_cn_to_mci | (2000, 2, 43) | Task 1 | 1 |
| X_mci_to_mci | (2000, 2, 191) | Task 2 | 0 |
| X_mci_to_dem | (2000, 2, 110) | Task 2 | 1 |

This gives 190 individuals for Task 1 and 301 individuals for Task 2.

### 2.2 Key findings from EDA

**Beta-value range and quality.** All methylation values fall within [0, 1] as expected for beta values. There are no missing values in any group.

**Class balance.** Task 1 has a positive rate of 22.6% (43 converters out of 190). Task 2 has a positive rate of 36.5% (110 converters out of 301). Both tasks are imbalanced, with Task 1 more so. This makes accuracy a misleading metric and motivates the use of class-weighted losses, PR-AUC as the primary ranking metric, and explicit reporting of sensitivity. PR-AUC is preferred over ROC-AUC in imbalanced settings because it focuses entirely on the minority (converter) class: precision measures the fraction of predicted converters that are correct, and recall measures the fraction of true converters identified. ROC-AUC includes true negatives in its calculation via specificity, which inflates apparent performance when the negative class dominates — a model that predicts all-negative would achieve a high ROC-AUC but a PR-AUC equal to the base rate. Sensitivity is reported alongside PR-AUC because the clinical cost of missing a true converter (false negative) is substantially higher than the cost of a false alarm.
![](./outputs/eda/class_balance.png)
**Longitudinal structure.** Every individual has exactly two methylation samples (t0 and t1). Within-individual Pearson correlation across all 2,000 features is approximately 0.998, and the mean absolute change per CpG between visits is approximately 0.015–0.020. This means methylation is highly stable over the observation window. The delta (t1 − t0) carries very little within-individual variation, which foreshadowed the poor predictive performance of the delta temporal mode observed during modelling.

**CpG panel overlap.** The two feature panels (cpg_ids_cn for Task 1, cpg_ids_mci for Task 2) have substantial but incomplete overlap. Features are task-specific selections and cross-task feature comparison is not meaningful.

**Feature pre-selection leakage.** The 2,000 features per task were selected globally using variance ratio between converters and non-converters across the full dataset. This means label information was used before any train/test split, and reported AUCs will be optimistically biased relative to a fully independent discovery pipeline. This is acknowledged as a limitation throughout; the full CpG space is not available for fold-wise reselection.

---

## 3. Modelling Decisions

### 3.1 Temporal framing: why we do not use a sequence model

The dataset has exactly two timepoints per individual. Initial plans included a temporal transformer to model the visit sequence. On inspection, this is not well-motivated:

- With T = 2, attention over visits reduces to a weighted combination of two embeddings. There are no multi-step dependencies to capture.
- Within-individual t0-t1 correlation is ~0.998. The visits carry almost identical information.
- The more scientifically useful question — given two timepoints, does the change or the level carry the signal? — is better answered by an explicit temporal ablation than by a learned attention weight.

The temporal transformer was therefore replaced by four explicit input modes:

| Mode | Feature representation | Interpretation |
|---|---|---|
| t0 | Baseline methylation profile (2,000 features) | Can baseline level predict future conversion? |
| t1 | Follow-up methylation profile (2,000 features) | Is the follow-up visit more informative than baseline? |
| concat | Concatenation of t0 and t1 (4,000 features) | Do both timepoints together improve prediction? |
| delta | t1 − t0 (2,000 features) | Does the rate of methylation change carry signal? |

This ablation directly answers the clinical question about whether longitudinal information adds value, and does so transparently rather than leaving it implicit in learned attention weights.

### 3.2 Model ladder

Three model classes were trained across all four temporal modes:

**Logistic regression (L2, class_weight='balanced').** The primary baseline. Strong regularisation is well-suited to the high-dimensional, low-sample regime (190–301 individuals, 2,000–4,000 features). L2 regularisation was preferred over L1 because DNA methylation features are not independent — CpGs in the same genomic region are correlated through shared regulatory context. L2 shrinks correlated features together rather than arbitrarily zeroing one out, which is more appropriate when the discriminative signal is spread across a region rather than concentrated in isolated sites. Interpretable coefficients map directly to CpG-level feature importance.

**Gradient-boosted trees (HistGradientBoostingClassifier).** A nonlinear tabular baseline included to test whether feature interactions add value beyond the logistic regression. The expectation going in was that it would underperform logistic regression in this regime, given its tendency to overfit on small samples with many features.

**Multi-layer perceptron (MLP, 2 hidden layers, dropout, AdamW, early stopping).** A compact neural baseline. Deliberately small architecture (2000 → 256 → 64 → 1) with strong regularisation (dropout 0.4, weight decay 1e-4) and early stopping on validation loss. The architecture was kept small deliberately: with 190–301 individuals, a larger network would memorise training folds before generalising. Two hidden layers were sufficient to model interactions between methylation regions without the parameter count growing faster than the dataset. The bottleneck from 256 to 64 forces the network to compress region-level representations before the final classification step. Dropout 0.4 and weight decay were set conservatively to prevent the positive class — already a minority — from being overfit to noise in individual training folds.

### 3.3 Validation strategy

A single train/validation/test split was rejected for two reasons. First, Task 1 has only 43 converters; a holdout set would contain approximately 8–9 positive examples, making AUC estimates unreliable. Second, repeated cross-validation produces a credible mean and standard deviation across folds, making it clear whether results are stable or artefactual.

**Chosen strategy: 5-fold repeated stratified cross-validation, 3 repeats (15 folds total).**

- Stratification preserves class ratio in each fold. Without it, a fold could by chance contain very few or zero converters — particularly in Task 1, where only 43 of 190 individuals are positive. A fold with no positive examples cannot produce a valid AUC estimate at all; a fold with one or two produces an estimate with extreme variance that dominates the reported mean.
- Since each individual appears exactly once in the dataset (one row per individual), standard stratified CV is equivalent to individual-level splitting. No group-level CV is needed. If individuals had contributed multiple rows — for example, one row per visit — standard CV would risk placing different rows from the same individual in both train and validation, leaking individual-level signal across the split.
- Leakage is verified by assertion: no index appears in both train and validation sets in any fold.

All metrics are reported as mean ± standard deviation across the 15 folds.

### 3.4 Preprocessing

Within each fold:
1. A `StandardScaler` is fitted on the training split only. Fitting on the full dataset before splitting would leak distributional information — the scaler's mean and variance would encode the hold-out data's statistics, making the model appear to perform better than it would on truly unseen data. This is a subtle form of leakage that is easy to miss but inflates reported performance.
2. The scaler is applied to the validation split without refitting.
3. No imputation is needed (no missing values).
4. For tree-based models, scaling does not affect the result but is applied for consistency so that all model comparisons operate on the same preprocessed input.

### 3.5 Class imbalance handling

- Logistic regression: `class_weight='balanced'` (sklearn reweights each class by inverse frequency).
- GBM: `class_weight='balanced'`.
- MLP: `pos_weight` computed per fold as `n_negative / n_positive`.

Loss weighting was preferred over resampling approaches such as SMOTE or random undersampling. SMOTE generates synthetic minority samples by interpolating between real ones — but with only 43 converters in Task 1, synthetic samples would closely resemble the real data and create artificial redundancy within folds rather than independent signal. Undersampling discards real majority-class data from an already small dataset, reducing the information available to the model. Loss weighting achieves the same calibration effect on the decision boundary without modifying the training data itself.

---

## 4. Results and Interpretation

### 4.1 Task 1 (CN → MCI)

| Model | Mode | ROC-AUC | PR-AUC | Bal. Accuracy | Sensitivity |
|---|---|---|---|---|---|
| Logistic regression | t0 | 0.948 ± 0.036 | 0.879 ± 0.078 | 0.703 ± 0.077 | 0.413 ± 0.152 |
| Logistic regression | t1 | 0.941 ± 0.051 | 0.891 ± 0.088 | 0.779 ± 0.074 | 0.569 ± 0.141 |
| Logistic regression | concat | 0.945 ± 0.050 | 0.888 ± 0.094 | 0.776 ± 0.095 | 0.564 ± 0.188 |
| Logistic regression | delta | 0.572 ± 0.094 | 0.394 ± 0.121 | 0.548 ± 0.061 | 0.131 ± 0.119 |
| GBM | best (concat) | 0.725 ± 0.079 | 0.528 ± 0.124 | 0.618 ± 0.069 | 0.279 ± 0.135 |
| MLP | t0 | **0.970 ± 0.025** | **0.926 ± 0.054** | 0.881 ± 0.062 | 0.803 ± 0.115 |
| MLP | t1 | 0.952 ± 0.045 | 0.909 ± 0.082 | 0.879 ± 0.071 | 0.794 ± 0.121 |
| MLP | concat | 0.953 ± 0.044 | 0.908 ± 0.071 | 0.884 ± 0.054 | 0.809 ± 0.095 |
| MLP | delta | 0.657 ± 0.100 | 0.457 ± 0.118 | 0.615 ± 0.070 | 0.548 ± 0.174 |

### 4.2 Task 2 (MCI → Dementia)

| Model | Mode | ROC-AUC | PR-AUC | Bal. Accuracy | Sensitivity |
|---|---|---|---|---|---|
| Logistic regression | t0 | 0.882 ± 0.040 | 0.854 ± 0.056 | 0.792 ± 0.077 | 0.664 ± 0.143 |
| Logistic regression | t1 | 0.852 ± 0.043 | 0.823 ± 0.051 | 0.774 ± 0.044 | 0.648 ± 0.085 |
| Logistic regression | concat | **0.897 ± 0.029** | **0.870 ± 0.038** | 0.801 ± 0.036 | 0.664 ± 0.083 |
| Logistic regression | delta | 0.498 ± 0.056 | 0.392 ± 0.046 | 0.506 ± 0.046 | 0.303 ± 0.090 |
| GBM | best (t1) | 0.733 ± 0.056 | 0.624 ± 0.078 | 0.633 ± 0.052 | 0.409 ± 0.094 |
| MLP | t0 | 0.891 ± 0.039 | 0.867 ± 0.047 | 0.815 ± 0.058 | 0.767 ± 0.098 |
| MLP | concat | 0.902 ± 0.028 | 0.875 ± 0.038 | 0.831 ± 0.040 | 0.770 ± 0.071 |
| MLP | delta | 0.562 ± 0.049 | 0.444 ± 0.051 | 0.530 ± 0.049 | 0.548 ± 0.170 |

### 4.3 Interpretation

**The delta mode is consistently uninformative.** Across both tasks and all three model classes, the delta (t1 − t0) input produces near-chance or substantially degraded performance. This confirms what the EDA suggested: with a within-individual t0-t1 correlation of ~0.998 and a mean absolute change of ~0.02 per CpG, the two visits are nearly identical. Methylation change over the observation window does not discriminate converters from non-converters; the baseline level does.

**Logistic regression is a strong baseline.** In this high-dimensional, small-sample regime (190–301 individuals, 2,000–4,000 features), strongly regularised logistic regression is highly competitive. It outperforms the gradient-boosted tree model substantially on both tasks, consistent with the expectation that tree models overfit when the number of features far exceeds the sample size.

**The MLP improves on logistic regression, especially on sensitivity.** On Task 1, the MLP achieves a mean ROC-AUC of 0.970 versus 0.948 for logistic regression, and a mean sensitivity of 0.803 versus 0.413. This improvement in sensitivity — the ability to correctly identify converters — is particularly relevant clinically. The MLP's nonlinear interactions and per-fold early stopping appear to recover signal that regularised linear regression misses, despite the small sample size. A plausible mechanism: the discriminative methylation signal is likely not uniformly distributed across all 2,000 features. A small subset of CpGs may carry most of the conversion-relevant signal, and their joint effect may be nonlinear — for instance, a combination of hypermethylation at one region and hypomethylation at another may be more informative than either feature alone. Logistic regression cannot model this kind of interaction; the MLP can learn it implicitly through its hidden layers.

**Gradient boosting underperforms throughout.** This is consistent with the data regime: tree ensembles are not well-suited to problems where the number of features greatly exceeds the number of samples. The gradient-boosted model achieved at most ROC-AUC 0.733 on Task 1 and 0.733 on Task 2, substantially below both logistic regression and the MLP.

**Adding t1 to t0 (concat mode) offers modest gains.** On Task 2, the best logistic regression result uses concatenated features (0.897) versus t0 alone (0.882). The improvement is meaningful but modest, suggesting that t1 adds some incremental information while being largely redundant with t0.

---

## 5. Challenges and Design Decisions

### 5.1 Discovering that labels are pre-defined

The initial approach document invested considerable effort in planning label construction from longitudinal trajectories: handling reversion, excluding ambiguous cases, choosing which visit to use as the prediction time point. On inspecting the HDF5 file, it became clear that labels are implicit in the group structure. This simplified the pipeline but also meant that the data provider's labelling assumptions could not be audited or varied.

### 5.2 Choosing the temporal representation

The initial plan included a temporal transformer. Two findings led to a different decision. First, every individual has exactly two timepoints — not the variable-length sequences of 2–4 that were assumed. Second, within-individual correlation across the two visits is ~0.998. At T = 2 with near-identical timepoints, transformer attention reduces to a single learned scalar weight. The four explicit temporal modes (t0, t1, concat, delta) provide the same information as any learned temporal combination, but in a fully interpretable and directly comparable form.

### 5.3 Cross-validation over a single split

Task 1 has 43 positive cases. A 60/20/20 single split would produce a test set with approximately 8–9 converters, far too few for stable AUC estimates. Repeated stratified cross-validation (5 folds × 3 repeats) was used instead. This adds computational cost but substantially increases the reliability of reported metrics.

### 5.4 Feature selection leakage

The 2,000 CpG features per task were selected using variance ratio between converters and non-converters computed on the full dataset — before any train/test partition. This is a form of label leakage that will optimistically inflate all reported performance figures. Since the full CpG matrix is not provided, fold-wise reselection is not possible. This limitation is explicitly stated alongside all results and should be kept in mind when interpreting the absolute AUC values.

---

## 6. Repository Evolution

This section documents how the codebase has developed over the course of the project. It is intended to show not just what was built, but the reasoning behind each architectural decision as new information emerged.

### Phase 1 — Initial build (data-first)

The first version of the repository was built immediately after inspecting the HDF5 file. Rather than starting from a pre-planned architecture, the design was driven by what the data actually contained.

Key discoveries at this stage:
- Labels are pre-defined by group membership in the HDF5 file. The planned label-construction pipeline (trajectory reconstruction, reversion handling, visit selection) was unnecessary.
- Every individual has exactly two timepoints with a within-individual correlation of ~0.998. This ruled out the planned temporal transformer, which requires meaningful sequence length to make attention interpretable.
- The delta mode (t1 − t0) was predicted to be uninformative given the near-identical visits. This was later confirmed empirically.

These findings shaped the core pipeline: four explicit temporal modes (t0, t1, concat, delta) as a transparent ablation instead of a learned temporal model, and repeated stratified cross-validation instead of a single split due to the small number of converters in Task 1.

Components built in this phase:
- `src/data.py` — HDF5 loading with axis transposition
- `src/splits.py` — repeated stratified CV with leakage assertion
- `src/preprocessing.py` — temporal mode preparation and per-fold scaling
- `src/models/sklearn_models.py` — logistic regression and GBM
- `src/models/mlp.py` — compact MLP with early stopping
- `src/train_sklearn.py` and `src/train_torch.py` — training loops
- `src/metrics.py` — PR-AUC-first evaluation suite
- `tests/` — 15 unit tests covering data loading, split correctness, and label counts

### Phase 2 — Experiment tracking and evaluation (results-first)

After confirming that training ran correctly, the focus shifted to making results reproducible, trackable, and visualisable.

Key additions:
- **Weights & Biases integration** (`src/wandb_utils.py`). A thin no-op wrapper makes W&B opt-in via `--wandb` flag without coupling the training loop to the logging library. sklearn runs log per-fold metrics and aggregated summaries; MLP runs additionally log per-epoch train and validation loss curves.
- **OOF prediction saving.** Training scripts now save out-of-fold predicted probabilities to CSV alongside aggregated metrics JSON. This enables post-hoc ROC and precision-recall curve generation from all 15 folds without re-running training.
- **`src/evaluate.py`** rewritten to load OOF predictions and produce ROC curves, PR curves, and bar charts comparing all model and temporal mode combinations.
- **Docker support** (`Dockerfile`, `docker-compose.yml`). The training and test environments are fully containerised, with data and outputs mounted as volumes so nothing is baked into the image.

### Phase 3 — Dataset and model abstraction (scale-first)

With the core pipeline stable and results validated, the architecture was refactored to support new datasets and models without editing source files.

**Problem:** `src/data.py` contained ADNI-specific HDF5 keys, axis conventions, and task definitions hardcoded in Python. Adding a second dataset (e.g. ROSMAP, AIBL) would require editing source files, which is fragile.

**Solution — dataset interface and registry:**

A `BaseDataset` abstract class (`src/datasets/base.py`) defines the interface that any loader must implement: `load_task(task) -> (X, y, feature_ids)` where `X` is `(N, T, D)`. The ADNI-specific logic moves to `src/datasets/adni.py`, which registers itself via a decorator:

```python
@register_dataset('adni')
class AdniDataset(BaseDataset): ...
```

Task definitions move from hardcoded Python dictionaries to a `tasks` block in the YAML config, so a new dataset is a config change, not a code change. The `dataset` key in the config selects which loader to instantiate:

```yaml
dataset: adni
tasks:
  task1:
    negative: X_cn_to_cn
    positive: X_cn_to_mci
    feature_ids: cpg_ids_cn
```

**Solution — model registry:**

`src/models/registry.py` provides a `@register_model` decorator. Each model builder registers itself on import:

```python
@register_model('logreg')
def build_logreg(config): ...
```

Training scripts call `get_model(config['model'], config)` without any `if/elif` chain. Adding a new model means creating a new file with a `@register_model` decorator — no changes to training scripts.

**Backward compatibility:** `src/data.py` is retained as a thin shim that delegates to `AdniDataset`, so existing tests and the `inspect_hdf5.py` script continue to work unchanged.

### What would be built next

With data scale as the driver, the next phase of development would follow this order:

1. **Survival analysis module** — a `src/train_survival.py` script using `pycox` or `lifelines` to model time-to-conversion rather than a binary label. This is the highest-impact methodological change that does not require new data.
2. **Composable preprocessing pipeline** — replace the flat `src/preprocessing.py` with a sklearn `Pipeline` of configurable steps (scaler, imputer, optional feature selector) driven by config. This becomes important when datasets have missing values or require batch correction.
3. **Temporal models at scale** — with 5,000+ individuals and 5–10 timepoints per person, a small temporal transformer or GRU becomes well-motivated. The dataset interface already returns `X` as `(N, T, D)`, so sequence models can be added without changing data loading.
4. **Semi-supervised pre-training** — masked methylation modelling on the large unlabelled CpG pool, followed by fine-tuning on labelled conversion data. This is the most promising direction for methylation-specific representation learning.

---

## 7. What Would Be Improved With More Time

- **Fold-wise feature reselection.** If the full CpG matrix were available, rerunning variance-ratio selection within each training fold would give a much less biased performance estimate.
- **Clinical covariates.** Age, sex, APOE genotype, baseline cognitive scores, and medication use are known predictors of AD progression. Including them as additional features would likely improve performance and is the standard approach in ADNI-based studies. Their absence is also a confounding risk: APOE ε4 carrier status has a strong independent effect on conversion and is correlated with specific methylation patterns. A model trained without APOE genotype may learn to recognise methylation signatures that track APOE status rather than conversion itself — appearing predictive within ADNI while capturing the wrong underlying biology.
- **Blood cell type deconvolution.** Peripheral blood methylation is heavily confounded by variation in cell-type composition (e.g., neutrophil-to-lymphocyte ratio). Applying a reference-based deconvolution method (e.g., Houseman or EpiDISH) and including estimated cell proportions as covariates is standard practice in blood methylation studies. This was not possible here due to the absence of full CpG-space data.
- **Batch and site effects.** ADNI data are collected across multiple sites with different assay protocols and processing dates. Without batch metadata in the provided file, no correction could be applied.
- **Nested cross-validation.** Hyperparameter selection was performed using fixed values informed by the data regime rather than a search. This is a form of implicit optimisation — the fixed choices (regularisation strength, architecture size, dropout) were informed by general knowledge of the data type, which is a weaker form of the same selection pressure that nested CV is designed to control. Nested CV (outer folds for evaluation, inner folds for tuning) would give a less optimistic performance estimate by ensuring each model is evaluated on data completely unseen during any form of hyperparameter selection.
- **Survival analysis framing.** Reducing progression to a binary outcome discards information about how quickly individuals convert. A time-to-event model (e.g., Cox proportional hazards with methylation features) would give a clinically richer output and avoid the binary cutoff entirely.
- **External validation.** All evaluation is within ADNI. Validating on an independent cohort (e.g., ROSMAP, AIBL) would be the only true test of generalisability. This matters more than it would for a general benchmark because ADNI has known selection characteristics — participants are older, more educated, and predominantly white, with structured follow-up at academic medical centres. A model that performs well in ADNI may be capturing population-specific confounders rather than a biological signal that holds in a broader clinical population.

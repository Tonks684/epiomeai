# Presentation & Q&A Reference

**Format:** Part 1 is the 10-20 min talk track (narrative arc, section by section).  
Part 2 is the Q&A bank (indexed by theme, designed for fast lookup during questioning).  
Cross-references to `report.md` sections are in parentheses.

---

# Part 1 — Talk Track (10–20 min)

> **30-second elevator summary (memorise this):**  
> "Two binary classification tasks on ADNI methylation. The data has T=2 with within-individual r=0.998, so I replaced the planned temporal transformer with explicit t0/t1/concat/delta modes. Best results: MLP at ROC-AUC 0.970 on Task 1, 0.902 on Task 2. Delta is uninformative across all models — confirming the EDA prediction. All numbers are inflated by upstream feature pre-selection leakage; the relative comparisons are robust, the absolute AUCs are not."

---

**Suggested plots to project** (from `outputs/`):  
- `eda/longitudinal_delta.png` — the killer EDA finding (delta distributions are identical between groups)  
- `plots/task1_roc_auc_bars.png` + `plots/task2_roc_auc_bars.png` — model comparison at a glance  
- `eda/feature_variance.png` — motivates the leakage discussion

---

### 1. Framing (~30 s)

> "The task was to predict Alzheimer's disease progression from DNA methylation data as two independent binary classification problems — cognitively normal individuals who develop MCI, and MCI individuals who progress to dementia. The HDF5 file was the starting point. Everything I built was driven by what the data actually contained."

### 2. The pivotal EDA findings (~3 min)

_Everything downstream followed from three discoveries. Refer to `outputs/eda/` plots if presenting visually._

**Finding 1 — Exactly 2 timepoints per individual, not variable-length sequences.**  
- Task 1: 190 individuals (147 non-converter, 43 converter) — 22.6% positive rate.  
- Task 2: 301 individuals (191 non-converter, 110 converter) — 36.5% positive rate.  
- Fixed T = 2 rules out any need for padding/masking and makes sequence models unwarranted.

**Finding 2 — Methylation is nearly identical across the two visits.**  
- Within-individual Pearson r ≥ 0.998 (median) across all four groups.  
- Mean absolute change per CpG ≈ 0.015–0.020 on the [0, 1] beta scale.  
- Delta distributions for converters and non-converters are visually indistinguishable.  
- Prediction → the delta temporal mode will carry no discriminative signal. Confirmed empirically.

**Finding 3 — Class imbalance requires deliberate metric and training choices.**  
- Task 1 at 22.6% means a naive all-negative model achieves 77.4% accuracy. Accuracy is meaningless.  
- PR-AUC is the primary metric: it focuses on the minority (converter) class. ROC-AUC inflates via true negatives when negatives dominate.  
- Sensitivity is reported separately: missing a true converter has higher clinical cost than a false alarm.

### 3. Decisions that fell out from the EDA (~3 min)

_"The EDA didn't just describe the data — it collapsed several planned design choices."_

**Temporal representation** (report §3.1–3.2)  
The initial plan included a temporal transformer. With T = 2 and r = 0.998, attention reduces to a single scalar weight — no multi-step dependency exists. Replaced by four explicit temporal modes:

| Mode | Representation | Clinical question answered |
|---|---|---|
| t0 | Baseline methylation (2,000 features) | Can level at first visit predict conversion? |
| t1 | Follow-up methylation (2,000 features) | Is the later visit more informative? |
| concat | t0 ∥ t1 (4,000 features) | Do both timepoints together add up? |
| delta | t1 − t0 (2,000 features) | Does rate of change carry signal? |

This is a transparent ablation that directly answers the longitudinal information question rather than hiding it in a learned weight.

**Validation strategy** (report §3.3)  
A single holdout was rejected: 43 converters × 20% hold-out ≈ 8–9 positives, far too few for stable AUC estimates. Used **5-fold repeated stratified CV, 3 repeats (15 folds total)**. Stratification is essential — a fold with zero converters cannot produce a valid AUC. Individual-level split is equivalent here since each individual has exactly one row.

**Model ladder** (report §3.2)  
- **Logistic regression (L2, balanced weights):** primary baseline; well-suited to the high-D, low-N regime.  
- **HistGradientBoosting (balanced weights):** nonlinear tabular check; expected to overfit — included to confirm.  
- **MLP (2000 → 256 → 64 → 1, BatchNorm, Dropout 0.4, AdamW, early stopping):** compact neural baseline.  

Class imbalance handled via loss weighting not resampling: SMOTE with 43 converters creates artificial redundancy from the same sparse minority; undersampling discards already-scarce majority data.

### 4. Results — headline numbers (~3 min)

_"The delta prediction was confirmed. Logistic regression was a strong baseline. The MLP improved substantially on sensitivity."_

**Task 1 (CN → MCI) — best results per model:**

| Model | Mode | ROC-AUC | PR-AUC | Sensitivity |
|---|---|---|---|---|
| Logistic regression | t0 | 0.948 ± 0.036 | 0.879 ± 0.078 | 0.413 ± 0.152 |
| Logistic regression | t1 | 0.941 ± 0.051 | 0.891 ± 0.088 | 0.569 ± 0.141 |
| GBM | concat | 0.725 ± 0.079 | 0.528 ± 0.124 | 0.279 ± 0.135 |
| **MLP** | **t0** | **0.970 ± 0.025** | **0.926 ± 0.054** | **0.803 ± 0.115** |
| (delta, any model) | — | ~0.57–0.66 | ~0.39–0.46 | — |

**Task 2 (MCI → Dementia) — best results per model:**

| Model | Mode | ROC-AUC | PR-AUC | Sensitivity |
|---|---|---|---|---|
| Logistic regression | concat | 0.897 ± 0.029 | 0.870 ± 0.038 | 0.664 ± 0.083 |
| GBM | t1 | 0.733 ± 0.056 | 0.624 ± 0.078 | 0.409 ± 0.094 |
| **MLP** | **concat** | **0.902 ± 0.028** | **0.875 ± 0.038** | **0.770 ± 0.071** |
| (delta, any model) | — | ~0.50–0.56 | ~0.39–0.44 | — |

Key takeaways to state explicitly:
1. Delta is consistently uninformative — confirms the EDA prediction.
2. Logistic regression beats GBM substantially in this regime (high-D, low-N).
3. MLP's main gain over logistic regression is **sensitivity**, not ROC-AUC. The discrimination gap is small; the recall gap is large. This is a threshold/calibration story (see Q&A §Results).
4. Adding t1 to t0 (concat) offers modest but real gains on Task 2; on Task 1, t0 alone is effectively optimal.
5. All numbers are **optimistically biased** by pre-selection leakage (see §5).

### 5. Limitations (~3–4 min)

_"The most important limitation is the one baked in from the start — but there are four others that matter for interpreting clinical relevance."_ (report §2.2 and §5.4)

**1. Feature pre-selection leakage — the dominant bias.**  
The 2,000 CpGs were selected using a variance ratio between converters and non-converters on the **full dataset**, before any train/test split. This is label-leaking feature selection: the model never sees unseen CpGs, but the 2,000 it does see were chosen because they already discriminate groups. Every reported AUC is optimistically inflated. Studies comparing global versus fold-wise feature selection on similar high-D, low-N methylation problems have reported AUC drops of 0.05–0.15 or more. I would not cite 0.970 as a generalisation estimate.

Two things I want to be clear about: first, the **relative comparisons** between models and between temporal modes are still valid — all models train on the same biased feature set, so the bias is constant across comparisons. MLP > LogReg > GBM, and t0/t1 >> delta, are robust findings. Second, fold-wise reselection cannot be done without the full CpG matrix; it is not a gap in the pipeline design but a constraint of the provided data.

**2. Missing clinical covariates — especially APOE ε4.**  
APOE ε4 is the strongest known genetic risk factor for late-onset Alzheimer's disease, with substantially elevated conversion rates in carriers. Crucially, APOE genotype also correlates with specific methylation patterns. Without including APOE as a covariate or stratification variable, the model may be learning methylation signatures that proxy for APOE status rather than for conversion itself. That would inflate ADNI performance while the model fails to generalise to populations with a different APOE ε4 prevalence. Age and sex are similarly important methylation confounders — methylation clocks are age-dependent, and sex-linked CpGs show systematic differences. None of these were provided in the HDF5 metadata.

**3. Cell-type deconvolution — the blood methylation confound.**  
Peripheral blood methylation is a mixture signal: monocytes, T-cells, B-cells, neutrophils, and other cell types each carry distinct methylation profiles, and the proportions of these cell types are known to differ between AD cases and controls. Without deconvolution (e.g., Houseman reference-based correction using the Reinius reference panel), a proportion of the discriminative signal the model captures may reflect cell-type shifts rather than neurodegenerative-relevant CpG changes. This is a standard preprocessing step in blood methylation studies that was not applied because cell-type reference data was not provided.

**4. No batch correction — unknowable effect.**  
Illumina EPIC and 450k arrays are susceptible to systematic technical variation by processing batch, scanner, and date of hybridisation. The provided HDF5 contains no array or batch metadata. If samples from converters and non-converters were processed in different batches — even partially — the model can learn batch-associated CpG patterns that spuriously separate groups. Since the metadata is unavailable, the direction and magnitude of this effect cannot be assessed. For any follow-on study, batch metadata would be the first thing to request.

**5. External validity — ADNI is not the general population.**  
All evaluation is within ADNI, which selects for older, highly educated, predominantly white, English-speaking individuals who are willing and able to participate in longitudinal studies. Conversion rates and methylation profiles may differ substantially in a clinical or population-representative cohort. The numbers here are internally valid given the provided data, but they are not estimates of deployment performance.

### 6. What I'd do next (~2 min)

_"In priority order by expected return on validity and clinical relevance:"_

**1. Fold-wise feature reselection** — the highest-impact methodological fix. Re-run variance-ratio feature selection inside each CV fold using only training-fold labels, then evaluate on the held-out fold using only those fold-specific features. This removes the label leakage entirely and gives unbiased AUC estimates. It requires access to the full CpG matrix (~850k CpGs on the EPIC array), which was not part of the provided data. With the full matrix, this is a two-line change to the `src/splits.py` loop — the pipeline is already structured to support it.

**2. Add clinical covariates** — specifically APOE ε4 genotype, age, and sex as additional input features or as stratification variables. APOE ε4 alone could substantially improve discrimination and would also reveal whether methylation adds predictive value beyond genotype — which is the scientifically interesting question. This would require linking the HDF5 subject IDs to ADNI clinical metadata, which is available through the ADNI data portal.

**3. Ensemble LogReg + MLP** — a natural, low-cost extension. Since OOF predictions are already saved for all 15 folds and both models, averaging predicted probabilities from LogReg (stable, well-ranked) and MLP (higher sensitivity) is a post-hoc step requiring no retraining. LogReg and MLP make different errors — their combination typically improves PR-AUC and reduces variance across folds.

**4. Calibration and threshold selection** — the current probabilities from class-weighted training are shifted from the true prevalence and should not be interpreted as conversion probabilities. Platt scaling or isotonic regression (fit on a held-out subset using nested CV) would give calibrated probabilities. Separately, threshold optimisation using Youden's J or a cost-weighted criterion on a held-out fold would close the sensitivity gap seen in logistic regression without requiring a different model.

**5. Survival analysis** — replace the binary label with time-to-conversion modelling using a Cox proportional hazards model or a discrete-time model (e.g., `pycox`). The current binary label discards information about *when* conversion happens within the observation window, and treats individuals censored before conversion (not yet converted) identically to true non-converters. A survival model handles censoring correctly and produces clinically richer outputs: a predicted hazard curve over time rather than a single probability.

**6. External validation** — apply the ADNI-trained model to ROSMAP (Rush University Memory and Aging Project) or AIBL (Australian Imaging, Biomarker & Lifestyle study), both of which have longitudinal blood methylation data in AD populations. The main practical barrier is CpG panel harmonisation across array generations. This is the step that converts an internal validation result into a generalisable finding.

---

# Part 2 — Q&A Bank

## Theme A: Data & EDA

**Q: The brief says 1,905 samples from 649 individuals. Your HDF5 has 491 individuals with 2 timepoints = 982 samples. Where did the rest go?**  
> The HDF5 is a curated subset provided as the starting point. I worked with what was provided and did not attempt to reconstruct the upstream filtering. Possible reasons for the reduction (158 individuals excluded from 649): missing follow-up visit, ambiguous or reverting diagnosis (CN → MCI → CN), insufficient observation window to assign a definitive group. The 1,905 sample count may reflect individuals with more than 2 visits in the full ADNI dataset before subsetting to exactly one baseline and one follow-up. I didn't have access to the full ADNI dataset to verify.

**Q: Why does the within-individual correlation of 0.998 matter so much?**  
> It tells you that the two methylation profiles per individual are almost informationally identical. The delta (t1 − t0) is therefore dominated by measurement noise, not biological signal. A model trained on delta is effectively predicting from noise. This was observed before any modelling and confirmed empirically in the results. It also means a sequence model has nothing to learn from the temporal dimension — any attention weight it assigns to t0 vs t1 is learning a meaningless distinction.

**Q: Is the 0.998 correlation biological stability or a technical artefact (e.g., same Illumina array for an individual's visits)?**  
> Both are plausible. Methylation at established CpG sites is genuinely stable over months to a few years in peripheral blood — this is well-documented. However, if the same array or processing batch was used for both visits per individual, within-individual technical consistency could inflate the apparent stability. Without batch metadata in the provided file, the two cannot be disentangled. The modelling decision (do not use delta) is the same regardless of cause.

**Q: Why did you treat the two tasks as independent rather than building a joint model?**  
> Three reasons. First, the feature panels share only 4.5% of CpGs (90 of 2,000) — a joint feature space doesn't exist in the provided data. Second, the biological questions differ: Task 1 is early-stage conversion from CN, Task 2 is progression from established MCI. Third, the class distributions, sample sizes, and baseline characteristics differ. A joint multi-state model would need to handle these differences explicitly without obvious benefit at this sample size. Treating them independently is the cleaner, more defensible choice here.

**Q: The features were variance-ratio selected globally. How biased are your AUCs?**  
> The direction of bias is clear: all AUCs are inflated. The magnitude is unknown without a held-out unseen dataset where the feature selection was run from scratch using only training data. The bias is likely substantial — studies that have compared global vs fold-wise feature selection on similar high-D low-N problems have reported AUC drops of 0.05 to 0.15+. I would not cite the absolute numbers (e.g., 0.970) as generalisation performance estimates. The comparisons between models and modes are still valid since all models operate on the same biased feature set.

---

## Theme B: Temporal Representation

**Q: Why not just use a temporal transformer — that's what you initially planned?**  
> The initial plan assumed variable-length sequences of 2–4 timepoints and meaningful longitudinal dynamics. The data has exactly 2 timepoints per individual, and those two timepoints are nearly identical (r = 0.998). At T = 2, transformer attention is a weighted combination of two embeddings — no multi-step dependency to model. With r = 0.998, that weight would be nearly 0.5 regardless of training, adding nothing over averaging. The four explicit temporal modes are more informative: they directly answer the scientific question of which temporal representation carries signal, rather than letting it be implicit in a learned weight.

**Q: Could concat underfit because it doubles the feature space to 4,000 with no more samples?**  
> Yes, that's the concern going in, and it's why the logistic regression with concat uses L2 regularisation and the MLP uses dropout and weight decay. In practice, concat performs comparably to or slightly better than t0 alone (Task 1 logistic regression: concat PR-AUC 0.888 vs t0 0.879; Task 2 concat 0.870 vs t0 0.854). The additional features didn't hurt. But given the pre-selection leakage, it's hard to say whether the small gains are real or artefactual.

**Q: Did you try weighting t0 and t1 differently rather than just concatenating?**  
> No — that would require learning the optimal weight, which is effectively what an attention layer does. With T = 2 and r = 0.998, a learned weight will be approximately equal-weighting regardless of training dynamics. Explicit concatenation with regularisation achieves the same effect transparently. A small hyperparameter search on a t0/t1 linear combination was considered but not implemented — with 15 CV folds, any within-sample optimisation on the weighting coefficient would require a nested loop to avoid inflating results.

---

## Theme C: Model Choices

**Q: Why L2 regularisation for logistic regression instead of L1 (Lasso)?**  
> DNA methylation features are not independent — CpGs in the same genomic region are correlated through shared regulatory mechanisms. L1 regularisation arbitrarily zeroes one of a correlated pair while shrinking the other to zero, which is biologically arbitrary. L2 shrinks correlated features together, which is more appropriate when the discriminative signal is distributed across a genomic region rather than concentrated in a single site. L1 would also produce a sparse model that is harder to interpret biologically, since the zero weights don't mean those CpGs are uninformative — they just lost the L1 coin flip against a correlated neighbour. (report §3.2)

**Q: Why didn't you try ElasticNet, which combines L1 and L2?**  
> That's a reasonable extension. ElasticNet would allow grouped sparsity — keeping correlated CpGs together while zeroing out uninformative groups. The reason for not including it: it introduces a second hyperparameter (the mixing ratio) that would need tuning, and with 190–301 samples, any hyperparameter optimisation would need nested CV to avoid inflating results. L2 was chosen as the principled default for correlated features; ElasticNet would be the next step with nested CV in place.

**Q: Why did you include GBM if you expected it to underperform?**  
> A few reasons. First, expectation is not certainty — it's possible the data has nonlinear feature interactions that GBM captures and logistic regression doesn't. Including it makes the comparison empirical rather than assumed. Second, it's standard practice to include a nonlinear tabular baseline when you have linear models and neural networks. Third, the result is informative: GBM's substantial underperformance (0.725 vs 0.948 ROC-AUC on Task 1) is itself a finding that confirms the data regime — trees need the feature count to be far below the sample count to be competitive, and here it's the opposite.

**Q: The MLP architecture is quite small. Did you try larger networks?**  
> No. The deliberate choice was to keep the architecture constrained to the data regime. 190–301 individuals is not enough to train a larger network without memorising training folds. The bottleneck from 256 → 64 was chosen to force the network to compress region-level methylation patterns before the classification step. BatchNorm and Dropout 0.4 at both hidden layers provided the regularisation. Early stopping with patience=15 added a further guard against overfitting. A larger architecture in this regime typically produces higher training AUC and lower or more variable validation AUC — the complexity is not warranted.

**Q: BatchNorm after linear layers with batch_size=32 and 190 samples — does that have issues?**  
> Potential issue: with batch_size=32 and small folds (≈152 training individuals in each of 15 folds), you get 4–5 batches per epoch. BatchNorm statistics are computed per batch, and with 32 samples per batch the estimates are somewhat noisy. In practice this was stable — no NaN losses, training curves converged. InstanceNorm or GroupNorm would be safer alternatives with fewer batch statistics issues, and that's worth revisiting. The choice of BatchNorm was made as a standard default that typically works; it wasn't a deliberate comparative choice.

**Q: Why AdamW instead of Adam or SGD?**  
> AdamW separates weight decay from the gradient update, which means the L2 regularisation is applied correctly to the weights rather than being absorbed into the adaptive learning rate scaling. With Adam (not W), weight decay and L2 regularisation are not equivalent for networks with adaptive per-parameter learning rates. AdamW is the standard for transformer-like architectures and small MLPs with weight decay; the difference matters in the small-sample regime where regularisation is important.

**Q: Why CPU for the MLP? Isn't that slow?**  
> The MLP is small — 2000 → 256 → 64 → 1, trained on 150–240 samples per fold. Forward and backward passes take milliseconds on CPU. The overhead of moving data to GPU and back would dominate for this batch size. 15 folds × up to 200 epochs runs in under a minute per temporal mode on CPU. GPU training would be warranted for larger architectures or datasets.

**Q: Your early stopping monitors val_loss, but val_loss uses the same pos_weight as training. Does that affect the stopping criterion?**  
> Yes. The val_loss for early stopping is a pos_weight-scaled BCE, not an unweighted one, so stopping prefers epochs where fewer true converters are missed. This is arguably clinically appropriate — you stop the model when it's best at not missing converters — but it's an implicit effect rather than a deliberate choice. A cleaner alternative would be early-stopping directly on val PR-AUC or val sensitivity every epoch. The current implementation is not wrong, just undocumented in intent.

**Q: Your MLP uses pos_weight = n_neg/n_pos per fold while sklearn uses class_weight='balanced'. Are these equivalent?**  
> The gradient *ratio* (positive example gradient vs negative example gradient) is the same. sklearn's balanced weighting gives w₁/w₀ = n_neg/n_pos — identical to PyTorch's pos_weight. The absolute loss magnitudes differ (sklearn upweights both classes relative to unweighted; PyTorch leaves the negative class at 1 and scales only positives), but the decision boundary shift is equivalent. Per-fold computation in both cases is important: if pos_weight were computed once on the full dataset it would differ slightly from per-fold values when class balance varies across folds.

**Q: The brief explicitly mentions CNNs and RNNs. Why didn't you use either?**  
> **RNN:** A recurrent model (GRU, LSTM) requires a sequence of meaningful length where earlier steps inform later ones. With T=2 and within-individual r=0.998, there is no temporal dependency to model — the hidden state after the first step carries almost no additional information over the input itself. An RNN here is a one-step transformation with extra parameters.  
>   
> **1D CNN:** A convolutional network over features assumes spatial locality — nearby features share structure (e.g., pixels in an image, nucleotides in a sequence). The 2,000 CpGs here are ordered arbitrarily by the data provider (variance-ratio rank, not genomic position). A convolution kernel sliding over these features would aggregate random CpGs with no shared regulatory context. For CNNs on methylation to be meaningful, features need to be ordered by genomic coordinate so that kernels capture co-regulated CpG windows. Since genomic coordinates were not provided (only CpG IDs), a 1D CNN would have been biologically unmotivated. The biologically correct analogue — an attention-based model over genomic position embeddings — requires the full CpG space and chromosomal coordinates.

**Q: Walk me through the pipeline end-to-end.**  
> 1. **Load** — `src/data.py` (or `src/datasets/adni.py`) reads the HDF5 and returns X as (N, T, D) and y as (N,) per task.  
> 2. **Temporal mode** — `src/preprocessing.py` computes the requested representation (t0, t1, concat, or delta) from the (N, T, D) array, giving (N, D') where D' is 2,000 or 4,000.  
> 3. **CV split** — `src/splits.py` generates 15 stratified folds (5-fold × 3 repeats) from (N, D'). A leakage assertion checks no index overlaps.  
> 4. **Per-fold preprocessing** — StandardScaler fitted on training split, applied to validation split.  
> 5. **Train** — `src/train_sklearn.py` or `src/train_torch.py` trains the model on the fold's training data with class-weighted loss.  
> 6. **Evaluate** — model predicts probabilities on the validation fold; metrics are computed and stored.  
> 7. **OOF predictions** — each fold's validation predictions are concatenated into a full out-of-fold prediction file (`outputs/predictions/*.csv`).  
> 8. **Aggregate** — mean ± std across 15 folds is written to `outputs/metrics/*.json`.  
> 9. **Visualise** — `src/evaluate.py` loads all OOF CSVs and produces ROC curves, PR curves, and comparison bar charts.  

**Q: Why not try an ensemble of LogReg + MLP?**  
> Not implemented in this submission. Averaging predicted probabilities from LogReg (well-calibrated, stable) and MLP (higher sensitivity but more variable) is a natural extension that could improve both PR-AUC and sensitivity stability. With 15 folds of OOF predictions saved for each model, post-hoc ensembling could be done without re-running training — that's exactly why OOF predictions are saved. Worth doing as a first next step.

---

## Theme D: Validation & Methodology

**Q: Why 5-fold × 3 repeats rather than a single held-out test set?**  
> Task 1 has 43 converters. A standard 80/20 split would leave approximately 8–9 positives in the test set. AUC estimated from 8 examples has a standard error of roughly 0.15–0.20 — the number is almost meaningless. Repeated stratified CV gives 15 estimates of the same metric; the mean and standard deviation are interpretable and show whether results are stable or fold-specific. The cost is more compute; the benefit is substantially more reliable metrics. (report §3.3)

**Q: Why is individual-level CV equivalent to stratified group CV here?**  
> Each individual contributes exactly one row (one feature vector, one label). There's no risk of the same individual appearing in both train and validation splits because each individual appears only once. If individuals had contributed multiple rows — for example one row per timepoint, or repeated measurements — standard CV would place different rows from the same individual in different splits and leak individual-level signal. The data loader aggregates timepoints into a single feature vector per individual before splitting, so the split is clean. This is verified by an assertion in `src/splits.py`. (report §3.3)

**Q: Why didn't you use nested CV for hyperparameter tuning?**  
> Hyperparameter values were fixed based on domain knowledge of the data regime rather than searched. The choices — C=1.0 for logistic regression, architecture (256, 64) and dropout 0.4 for the MLP — were informed by general principles (strong regularisation in high-D low-N problems) rather than fitted to this specific dataset. That said: this is a form of implicit selection pressure. A reviewer is right to note that any hyperparameter decision informed by the dataset — even at the level of "256 units is appropriate for ~200 samples" — benefits from not being evaluated on the same data that informed it. Nested CV (inner folds for tuning, outer folds for evaluation) would give a less optimistic estimate. With 15 outer folds already, adding inner folds would produce 15 × 5 = 75 training runs per model and mode. Given the pre-selection leakage issue, the priority was to acknowledge both sources of bias and report honestly rather than fix one while the other remained.

**Q: Did you check that there's no data leakage in the cross-validation loop?**  
> Yes. `src/splits.py` contains an assertion that no index appears in both the training and validation sets for any fold. `src/preprocessing.py` fits the StandardScaler on the training fold only and applies it to the validation fold — the scaler never sees validation distribution statistics during fitting. The full preprocessing and training pipeline is inside the CV loop. (report §3.4)

**Q: Why class-weighted loss rather than oversampling (SMOTE) or undersampling?**  
> With 43 converters in Task 1, SMOTE generates synthetic minorities by interpolating between real ones. With so few positives, the synthetic samples are nearly identical to existing converters — they add artificial redundancy rather than independent signal, potentially giving overconfident minority-class decision boundaries. Random undersampling discards majority-class data from an already small dataset. Loss weighting achieves the same calibration effect on the decision boundary — upweighting the minority class during training — without modifying the training data. (report §3.5)

---

## Theme E: Results

**Q: LogReg Task 1 has ROC-AUC 0.948 but sensitivity only 0.413. The MLP has ROC-AUC 0.970 but sensitivity 0.803. How can two models with similar ROC-AUC have such different sensitivity?**  
> ROC-AUC is a threshold-free ranking metric — it measures the probability that the model ranks a random positive above a random negative across all possible thresholds. Sensitivity is the true positive rate at a specific threshold (here, 0.5 by default). A model with good ranking but poor calibration will have high ROC-AUC and low sensitivity at 0.5 if its predicted probabilities for positives are systematically low (e.g., predicted 0.3 for true converters — correctly ranked above negatives at 0.1, but below the 0.5 threshold). The mechanism for LogReg: L2 regularisation compresses all coefficients toward zero, so logits (the linear combination of features) cluster near zero, which maps to predicted probabilities clustering near 0.5. Relatively few predictions cross 0.5 firmly for the positive class, so sensitivity at 0.5 is low even when ranking (AUC) is good. The MLP's nonlinear activations produce more bimodal predicted probability distributions — outputs cluster near 0 or 1 rather than near 0.5 — so more true positives exceed the 0.5 threshold. `class_weight='balanced'` shifts the decision boundary but doesn't resolve the logit compression problem.  
>   
> The practical implication: if clinical deployment used a threshold other than 0.5 (e.g., Youden's J to balance sensitivity and specificity), the LogReg sensitivity gap might close. The ROC-AUC comparison is still the most model-selection-agnostic metric for this reason.

**Q: Why is 0.5 used as the decision threshold?**  
> The 0.5 threshold was used as the default for reporting balanced accuracy, sensitivity, and F1. This is a deliberate choice not to optimise the threshold on the same data used for evaluation — threshold optimisation (e.g., Youden's J, F1-maximising threshold) on the validation fold would constitute another source of selection pressure and inflate those metrics further. With OOF predictions saved for all 15 folds, post-hoc threshold calibration on a held-out subset (or via nested CV) would be the principled way to select and evaluate a deployment threshold. That analysis was not completed within the submission scope.

**Q: Were the calibrated predicted probabilities checked (reliability diagrams)?**  
> Calibration was not formally assessed. Class-weighted training often shifts predicted probabilities away from the true prevalence — the model sees an artificially balanced training distribution, so probabilities near 0.5 map to a different actual frequency in the original imbalanced distribution. For clinical deployment, probability calibration (e.g., Platt scaling or isotonic regression post-hoc) would be important. The current probabilities should not be interpreted as direct estimates of conversion probability without recalibration.

**Q: Is the difference between MLP (0.970) and LogReg (0.948) statistically significant?**  
> With 15 CV folds, a paired t-test or Wilcoxon signed-rank test on per-fold ROC-AUC would be the appropriate check. This was not formally run. The MLP mean exceeds LogReg by 0.022 ± overlapping SDs (MLP σ = 0.025, LogReg σ = 0.036 on t0). The difference is consistent in direction across most folds but may not reach significance in a paired test. The sensitivity difference (0.803 vs 0.413) is more striking and likely more robust, though sensitivity at a fixed threshold is noisier than AUC across folds. The honest answer: the ranking difference is clear; statistical significance of the AUC gap was not tested.

**Q: Why does GBM perform so poorly?**  
> HistGradientBoosting is building a tree ensemble over 2,000+ features with 190–240 training samples (after holding out a fold). Tree splits are evaluated on each feature sequentially, and with far more features than samples, the splits overfit to noise. The model cannot generalise the feature-interaction structure from so few examples. Logistic regression with L2 regularisation naturally handles this regime — it shares information across all features simultaneously through the regularisation term rather than splitting greedily on individual ones. GBM performance typically improves when the sample-to-feature ratio is higher (e.g., >10:1). Here the ratio is <0.1:1.

**Q: Why does adding t1 help on Task 2 but not much on Task 1?**  
> Task 2 (MCI → Dementia) has 301 individuals and 36.5% positives — a richer dataset with less extreme imbalance. The concat model can afford the larger feature space (4,000 features) better with more samples. Task 1 has 190 individuals and 22.6% positives — the added features from t1 may not contribute independent signal beyond what t0 already captures, and the higher dimensionality adds noise at this sample size. Task 2 also showed a consistent LogReg advantage for concat over t0 (0.897 vs 0.882 ROC-AUC), suggesting incremental longitudinal information is real there. On Task 1 the difference was negligible (0.945 concat vs 0.948 t0).

**Q: What does the near-chance delta performance tell us biologically?**  
> It tells us that the methylation change between visits — whatever its cause — does not discriminate converters from non-converters. Two interpretations: (1) The conversion-relevant methylation differences are established before the first visit and remain stable; they are state markers, not dynamic markers of ongoing progression. (2) The observation window (time between t0 and t1) is too short to capture epigenetic change that precedes clinical progression. Both are consistent with the data. A stronger claim would require a permutation test — shuffle the delta features and confirm performance drops to chance — which was not run.

**Q: Could you have used feature importance from LogReg to select features for the MLP?**  
> In principle yes. Fitting LogReg to extract the top-K features by absolute coefficient and then training the MLP on that subset might reduce noise for the MLP. But this introduces a new form of leakage: the LogReg is fit on training data and its top features are selected before the MLP is trained, which means the feature selection is conditioned on the same training fold and may overfit. To do it correctly, LogReg feature selection should be in the inner loop of nested CV, evaluated on outer folds. Given the pre-selection leakage already present, adding another selection step without nested CV would compound the bias. The cleaner baseline is to train models on the same feature set.

**Q: Which CpGs have the highest LogReg coefficients — any biological signal?**  
> Logistic regression coefficients were computed per fold; the most stable coefficients (consistent sign and magnitude across 15 folds) would be the most reliable feature importance estimates. This analysis was not completed within the submission scope but is straightforward: compute mean coefficient and its coefficient of variation across folds, filter to low-CV sites, and check against known AD-related gene databases (e.g., AlzGene, EWAS Atlas). Given the pre-selection leakage, any biological story would need external validation before being taken seriously.

---

## Theme F: Limitations & Future Work

**Q: How much do you trust the absolute AUC numbers?**  
> The ranking-based metrics (ROC-AUC, PR-AUC) are likely inflated by 0.05–0.15 or more due to global pre-selection leakage. They should not be cited as estimates of performance on a new, independently collected cohort. The relative comparisons between models (MLP > LogReg > GBM) and between temporal modes (t0 ≈ t1 > concat ≥ t0 >> delta) are more trustworthy because all models operate on the same biased feature set — the bias is constant across comparisons. Performance estimates from fold-wise feature reselection on the full CpG matrix would be the right benchmark.

**Q: What is the clinical use case — and would these numbers be good enough for clinical deployment?**  
> The framing is early detection / risk stratification: identifying individuals likely to progress to enable early intervention or trial enrolment. For clinical deployment, the requirements would be much stricter than reported here: validated on an external cohort, calibrated predicted probabilities (not raw scores), sensitivity/specificity at a clinically meaningful threshold, and comparison to existing clinical predictors (cognitive scores, APOE, imaging). The current work is a methodological demonstration using the provided data, not a validated clinical tool.

**Q: You mention APOE ε4 as a confounder. How serious is it?**  
> APOE ε4 is the strongest known genetic risk factor for late-onset AD. Carriers have a substantially higher conversion rate. APOE ε4 carrier status is also correlated with specific methylation patterns. Without including APOE genotype as a covariate (or as a stratification variable), the model may learn methylation signatures that track APOE status rather than conversion itself. This would inflate ADNI performance while the model fails to generalise to populations with a different APOE ε4 prevalence. It is the most important missing covariate.

**Q: How would you validate on an external cohort?**  
> The pipeline produces saved model weights and a StandardScaler per fold (the scaler parameters are not currently saved — that's a gap). For external validation: load the ADNI-trained model, apply the ADNI-fitted scaler to the external cohort features (assuming the same CpG panel is available), and compute predicted probabilities. The external cohort would need the same 2,000 CpG features (cpg_ids_cn or cpg_ids_mci depending on task), measured from peripheral blood, with the same Illumina platform and preprocessing pipeline. ROSMAP (Rush University) and AIBL (Australian Imaging, Biomarker & Lifestyle study) are natural candidates, though assay platform matching would require harmonisation.

**Q: What would survival analysis add over binary classification?**  
> The binary label discards when conversion happens, only recording that it does within the observation window. A Cox proportional hazards model or a discrete-time survival model (e.g., using `pycox`) would model the time-to-conversion as the outcome, making full use of censored data (individuals who have not yet converted within the follow-up window). This is clinically richer — a model that says "this person has a 30% chance of converting within 2 years" is more useful than "this person is likely to convert." It also avoids the arbitrary binary cutoff imposed by the observation window length.

**Q: You mention the dataset interface already returns (N, T, D). What would you build next on that?**  
> The `BaseDataset` abstract class (`src/datasets/base.py`) returns X as (N, T, D) — individuals × timepoints × features. A GRU or small transformer encoder that takes (T, D) per individual and outputs a fixed representation for classification is the natural next step when T > 2 and r < 0.99. The training scripts already accept X in this shape and flatten to a temporal mode before model input; adding a path that passes the full (N, T, D) tensor to a sequence model is a config change, not a data pipeline change.

---

## Theme G: Code & Reproducibility

**Q: How do you ensure reproducibility across runs?**  
> Seeds are set explicitly: `torch.manual_seed(seed)` and `np.random.seed(seed)` at the start of each fold in `src/models/mlp.py:43`. The CV split generator is seeded via `random_state` in `src/splits.py`. OOF predictions are saved to CSV so evaluation plots can be regenerated without re-running training. The environment is fully containerised (Dockerfile + docker-compose) with a pinned `requirements.txt`. Results may differ across hardware due to floating-point non-determinism in parallel BLAS operations; setting `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms(True)` would close this gap on GPU.

**Q: Why did you build a dataset registry and model registry in Phase 3?**  
> Before the refactor, adding a new dataset required editing `src/data.py` — a source file that is also imported by tests and the inspect script. The registry pattern (`@register_dataset`, `@register_model`) means a new dataset or model is a new file + config change, not a source edit. This matters for scale: if the pipeline is extended to ROSMAP or AIBL, the ADNI-specific code stays unchanged. It also means tests for the existing pipeline don't need updating when new datasets are added. (report §6, Phase 3)

**Q: What is the `src/data.py` shim?**  
> After the Phase 3 refactor, `src/datasets/adni.py` contains all the ADNI-specific logic. `src/data.py` was retained as a thin wrapper that calls `AdniDataset` for backwards compatibility — existing tests and the `inspect_hdf5.py` script import from `src/data.py` and continue to work without modification. It's a deliberate backwards-compat shim that avoids breaking the test suite mid-project.

**Q: What does the W&B integration look like — can it be turned off?**  
> `src/wandb_utils.py` is a thin wrapper around `wandb`. Training scripts accept a `--wandb` flag; when omitted, all logging calls are no-ops. The training loop is completely decoupled from W&B — there are no direct `wandb.*` calls in `src/train_sklearn.py` or `src/train_torch.py`. This was a deliberate choice: W&B is opt-in and the pipeline runs identically without it. (report §6, Phase 2)

**Q: The HDF5 stores data as (features, timepoints, individuals). Walk me through the axis transposition.**  
> `adni.py:46` calls `f[group][:].transpose(2, 1, 0)` on each group. The raw HDF5 array has shape `(D, T, N)` — features first, then timepoints, then individuals. `.transpose(2, 1, 0)` reorders axes to `(N, T, D)` — individuals × timepoints × features — which is the convention every downstream component expects. The test `test_load_shape` asserts `X.shape == (N, 2, 2000)` and would catch any transposition regression immediately.

**Q: Walk me through what your tests cover and what they don't.**  
> The 15 tests across two files cover: data loading shape, beta-value range, NaN absence, label counts, invalid-task error handling, task_summary consistency, no-individual-leakage per fold, full-coverage CV (every individual in exactly one val fold), and stratification ratio preservation. What's not covered: `src/metrics.py` (compute_metrics, aggregate_fold_metrics), `src/preprocessing.py` (prepare_features, fit_transform), model training (`mlp.py`, `sklearn_models.py`), `evaluate.py`, and the OOF CSV write path. A model-level test would assert that a trained fold produces probabilities in [0, 1] and that sensitivity is above the positive rate baseline. Adding those tests would be straightforward with a synthetic HDF5 fixture.

**Q: Your CI smoke test runs without the real ADNI HDF5. What does and doesn't it validate?**  
> The smoke pipeline (scripts/e2e_smoke.sh) uses synthetic lightweight data so the private ADNI file is never committed or required by CI. It validates that the full code path (load → temporal mode → CV → train sklearn + MLP → evaluate → write outputs) executes without errors, and that output files appear at the expected locations. It doesn't validate that numerical results on real ADNI data haven't regressed, or that model accuracy is above any threshold. Numerical regression testing would require either a committed reference output file (fragile to floating-point changes) or a small public methylation dataset to substitute for ADNI in CI.

**Q: The StandardScaler is fitted per fold but the fitted scaler is discarded. How would you run inference on a new patient?**  
> This is a real gap: `train_torch.py:46` calls `fit_transform(...)` and discards the returned scaler via `_`. For inference, you need the fitted scaler from the same fold used to train the model. The fix is a one-liner — `joblib.dump(scaler, scaler_path)` inside the training loop — and the scaler should be saved alongside the model weights. For a production endpoint the most practical approach is to refit the scaler on the full training dataset (all 15 folds combined) after CV, save it once, and use that for inference. The model weights aren't saved either; adding `torch.save(model.state_dict(), model_path)` completes the inference artefacts.

**Q: The presentation notes mention fixing the model registry after submission. What changed and why?**  
> Before the fix (`git commit 7e6f874`), the model registry and dataset registry used different patterns — the dataset registry used a class decorator (`@register_dataset`) while the model registry used function-level registration that didn't match. The fix aligned them so both follow the same decorator pattern, making the extension story consistent. This was completed after submission as a polish step — the core training results and all reported metrics are unchanged. It's worth being transparent: the submission included the functional pipeline, and the registry alignment was a code-quality improvement identified during review.

---

## Theme H: AI Tool Usage

**Q: The brief asks you to describe how you used AI tools. Walk us through that.**  
> Claude was used across three areas: (1) **Data exploration** — understanding the HDF5 structure and confirming that labels were pre-defined rather than requiring trajectory construction; (2) **Architectural reasoning** — the switch from a temporal transformer to explicit temporal modes was developed in dialogue with Claude after examining the within-individual correlation, which confirmed that T=2 with r=0.998 makes attention a near-trivial operation; (3) **Code generation** — training loops, CV scaffolding, the dataset/model registry pattern, evaluation plots, and the written report were generated with Claude and then reviewed and tested manually. The AI usage is documented in the README's AI tool usage section.

**Q: How did you verify that AI-generated code was correct?**  
> Two mechanisms. First, 15 unit tests cover the critical pipeline components — any generated code that broke data loading, split leakage, or stratification invariants was caught immediately. Second, the results were compared against analytical predictions made before modelling: delta should be uninformative, GBM should underperform in high-D low-N, logistic regression should be competitive. When all three predictions held empirically, it gave confidence the pipeline was correct. Any code that produced surprising results triggered investigation rather than acceptance.

**Q: Did AI tools make the architectural decisions, or just implement ideas you had already formed?**  
> Both. The decision to abandon the temporal transformer was mine after inspecting the correlation structure — the reasoning was clear before any Claude interaction. But the specific form of the temporal ablation (four named modes evaluated comparatively rather than a learned weighting), the OOF prediction saving strategy, and the dataset/model registry pattern were developed in dialogue with Claude. Claude's suggestions were filtered against domain knowledge of methylation studies and against empirical results on the actual data — suggestions that contradicted either were rejected or modified.

---

## Theme I: Deployment & Clinical Translation

**Q: If this were deployed as a clinical decision support tool, what are the minimal productionization steps?**  
> Five steps in order of priority: (1) Save the fitted scaler and model weights — currently neither is persisted; (2) Calibrate predicted probabilities — class-weighted training shifts outputs away from true conversion rates, so Platt scaling or isotonic regression on a held-out set is needed before probabilities can be interpreted as risk estimates; (3) Select an operating threshold explicitly using a clinical cost criterion (Youden's J, or a cost-weighted threshold from the ROC curve) rather than the default 0.5; (4) Validate on an external cohort (ROSMAP, AIBL) before any clinical use; (5) Wrap in an API endpoint that takes CpG beta values, applies the saved scaler, and returns a calibrated risk score. Steps 1–3 are post-hoc additions to the existing pipeline — none require retraining.

**Q: What would you monitor once this model is deployed?**  
> Three layers: (1) **Data drift** — monitor the distribution of incoming beta values against the ADNI training distribution; if mean or variance per CpG shifts significantly, the standardisation will be wrong and outputs unreliable; (2) **Prediction drift** — monitor the distribution of predicted probabilities over time; a systematic shift toward extreme values can indicate input distribution change; (3) **Outcome monitoring** — if follow-up diagnoses are eventually available, compare predicted conversion probabilities against actual outcomes to track calibration decay. In a methylation context, array platform changes (e.g., EPIC v1 to v2) are a known distribution-shift trigger that would require recalibration or retraining.

---

_Document covers: Problem framing · EDA & decisions · Model choices · Validation methodology · Results interpretation · Limitations · Reproducibility · AI tool usage · Deployment_

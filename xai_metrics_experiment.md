# XAI Metrics Experiment Protocol

## Goal
Define a reproducible evaluation protocol for SMX as an XAI method on spectral data, with direct comparison against SHAP, VIP, and a simple spectral baseline (permutation importance by zone).
The protocol combines objective and domain-grounded evaluation dimensions [R1].

## Scope
- Task type: binary classification (main metric: F1 and Accuracy).
- Explanation granularity:
  - Zone-level ranking.
- Methods:
  - `SMX`
  - `SHAP`
  - `VIP`
  - `Permutation-by-zone` (simple spectral baseline)

## Common Protocol (for all experiments)

### Data splitting
- Use `10` split seeds (`split_seed`).
- For each `split_seed`:
  - Shuffle data and run repeated stratified CV.
  - Recommended: `5x2` repeated stratified K-fold (5 folds, 2 repeats).
- Keep train/test splits identical across explainers within each seed/fold.

### Seed control
- `split_seed`: controls only dataset shuffling and train/test fold assignment.
- `model_seed`: controls only model creation/training randomness (e.g., MLP initialization, stochastic optimizer order).
- Use both and log both in outputs.
- Recommended protocol:
  - Stability wrt data split: vary `split_seed`, keep `model_seed` fixed.
  - Stability wrt model stochasticity: vary `model_seed`, keep `split_seed` fixed.
  - Full variability analysis: vary both.

### Models
- Compare at least: `PLS-DA`, `SVM`, `MLP`.
- Fit model on train split only.
- Compute test predictions and test performance (F1, Accuracy).

### Explanation extraction
- Always compute explanations on test samples (or train+test if explicitly needed, but report which one).
- For each method, export:
  - Zone ranking list.

### Statistical reporting
- Report mean and `95% CI` over all seed-fold runs.
- CI method: bootstrap CI on run-level scores (recommended 2000 resamples).
- Paired test (method vs method on same runs):
  - If paired differences are approximately normal: paired t-test.
  - Otherwise: Wilcoxon signed-rank test.
- Report effect size where possible.
  Statistical references: [R12], [R13], [R14].

---

## 1) Faithfulness

Faithfulness here means: if top-ranked explanation units are removed, model performance should degrade more.
This is aligned with perturbation-based faithfulness/comprehensiveness evaluation [R5].

### 1.1 Sub-experiment A: Rank zones and remove top-k zones

#### Ranking definitions
- SMX: use zone importance from SMX relevance (e.g., LRC-based natural ranking).
- SHAP: zone score = mean absolute SHAP value aggregated per zone [R6].
- VIP: zone score = mean VIP per zone [R7].
- Permutation baseline: zone score = permutation importance per zone.

#### Removal operator
- For each method and each `k` in `{1, 2, 3, 5, 10, 20, ...}` (bounded by number of zones):
  - Remove (mask) top-k zones in test data.
  - Apply the same masking rule for all methods.
  - Re-evaluate model with no retraining.

#### Metrics
- `F1_drop(k) = F1_original - F1_masked(k)`
- `ACC_drop(k) = ACC_original - ACC_masked(k)`
- Optional summary: area under degradation curve (larger means more faithful ranking).


---

## 2) Stability

Measure ranking consistency using RBO under controlled seed variation.
This uses RBO for top-weighted rank similarity [R9].

### Definition
- For each method, build ranking lists from repeated runs with explicit seed mode:
  - `split-seed stability`: rankings across different `split_seed` with fixed `model_seed`.
  - `model-seed stability`: rankings across different `model_seed` with fixed `split_seed`.
- Compute pairwise `RBO` among runs of the same method.
- Stability score = mean pairwise RBO.

### Report
- `mean_RBO_zones_split_seed(method)`
- `mean_RBO_zones_model_seed(method)`
- Include 95% CI.

Recommended RBO parameter:
- `p = 0.7` (top-weighted; adjust if stronger top emphasis is desired).

---

## 3) Model Dependence

Assess how explanation rankings depend on model family.
Motivation: explanations can vary across different model classes fitted on the same data (Rashomon/model dependence) [R10], [R11].

### Procedure
- Reuse the same seed/fold protocol from Stability.
- For each method (SMX, SHAP, VIP, permutation):
  - Compare rankings generated from different algorithms (`PLS-DA`, `SVM`, `MLP`) on the same data splits.
  - Use fixed `split_seed` and fixed `model_seed` when comparing algorithms in the same run.
  - Compute RBO between algorithm pairs:
    - `PLS-DA vs SVM`
    - `PLS-DA vs MLP`
    - `SVM vs MLP`

### Report
- `model_dependence_RBO` per method and per pair.
- Lower RBO => stronger model dependence.

---

## 4) Domain-Grounded Evaluation

Human expert validation of top zones.
This corresponds to application-grounded evaluation [R1], [R12].

### Input to expert
- For each method and run, show top-N zones with:
  - mapped spectral range,
  - associated elemental annotation (if available).

### Expert task
- For each item: mark `physically_plausible` as `yes/no`.

### Metrics
- Expert agreement rate:
  - `agreement_rate = (# yes) / (total reviewed)`
---

## 5) Baselines and Comparison Matrix

Methods to include:
- SMX
- SHAP
- VIP
- Permutation-by-zone baseline

Compare methods on:
- Faithfulness (zone only)
- Stability (RBO across seeds)
- Model dependence (RBO across algorithms)
- Domain-grounded agreement
- Comprehensibility (section 6)

All comparisons must use:
- identical seed/fold splits,
- identical metric computation.

---

## 6) Comprehensibility

Focus metrics:

### 6.1 Number of top zones to explain 80% relevance
- Sort zones by relevance descending.
- Compute cumulative relevance ratio.
- `N80_zones = minimum n such that cumulative_relevance(n) >= 0.80`.
- Lower is more comprehensible (more compact explanation).
Related interpretability notions: simulatability and compactness [R15], [R16].

### 6.2 Percentage of top-ranked zones matching expected elemental signatures
- Define expected elemental signature zones from domain knowledge.
- For top-K zones:
  - `elemental_match_pct = (# top-K zones in expected list) / K`.
- Report at K = 2, K = 5 and K = 10.

---

## Minimal Output Artifacts

Recommended CSV outputs:
- `faithfulness_zone_curve.csv`
  - columns: `split_seed, model_seed, fold, model, method, k, f1_original, f1_masked, f1_drop, acc_original, acc_masked, acc_drop`
- `stability_rbo.csv`
  - columns: `method, granularity(zone), stability_mode(split_seed|model_seed), run_i, run_j, split_seed_i, model_seed_i, split_seed_j, model_seed_j, rbo`
- `model_dependence_rbo.csv`
  - columns: `method, granularity(zone), model_a, model_b, split_seed, model_seed, fold, rbo`
- `domain_grounded_expert_review.csv`
  - columns: `split_seed, model_seed, method, model, item_type(zone), item_id, rank, plausible_yes_no, expert_id`
- `comprehensibility_metrics.csv`
  - columns: `split_seed, model_seed, fold, model, method, n80_zones, elemental_match_k5, elemental_match_k10`

Recommended summary file:
- `xai_metrics_summary.csv`
  - mean, 95% CI, paired-test p-values for each metric and method comparison.

---

## References

- [R1] Vilone, G., & Longo, L. (2021). *Notions of explainability and evaluation approaches for explainable artificial intelligence*. Information Fusion, 76, 89-106.
- [R5] Yeh, C.-K., Kim, J. S., Yen, I. E.-H., & Ravikumar, P. (2019). *On the (In)fidelity and Sensitivity of Explanations*. NeurIPS.
- [R6] Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- [R7] Chong, I.-G., & Jun, C.-H. (2005). *Performance of some variable selection methods when multicollinearity is present*. Chemometrics and Intelligent Laboratory Systems, 78(1-2), 103-112.
- [R9] Webber, W., Moffat, A., & Zobel, J. (2010). *A Similarity Measure for Indefinite Rankings*. ACM TOIS, 28(4), 20.
- [R10] Breiman, L. (2001). *Statistical Modeling: The Two Cultures*. Statistical Science, 16(3), 199-231.
- [R11] Rudin, C. (2019). *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*. Nature Machine Intelligence, 1, 206-215.
- [R12] Doshi-Velez, F., & Kim, B. (2017). *Towards A Rigorous Science of Interpretable Machine Learning*. arXiv:1702.08608.
- [R13] Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- [R14] Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation, 10(7), 1895-1923.
- [R15] Lipton, Z. C. (2016). *The Mythos of Model Interpretability*. arXiv:1606.03490.
- [R16] Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).

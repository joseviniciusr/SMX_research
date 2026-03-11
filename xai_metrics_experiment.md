# XAI Metrics Experiment Protocol

## Goal
Define a reproducible evaluation protocol for SMX as an XAI method on spectral data, with direct comparison against SHAP, VIP, and a simple spectral baseline (permutation importance by zone).
The protocol combines objective and domain-grounded evaluation dimensions [R1].

## Scope
- Task type: binary classification (main metric: F1 and Accuracy).
- Explanation granularity:
  - Zone-level ranking.
  - Predicate-boundary ranking (SMX predicates).
- Methods:
  - `SMX`
  - `SHAP`
  - `VIP`
  - `Permutation-by-zone` (simple spectral baseline)

## Common Protocol (for all experiments)

### Data splitting
- Use `10` random seeds.
- For each seed:
  - Shuffle data and run repeated stratified CV.
  - Recommended: `5x2` repeated stratified K-fold (5 folds, 2 repeats).
- Keep train/test splits identical across explainers within each seed/fold.

### Models
- Compare at least: `PLS-DA`, `SVM`, `MLP`.
- Fit model on train split only.
- Compute test predictions and test performance (F1, Accuracy).

### Explanation extraction
- Always compute explanations on test samples (or train+test if explicitly needed, but report which one).
- For each method, export:
  - Zone ranking list.
  - Predicate ranking list (when applicable or derived as defined below).

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

### 1.2 Sub-experiment B: Predicate boundaries and remove top-k predicates

Use the same principle on predicate boundaries (threshold predicates).
This sub-experiment is evaluated for `SMX` only.

#### Predicate set
- Use a common predicate dictionary (SMX-generated boundaries from quantiles), shared by all methods for fairness.

#### Predicate scoring
- SMX: use predicate relevance directly.

#### Predicate removal operator
- For top-k predicates:
  - Neutralize predicate condition in test set (set involved zone score to boundary value or local median so condition impact is removed).
  - Recompute predictions with same trained model.

#### Metrics
- `F1_drop_pred(k)`
- `ACC_drop_pred(k)`


---

## 2) Stability

Measure ranking consistency across random seeds (10 seeds), using RBO.
This uses RBO for top-weighted rank similarity [R9].

### Definition
- For each method, build ranking lists from different seed runs.
- Compute pairwise `RBO` among runs of the same method.
- Stability score = mean pairwise RBO.

### Report
- `mean_RBO_zones(method)`
- `mean_RBO_predicates(method)`
- Include 95% CI.

Recommended RBO parameter:
- `p = 0.9` (top-weighted; adjust if stronger top emphasis is desired).

---

## 3) Model Dependence

Assess how explanation rankings depend on model family.
Motivation: explanations can vary across different model classes fitted on the same data (Rashomon/model dependence) [R10], [R11].

### Procedure
- Reuse the same seed/fold protocol from Stability.
- For each method (SMX, SHAP, VIP, permutation):
  - Compare rankings generated from different algorithms (`PLS-DA`, `SVM`, `MLP`) on the same data splits.
  - Compute RBO between algorithm pairs:
    - `PLS-DA vs SVM`
    - `PLS-DA vs MLP`
    - `SVM vs MLP`

### Report
- `model_dependence_RBO` per method and per pair.
- Lower RBO => stronger model dependence.

---

## 4) Domain-Grounded Evaluation

Human expert validation of top predicates/zones.
This corresponds to application-grounded evaluation [R1], [R12].

### Input to expert
- For each method and run, show top-N predicates (and top-N zones), with:
  - Boundary interval / threshold,
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
- Faithfulness (zone for all methods; predicate for SMX only)
- Stability (RBO across seeds)
- Model dependence (RBO across algorithms)
- Domain-grounded agreement
- Comprehensibility (section 6)

All comparisons must use:
- identical seed/fold splits,
- identical masking/neutralization operators,
- identical metric computation.

---

## 6) Comprehensibility

Focus metrics:

### 6.1 Number of top predicates to explain 80% relevance
- Sort predicates by relevance descending.
- Compute cumulative relevance ratio.
- `N80_predicates = minimum n such that cumulative_relevance(n) >= 0.80`.
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
  - columns: `seed, fold, model, method, k, f1_original, f1_masked, f1_drop, acc_original, acc_masked, acc_drop`
- `faithfulness_predicate_curve.csv`
  - same schema with predicate fields
- `stability_rbo.csv`
  - columns: `method, granularity(zone|predicate), run_i, run_j, rbo`
- `model_dependence_rbo.csv`
  - columns: `method, granularity, model_a, model_b, seed, fold, rbo`
- `domain_grounded_expert_review.csv`
  - columns: `method, model, item_type(zone|predicate), item_id, rank, plausible_yes_no, expert_id`
- `comprehensibility_metrics.csv`
  - columns: `seed, fold, model, method, n80_predicates, elemental_match_k5, elemental_match_k10`

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

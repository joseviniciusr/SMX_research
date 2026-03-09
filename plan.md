# SMX Refactoring Plan

## 1. Current State Analysis

### Repository overview
SMX is a Python library it builds interpretable explanations for classification models trained on spectral data (XRF, Vis-NIR, etc.).
The core pipeline is:

1. **Load data** (CSV + JSON config) → **preprocess** (Poisson, MC, Savgol, Pareto, auto-scaling, MSC)
2. **Train model** (PLS-DA, SVM, MLP) → extract continuous predictions
3. **Extract spectral zones** → aggregate via PCA (PC1 scores)
4. **Generate predicates** from quantile thresholds on zone scores
5. **Bagging + metric computation** (covariance / perturbation) per predicate
6. **Build directed graph** of predicates (weighted by metric × explained variance)
7. **Compute LRC** (Local Reaching Centrality) → rank spectral zones by importance
8. **Map thresholds back** to natural spectral space for interpretation

### Current file map

| File | Lines | Role |
|------|------:|------|
| `smx/config.py` | 33 | Dataset config loading (JSON) |
| `smx/preprocessings.py` | ~120 | Loose functions: poisson, pareto, mc, auto_scaling, msc |
| `smx/modeling.py` | 753 | `pls_optimized`, `svm_optimized`, `mlp_optimized` + VIP/explained-variance helpers |
| `smx/explaining.py` | 2 514 | The heart: zone extraction, PCA aggregation, predicates, bagging, metrics (covariance, perturbation), graph building, LRC, permutation importance, threshold mapping, plotting |
| `smx/debugging.py` | ~130 | VIP/reg-coef/SHAP/SVM-pvector per zone, RBO comparison, export performance metrics |
| `smx/synthetic.py` | ~160 | Synthetic Gaussian-peak spectral data generator |
| `experiments/run_experiment.py` | 669 | CLI orchestrator that wires everything together |

### Key problems

1. **No package structure** — no `__init__.py`, no `setup.py`/`pyproject.toml`. The `smx/` directory is imported via `sys.path` hacks.
2. **Everything is functions** — no classes, no state encapsulation. The pipeline passes around raw dicts and tuples with implicit positional semantics (e.g. `result[5].iloc[:, -1]`).
3. **`explaining.py` is a 2 500-line monolith** — mixes zone extraction, PCA aggregation, predicate generation, bagging, two metric engines, graph construction, LRC computation, threshold mapping, and plotting.
4. **`modeling.py` has massive copy-paste** — the regression/classification branches of PLS, SVM, and MLP repeat nearly identical metric-computation blocks.
5. **Imports inside functions** — `numpy`, `pandas`, `sklearn` are imported inside almost every function body instead of at module level.
6. **Inconsistent return types** — `pls_optimized` returns 5 items for regression, 7 for classification. `svm_optimized` returns 4 or 8. Callers must know the tuple layout.
7. **Debugging/research utilities mixed in** — SHAP, RBO, VIP-per-zone, permutation importance live alongside the core algorithm, but they are only needed by maintainers.
8. **No tests, no docstring standard, no type hints on most functions.**
9. **Hardcoded paths** — `CONFIGS_DIR` points to a sibling `real_datasets/xrf/` folder relative to the package source.

---

## 2. Target Architecture

```
smx/
├── __init__.py                  # Public API surface
├── _version.py                  # Single-source version string
│
*├── preprocessing/
│   ├── __init__.py              # re-export public names
│   ├── scalers.py               # PoissonScaler, ParetoScaler, AutoScaler, MeanCenterer
│   ├── spectral.py              # SavGolFilter, MSCCorrector
│   └── pipeline.py              # PreprocessingPipeline (chains steps, fit/transform)
│
*├── models/
│   ├── __init__.py
│   ├── base.py                  # BaseSpectralModel (ABC) – unified return type
│   ├── pls.py                   # PLSModel (wraps PLSRegression for reg + DA)
│   ├── svm.py                   # SVMModel (wraps SVR/SVC)
│   └── mlp.py                   # MLPModel (wraps MLPRegressor/MLPClassifier)
│
├── zones/
│   ├── __init__.py
│   ├── extraction.py            # extract_spectral_zones()
│   └── aggregation.py           # ZoneAggregator: sum/mean/max/PCA, etc.
│
├── predicates/
│   ├── __init__.py
│   ├── generation.py            # PredicateGenerator (quantile-based)
│   ├── bagging.py               # PredicateBagger
│   └── metrics.py               # CovarianceMetric, PerturbationMetric (strategy)
│
├── graph/
│   ├── __init__.py
│   ├── builder.py               # PredicateGraphBuilder
│   ├── centrality.py            # LRC computation + seed aggregation
│   └── interpretation.py        # map_thresholds_to_natural, reconstruct_threshold_to_spectrum
│
*├── synthetic.py                 # SyntheticSpectraGenerator (class)
*├── config.py                    # DatasetConfig dataclass + load/list helpers
│
*├── _contrib/                    # Maintainer-only / research utilities
│   ├── __init__.py
│   ├── importance.py            # VIP-per-zone, reg-coef-per-zone, SVM-pvector-per-zone
│   ├── shap_utils.py            # SHAP zone mapping
│   ├── rbo.py                   # RBO rank comparison
│   ├── permutation.py           # permutation_importance_per_zone
│   └── export.py                # export_performance_metrics
│
└── plotting/
    ├── __init__.py
    └── threshold.py             # plot_threshold_spectrum (plotly)
```

Plus, at the repo root:
```
pyproject.toml          # PEP 621 metadata, build system, optional-deps
README.md               # Quick-start, installation, example
CHANGELOG.md
tests/                  # pytest suite
examples/               # Cleaned-up version of experiments/run_experiment.py
real_datasets/          # (stays as-is, referenced by config path or user arg)
experiments/            # (kept for internal research, .gitignore'd or separate branch)
```

---

## 3. Detailed Refactoring Steps

### Phase 1 — Packaging & project skeleton

| # | Task | Details |
|---|------|---------|
| 1.1 | Create `pyproject.toml` | PEP 621 metadata; `[project.optional-dependencies]` for `shap`, `rbo`, `plotly` (research extras). Build backend: `hatchling` or `setuptools`. |
| 1.2 | Add `smx/__init__.py` | Expose the public API: `SMXPipeline`, model classes, preprocessing classes. |
| 1.3 | Add `smx/_version.py` | `__version__ = "0.1.0"` — single source of truth. |
| 1.4 | Remove `sys.path` hacks | `experiments/run_experiment.py` should `import smx` after `pip install -e .` |
| 1.5 | Move top-level imports | Replace all in-function `import numpy as np` with module-level imports. |

### Phase 2 — Preprocessing as fit/transform classes

| # | Task | Details |
|---|------|---------|
| 2.1 | `PoissonScaler` | Stores `mean_original_`, `mean_poisson_`; `fit(X)`, `transform(X)`, `fit_transform(X)`. |
| 2.2 | `ParetoScaler` | Stores `escala_pareto_`, `mean_pareto_`; same interface. |
| 2.3 | `MeanCenterer` | Stores `mean_`; trivial wrapper. |
| 2.4 | `AutoScaler` | Stores `mean_`, `std_`; standard-scaler style. |
| 2.5 | `SavGolFilter` | Stateless; wraps `scipy.signal.savgol_filter` with consistent API. |
| 2.6 | `MSCCorrector` | Stores reference spectrum; fit/transform. |
| 2.7 | `PreprocessingPipeline` | Ordered list of steps; `fit(Xcal)` → `transform(Xpred)`. Replaces the ad-hoc `preprocess()` function in `run_experiment.py`. Config-driven (`from_config(list_of_dicts)`) constructor. |

**Why classes?** The current code scatters "fit on cal, manually apply to pred" logic across `run_experiment.py`. A fit/transform pattern makes this fool-proof and sklearn-compatible.

### Phase 3 — Models with unified interface

| # | Task | Details |
|---|------|---------|
| 3.1 | `BaseSpectralModel(ABC)` | Abstract: `fit(X, y)`, `predict(X)`, `predict_continuous(X)`, `get_results() → ModelResult`. |
| 3.2 | `ModelResult` dataclass | Fields: `metrics_df`, `cal_predictions`, `pred_predictions`, `model`, `continuous_predictions`, `vip_scores` (optional). Removes the fragile positional-tuple returns. |
| 3.3 | `PLSModel` | Wraps `PLSRegression`; handles regression vs. classification via `aim`; LV-loop baked in. Exposes `vip_scores` and `explained_variance` as properties. |
| 3.4 | `SVMModel` | Wraps `SVR`/`SVC`; stores proba and decision outputs. |
| 3.5 | `MLPModel` | Wraps `MLPRegressor`/`MLPClassifier`. |
| 3.6 | Extract shared metric computation | The repeated R², RMSE, RPD, RPIQ, Bias, tBias block → private `_compute_regression_metrics(y_true, y_pred)` helper. Same for the confusion-matrix block → `_compute_classification_metrics(y_true, y_pred)`. |

### Phase 4 — Explaining pipeline (break the monolith)

| # | Task | Details |
|---|------|---------|
| 4.1 | `smx/zones/extraction.py` | Move `extract_spectral_zones` here (unchanged logic). |
| 4.2 | `smx/zones/aggregation.py` | `ZoneAggregator` class. Constructor takes `method='pca'|'sum'|'mean'|...`. `fit(zones_dict)` → stores PCA info. `transform(zones_dict)` → returns scores DataFrame. Replaces `aggregate_spectral_zones` + `aggregate_spectral_zones_pca`. |
| 4.3 | `smx/predicates/generation.py` | `PredicateGenerator` class. `from_quantiles(zone_scores_df, quantiles)` → stores `predicates_df`, `indicator_df`, `co_occurrence_matrix`. |
| 4.4 | `smx/predicates/bagging.py` | `PredicateBagger` class. `run(zone_scores_df, y_pred, predicates_df, n_bags, ...)` → returns `bags_dict`. |
| 4.5 | `smx/predicates/metrics.py` | Strategy pattern: `BasePredicateMetric(ABC)` with `compute(bags_dict) → dict[str, DataFrame]`. Implementations: `CovarianceMetric`, `PerturbationMetric`. `PerturbationMetric` absorbs the enormous `calculate_predicate_perturbation` logic. |
| 4.6 | `smx/graph/builder.py` | `PredicateGraphBuilder.build(bags_result, ranking_dict, ...) → nx.DiGraph`. |
| 4.7 | `smx/graph/centrality.py` | `compute_lrc(graph, predicates_df)` and `aggregate_lrc_across_seeds(lrc_by_seed, seeds)`. |
| 4.8 | `smx/graph/interpretation.py` | `map_thresholds_to_natural`, `reconstruct_threshold_to_spectrum`, `extract_predicate_info` helpers. |

### Phase 5 — Separate debugging/research utilities into `_contrib`

| # | Task | Details |
|---|------|---------|
| 5.1 | `smx/_contrib/importance.py` | `vip_scores_per_zone`, `regression_coefficients_per_zone`, `svm_pvector_per_zone`. |
| 5.2 | `smx/_contrib/shap_utils.py` | `shap_per_zone`. |
| 5.3 | `smx/_contrib/rbo.py` | `rbo_rank_comparison`. |
| 5.4 | `smx/_contrib/permutation.py` | `permutation_importance_per_zone` (from `explaining.py`). |
| 5.5 | `smx/_contrib/export.py` | `export_performance_metrics`. |
| 5.6 | Mark as optional deps | `rbo`, `shap` move to `[project.optional-dependencies.dev]` in `pyproject.toml`. Import guarded with try/except + clear error message. |

**Naming rationale:** The `_contrib` prefix (underscore = internal/private by convention) communicates clearly that these modules are not part of the public API and may change without notice. Users installing from pip will have them available but won't see them in the documented surface.

### Phase 6 — High-level orchestrator / facade

| # | Task | Details |
|---|------|---------|
| 6.1 | `SMXPipeline` class | Top-level facade that chains: config → data → preprocessing → model → zones → predicates → bagging → metrics → graph → LRC → interpretation. Exposes `run(dataset, model_name, method)`. |
| 6.2 | Simplify `run_experiment.py` | Becomes a thin CLI that instantiates `SMXPipeline` + optionally runs `_contrib` debugging. Move to `examples/` or keep in `experiments/` as internal only. |

### Phase 7 — Code quality & testing

| # | Task | Details |
|---|------|---------|
| 7.1 | Add type hints | All public functions and class methods. |
| 7.2 | Docstrings | Numpy-style, English only (remove Portuguese remnants). |
| 7.3 | Tests | `tests/test_preprocessing.py`, `tests/test_models.py`, `tests/test_zones.py`, `tests/test_predicates.py`, `tests/test_graph.py` — at least one unit test per class/function with synthetic data. |
| 7.4 | `DatasetConfig` dataclass | Replace raw-dict config with a validated dataclass. Gives IDE auto-complete and catches missing fields early. |
| 7.5 | Logging | Replace all `print()` calls with `logging.getLogger(__name__)`. Let users control verbosity. |
| 7.6 | Remove dead code | `msc()` looks broken (inner loop logic), `modified_poisson` has integer division bug (`1//degre` should be `1/degre`). Fix or remove. |

---

## 4. Class Diagram (core)

```
SMXPipeline
 ├── DatasetConfig
 ├── PreprocessingPipeline
 │    └── [PoissonScaler | ParetoScaler | MeanCenterer | AutoScaler | SavGolFilter | MSCCorrector]
 ├── BaseSpectralModel  (ABC)
 │    ├── PLSModel  → ModelResult
 │    ├── SVMModel  → ModelResult
 │    └── MLPModel  → ModelResult
 ├── ZoneAggregator  (fit/transform, PCA or simple agg)
 ├── PredicateGenerator
 ├── PredicateBagger
 ├── BasePredicateMetric  (ABC)
 │    ├── CovarianceMetric
 │    └── PerturbationMetric
 ├── PredicateGraphBuilder  → nx.DiGraph
 └── LRCComputer  → lrc_df
```

---

## 5. Migration Strategy

**Approach:** Incremental, bottom-up refactoring. Each phase produces a working state.

1. **Phase 1** first — once packaging is in place, all subsequent work can be tested with `pip install -e .` and `pytest`.
2. **Phase 2-3** in parallel — preprocessing and models are independent of each other.
3. **Phase 4** depends on 2-3 being done (the explaining pipeline consumes model outputs and preprocessed data).
4. **Phase 5** can happen at any time (just moving existing functions).
5. **Phase 6** is the capstone — wiring everything with the facade.
6. **Phase 7** runs throughout (add tests as each module is refactored).

### Backward compatibility
- Keep the current flat-function API importable (via deprecation wrappers in `__init__.py`) for at least one minor version.
- `experiments/run_experiment.py` remains functional throughout — update it incrementally as modules are refactored.

---

## 6. Specific bugs / issues to fix during refactoring

| Location | Issue |
|----------|-------|
| `preprocessings.py:modified_poisson` | `1//degre` is integer division — should be `1/degre` for fractional exponent. |
| `preprocessings.py:msc` | Inner loop `for j in range(0, sampleCount, 10)` nested inside `for i` with `ref[i]` — likely index error when `i >= len(ref)`. Logic needs review. |
| `explaining.py:aggregate_spectral_zones` | Indentation bug: `aggregated_df = pd.DataFrame(...)` is inside the `for` loop body instead of outside — the intermediate DataFrame is rebuilt each iteration and only the last zone survives. |
| `modeling.py:pls_optimized` | Classification branch returns 7 values; regression returns 5. Callers must branch on `aim` to unpack — fragile. |
| `modeling.py:explained_variance_from_scores` | Returns inside the `if Q is not None` block — returns `None` implicitly when called without Y. |
| `config.py` | `CONFIGS_DIR` is hardcoded relative to package source — breaks when installed as a package. Should accept a user-supplied path or env var. |
| `debugging.py:rbo_rank_comparison` | Uses `pd.concat` inside a loop to build a DataFrame — O(n²) pattern. Build a list of dicts and concat once. |
| `explaining.py:calculate_lrc` | Has a `return lrc_by_seed` *inside* the `for seed` loop — returns after the first seed only. |

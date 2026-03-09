# SMX Refactoring Plan

## 1. Current State Analysis

### Repository layout

There are now **two repositories**:

| Repo | Purpose | Audience |
|------|---------|----------|
| **`SMX`** (submodule at `./SMX/`) | Minimal public library — the core SMX algorithm only | External users / `pip install smx` |
| **`SMX_research`** (this repo) | Research infrastructure — preprocessing, models, experiment runners, datasets, debugging utilities | Maintainers / researchers |

`SMX_research` includes `SMX` as a Git submodule and depends on it (`smx` is a normal import).

### SMX — the algorithm library
SMX is a Python library that builds interpretable explanations for classification models trained on spectral data (XRF, Vis-NIR, etc.).
Given a trained model's continuous predictions and a set of spectral zones, SMX:

1. **Extracts spectral zones** → aggregates via PCA (PC1 scores)
2. **Generates predicates** from quantile thresholds on zone scores
3. **Bags + computes metrics** (covariance / perturbation) per predicate
4. **Builds a directed graph** of predicates (weighted by metric × explained variance)
5. **Computes LRC** (Local Reaching Centrality) → ranks spectral zones by importance
6. **Maps thresholds back** to natural spectral space for interpretation

SMX does **not** handle data loading, preprocessing, or model training — those are the user's (or SMX_research's) responsibility.

### SMX_research — the experiments repo
Wraps SMX with everything needed to reproduce research results:

- **Preprocessing** (Poisson, MC, Savgol, Pareto, auto-scaling, MSC)
- **Model training** (PLS-DA, SVM, MLP) with CV + metrics
- **Dataset configs** (JSON) and **real datasets** (CSV)
- **Synthetic data** generation
- **Debugging/comparison utilities** (VIP-per-zone, SHAP, RBO, permutation importance, metric export)
- **Experiment runners** (notebooks + CLI scripts)

### Current file map (`smx_temp/`)

| File | Lines | Role |
|------|------:|------|
| `config.py` | 33 | Dataset config loading (JSON) |
| `preprocessings.py` | ~120 | Loose functions: poisson, pareto, mc, auto_scaling, msc |
| `modeling.py` | 753 | `pls_optimized`, `svm_optimized`, `mlp_optimized` + VIP/explained-variance helpers |
| `explaining.py` | 2 514 | **19 core functions** (zones, predicates, bagging, metrics, graph, LRC, threshold mapping, plotting) + 1 research utility (`permutation_importance_per_zone`) |
| `debugging.py` | ~130 | VIP/reg-coef/SHAP/SVM-pvector per zone, RBO comparison, export performance metrics |
| `synthetic.py` | ~160 | Synthetic Gaussian-peak spectral data generator |
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

### 2a. `SMX/` — Public library (submodule)

Minimal, focused on the core algorithm only. No preprocessing, no models, no dataset configs.

```
SMX/
├── pyproject.toml               # PEP 621 metadata, minimal deps (numpy, pandas, scikit-learn, networkx)
├── README.md                    # Quick-start, installation, example
├── CHANGELOG.md
│
└── smx/
    ├── __init__.py              # Public API surface
    ├── _version.py              # Single-source version string
    │
    ├── zones/
    │   ├── __init__.py
    │   ├── extraction.py        # extract_spectral_zones()
    │   └── aggregation.py       # ZoneAggregator: sum/mean/max/PCA, etc.
    │
    ├── predicates/
    │   ├── __init__.py
    │   ├── generation.py        # PredicateGenerator (quantile-based)
    │   ├── bagging.py           # PredicateBagger
    │   └── metrics.py           # CovarianceMetric, PerturbationMetric (strategy)
    │
    ├── graph/
    │   ├── __init__.py
    │   ├── builder.py           # PredicateGraphBuilder
    │   ├── centrality.py        # LRC computation + seed aggregation
    │   └── interpretation.py    # map_thresholds_to_natural, reconstruct_threshold_to_spectrum
    │
    ├── plotting/
    │   ├── __init__.py
    │   └── threshold.py         # plot_threshold_spectrum (plotly)
    │
    └── tests/
        ├── test_zones.py
        ├── test_predicates.py
        └── test_graph.py
```

**Design principles for SMX:**
- **Model-agnostic** — accepts continuous predictions (numpy array / pandas Series) from any model. No sklearn dependency beyond what's needed internally (PCA for zone aggregation).
- **No I/O** — no file reading, no config loading. Data comes in as DataFrames/arrays.
- **Minimal dependencies** — `numpy`, `pandas`, `scikit-learn` (PCA only), `networkx`, `plotly` (optional).

### 2b. `SMX_research/` — Research repository (this repo)

Everything needed to reproduce experiments. Depends on `smx` (installed from the submodule).

```
SMX_research/
├── pyproject.toml               # Depends on smx + heavy deps (shap, rbo, etc.)
├── SMX/                         # Git submodule → github.com/joseviniciusr/SMX
│
├── smx_research/                # Research-only Python package
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── scalers.py           # PoissonScaler, ParetoScaler, AutoScaler, MeanCenterer
│   │   ├── spectral.py          # SavGolFilter, MSCCorrector
│   │   └── pipeline.py          # PreprocessingPipeline (chains steps, fit/transform)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseSpectralModel (ABC) – unified return type
│   │   ├── pls.py               # PLSModel (wraps PLSRegression for reg + DA)
│   │   ├── svm.py               # SVMModel (wraps SVR/SVC)
│   │   └── mlp.py               # MLPModel (wraps MLPRegressor/MLPClassifier)
│   │
│   ├── config.py                # DatasetConfig dataclass + load/list helpers
│   ├── synthetic.py             # SyntheticSpectraGenerator (class)
│   │
│   └── _contrib/                # Maintainer-only / research utilities
│       ├── __init__.py
│       ├── importance.py        # VIP-per-zone, reg-coef-per-zone, SVM-pvector-per-zone
│       ├── shap_utils.py        # SHAP zone mapping
│       ├── rbo.py               # RBO rank comparison
│       ├── permutation.py       # permutation_importance_per_zone
│       └── export.py            # export_performance_metrics
│
├── real_datasets/               # CSV + JSON configs (stays as-is)
├── experiments/                 # Notebooks + CLI runners
├── smx_temp/                    # Legacy code (to be deleted after migration)
└── plan.md                      # This file
```

---

## 3. Detailed Refactoring Steps

### Phase 1 — Packaging & project skeletons (both repos)

| # | Task | Repo | Details |
|---|------|------|---------|
| 1.1 | Create `SMX/pyproject.toml` | SMX | PEP 621 metadata. Deps: `numpy`, `pandas`, `scikit-learn`, `networkx`. Optional: `plotly`. Build backend: `hatchling` or `setuptools`. |
| 1.2 | Create `SMX/smx/__init__.py` | SMX | Expose: `ZoneAggregator`, `PredicateGenerator`, `PredicateBagger`, `CovarianceMetric`, `PerturbationMetric`, `PredicateGraphBuilder`, `compute_lrc`, `map_thresholds_to_natural`. |
| 1.3 | Create `SMX/smx/_version.py` | SMX | `__version__ = "0.1.0"` |
| 1.4 | Create `SMX_research/pyproject.toml` | SMX_research | Deps: `smx` (path dep to `./SMX`), plus `shap`, `rbo`, `plotly`, `scipy`, etc. |
| 1.5 | Create `smx_research/__init__.py` | SMX_research | Expose preprocessing, model, and config classes. |
| 1.6 | Remove `sys.path` hacks | SMX_research | `experiments/run_experiment.py` uses `import smx` and `import smx_research` after `pip install -e .` for both. |
| 1.7 | Move top-level imports | Both | Replace all in-function `import numpy as np` with module-level imports. |

### Phase 2 — Core algorithm → SMX (break the monolith)

All of these go into the **SMX** submodule. Source: `smx_temp/explaining.py`.

| # | Task | Details |
|---|------|---------|
| 2.1 | `smx/zones/extraction.py` | Move `extract_spectral_zones` (unchanged logic). |
| 2.2 | `smx/zones/aggregation.py` | `ZoneAggregator` class. Constructor takes `method='pca'\|'sum'\|'mean'\|...`. `fit(zones_dict)` → stores PCA info. `transform(zones_dict)` → returns scores DataFrame. Replaces `aggregate_spectral_zones` + `aggregate_spectral_zones_pca`. |
| 2.3 | `smx/predicates/generation.py` | `PredicateGenerator` class. `from_quantiles(zone_scores_df, quantiles)` → stores `predicates_df`, `indicator_df`, `co_occurrence_matrix`. |
| 2.4 | `smx/predicates/bagging.py` | `PredicateBagger` class. `run(zone_scores_df, y_pred, predicates_df, n_bags, ...)` → returns `bags_dict`. |
| 2.5 | `smx/predicates/metrics.py` | Strategy pattern: `BasePredicateMetric(ABC)` with `compute(bags_dict) → dict[str, DataFrame]`. Implementations: `CovarianceMetric`, `PerturbationMetric`. `PerturbationMetric` absorbs the enormous `calculate_predicate_perturbation` logic. |
| 2.6 | `smx/graph/builder.py` | `PredicateGraphBuilder.build(bags_result, ranking_dict, ...) → nx.DiGraph`. |
| 2.7 | `smx/graph/centrality.py` | `compute_lrc(graph, predicates_df)` and `aggregate_lrc_across_seeds(lrc_by_seed, seeds)`. |
| 2.8 | `smx/graph/interpretation.py` | `map_thresholds_to_natural`, `reconstruct_threshold_to_spectrum`, `extract_predicate_info` helpers. |
| 2.9 | `smx/plotting/threshold.py` | `plot_threshold_spectrum` (plotly). Optional dep — guarded import. |

### Phase 3 — Preprocessing as fit/transform classes (SMX_research)

| # | Task | Details |
|---|------|---------|
| 3.1 | `PoissonScaler` | Stores `mean_original_`, `mean_poisson_`; `fit(X)`, `transform(X)`, `fit_transform(X)`. |
| 3.2 | `ParetoScaler` | Stores `escala_pareto_`, `mean_pareto_`; same interface. |
| 3.3 | `MeanCenterer` | Stores `mean_`; trivial wrapper. |
| 3.4 | `AutoScaler` | Stores `mean_`, `std_`; standard-scaler style. |
| 3.5 | `SavGolFilter` | Stateless; wraps `scipy.signal.savgol_filter` with consistent API. |
| 3.6 | `MSCCorrector` | Stores reference spectrum; fit/transform. |
| 3.7 | `PreprocessingPipeline` | Ordered list of steps; `fit(Xcal)` → `transform(Xpred)`. Config-driven (`from_config(list_of_dicts)`) constructor. |

**Why classes?** The current code scatters "fit on cal, manually apply to pred" logic across `run_experiment.py`. A fit/transform pattern makes this fool-proof and sklearn-compatible.

### Phase 4 — Models with unified interface (SMX_research)

| # | Task | Details |
|---|------|---------|
| 4.1 | `BaseSpectralModel(ABC)` | Abstract: `fit(X, y)`, `predict(X)`, `predict_continuous(X)`, `get_results() → ModelResult`. |
| 4.2 | `ModelResult` dataclass | Fields: `metrics_df`, `cal_predictions`, `pred_predictions`, `model`, `continuous_predictions`, `vip_scores` (optional). Removes fragile positional-tuple returns. |
| 4.3 | `PLSModel` | Wraps `PLSRegression`; handles regression vs. classification via `aim`; LV-loop baked in. Exposes `vip_scores` and `explained_variance` as properties. |
| 4.4 | `SVMModel` | Wraps `SVR`/`SVC`; stores proba and decision outputs. |
| 4.5 | `MLPModel` | Wraps `MLPRegressor`/`MLPClassifier`. |
| 4.6 | Extract shared metric computation | The repeated R², RMSE, RPD, RPIQ, Bias, tBias block → private `_compute_regression_metrics(y_true, y_pred)` helper. Same for the confusion-matrix block → `_compute_classification_metrics(y_true, y_pred)`. |

### Phase 5 — Research utilities → `_contrib` (SMX_research)

| # | Task | Details |
|---|------|---------|
| 5.1 | `smx_research/_contrib/importance.py` | `vip_scores_per_zone`, `regression_coefficients_per_zone`, `svm_pvector_per_zone`. |
| 5.2 | `smx_research/_contrib/shap_utils.py` | `shap_per_zone`. |
| 5.3 | `smx_research/_contrib/rbo.py` | `rbo_rank_comparison`. |
| 5.4 | `smx_research/_contrib/permutation.py` | `permutation_importance_per_zone` (from `explaining.py`). |
| 5.5 | `smx_research/_contrib/export.py` | `export_performance_metrics`. |
| 5.6 | Mark as optional deps | `rbo`, `shap` in `[project.optional-dependencies.dev]`. Import guarded with try/except + clear error message. |

### Phase 6 — High-level orchestrator (SMX_research)

| # | Task | Details |
|---|------|---------|
| 6.1 | `SMXPipeline` class | Facade in `smx_research` that chains: config → data → preprocessing → model → `smx` core (zones → predicates → bagging → metrics → graph → LRC → interpretation). Exposes `run(dataset, model_name, method)`. |
| 6.2 | Simplify `run_experiment.py` | Becomes a thin CLI that instantiates `SMXPipeline` + optionally runs `_contrib` debugging. |

### Phase 7 — Code quality & testing (both repos)

| # | Task | Repo | Details |
|---|------|------|---------|
| 7.1 | Add type hints | Both | All public functions and class methods. |
| 7.2 | Docstrings | Both | Numpy-style, English only (remove Portuguese remnants). |
| 7.3 | Tests (core) | SMX | `tests/test_zones.py`, `tests/test_predicates.py`, `tests/test_graph.py` — unit tests with synthetic data. |
| 7.4 | Tests (research) | SMX_research | `tests/test_preprocessing.py`, `tests/test_models.py` — integration tests. |
| 7.5 | `DatasetConfig` dataclass | SMX_research | Replace raw-dict config with a validated dataclass. |
| 7.6 | Logging | Both | Replace all `print()` calls with `logging.getLogger(__name__)`. |
| 7.7 | Remove dead code | SMX_research | `msc()` looks broken (inner loop logic), `modified_poisson` has integer division bug (`1//degre` should be `1/degre`). Fix or remove. |

---

## 4. Class Diagrams

### SMX (public library)

```
smx
 ├── ZoneAggregator  (fit/transform, PCA or simple agg)
 ├── PredicateGenerator
 ├── PredicateBagger
 ├── BasePredicateMetric  (ABC)
 │    ├── CovarianceMetric
 │    └── PerturbationMetric
 ├── PredicateGraphBuilder  → nx.DiGraph
 └── LRCComputer  → lrc_df
```

### SMX_research (research layer, depends on smx)

```
smx_research
 ├── DatasetConfig
 ├── PreprocessingPipeline
 │    └── [PoissonScaler | ParetoScaler | MeanCenterer | AutoScaler | SavGolFilter | MSCCorrector]
 ├── BaseSpectralModel  (ABC)
 │    ├── PLSModel  → ModelResult
 │    ├── SVMModel  → ModelResult
 │    └── MLPModel  → ModelResult
 ├── SyntheticSpectraGenerator
 └── SMXPipeline  (facade — chains smx_research models + smx core)
```

---

## 5. Migration Strategy

**Approach:** Incremental, bottom-up refactoring. Each phase produces a working state.

1. **Phase 1** first — packaging for both repos. Once in place, all subsequent work can be tested with `pip install -e ./SMX && pip install -e .` and `pytest`.
2. **Phase 2** — extract the core algorithm into `SMX/smx/`. This is the critical split.
3. **Phase 3-4** in parallel — preprocessing and models are independent of each other, both stay in `SMX_research`.
4. **Phase 5** can happen at any time (just moving existing functions within `SMX_research`).
5. **Phase 6** is the capstone — wiring the facade in `SMX_research` that calls `smx`.
6. **Phase 7** runs throughout (add tests as each module is refactored).

### Dependency flow
```
SMX_research  ──depends on──▶  SMX (submodule)
     │                            │
 smx_research package          smx package
 (preprocessing, models,       (zones, predicates,
  config, _contrib)             graph, plotting)
```

External users install **only** `smx`. Researchers clone `SMX_research` (which pulls the submodule) and install both.

### Backward compatibility
- Keep `smx_temp/` untouched until migration is complete, then delete.
- `experiments/run_experiment.py` remains functional throughout — update incrementally.
- No deprecation wrappers needed in `smx` since it's a fresh package.

---

## 6. Specific bugs / issues to fix during refactoring

| Location | Repo | Issue |
|----------|------|-------|
| `preprocessings.py:modified_poisson` | SMX_research | `1//degre` is integer division — should be `1/degre` for fractional exponent. |
| `preprocessings.py:msc` | SMX_research | Inner loop `for j in range(0, sampleCount, 10)` nested inside `for i` with `ref[i]` — likely index error when `i >= len(ref)`. Logic needs review. |
| `explaining.py:aggregate_spectral_zones` | SMX | Indentation bug: `aggregated_df = pd.DataFrame(...)` is inside the `for` loop body instead of outside — the intermediate DataFrame is rebuilt each iteration and only the last zone survives. |
| `modeling.py:pls_optimized` | SMX_research | Classification branch returns 7 values; regression returns 5. Callers must branch on `aim` to unpack — fragile. |
| `modeling.py:explained_variance_from_scores` | SMX_research | Returns inside the `if Q is not None` block — returns `None` implicitly when called without Y. |
| `config.py` | SMX_research | `CONFIGS_DIR` is hardcoded relative to package source — breaks when installed as a package. Should accept a user-supplied path or env var. |
| `debugging.py:rbo_rank_comparison` | SMX_research | Uses `pd.concat` inside a loop to build a DataFrame — O(n²) pattern. Build a list of dicts and concat once. |
| `explaining.py:calculate_lrc` | SMX | Has a `return lrc_by_seed` *inside* the `for seed` loop — returns after the first seed only. |

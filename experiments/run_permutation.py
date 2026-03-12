"""
Permutation importance generator for SMX pipeline.

Computes permutation feature importance per energy feature and saves the CSV
file that is consumed by run_analysis.py (via debugging.permutation_per_zone()).

Usage:
    python experiments/run_permutation.py --dataset tomato --model mlp
    python experiments/run_permutation.py --dataset all --model all
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import build_effective_config, load_dataset_config, list_available_datasets
from run_experiment import load_data, preprocess, train_model, MODEL_CONFIG

# ── Permutation scoring-function dispatch ────────────────────────────────────
# PLS uses raw predictions; MLP/SVM use P(positive class).
PERMUTATION_SCORING_FN = {
    'pls': None,
    'mlp': lambda model: lambda X: model.predict_proba(X)[:, 1],
    'svm': lambda model: lambda X: model.predict_proba(X)[:, 1],
}


# ── Core computation ─────────────────────────────────────────────────────────

def permutation_importance(estimator, X, n_repeats=10, random_state=42,
                           scoring_fn=None):
    """Compute permutation feature importance per energy feature.

    Parameters
    ----------
    estimator : fitted model with a ``predict`` method.
    X : pd.DataFrame
        Preprocessed calibration data.
    n_repeats : int, default 10
        Number of permutation repeats per feature.
    random_state : int, default 42
        Random seed.
    scoring_fn : callable, optional
        Custom prediction function ``f(X) -> predictions``.
        If *None*, uses ``estimator.predict()``.

    Returns
    -------
    permutation_df : pd.DataFrame
        Per-feature permutation importance with columns
        ``energy`` and ``Permutation_importance``, sorted descending.
    """
    rng = np.random.RandomState(random_state)
    predict = scoring_fn if scoring_fn is not None else estimator.predict
    baseline_pred = predict(X)
    importance_list = []
    X_arr = X.copy()

    for col in X.columns:
        diffs = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_pred = predict(X_perm)
            diffs.append(np.mean(np.abs(baseline_pred - perm_pred)))
        importance_list.append(np.mean(diffs))

    permutation_df = pd.DataFrame({
        'energy': X.columns,
        'Permutation_importance': importance_list,
    })
    permutation_df.sort_values('Permutation_importance', ascending=False, inplace=True)
    return permutation_df


# ── Pipeline routine ─────────────────────────────────────────────────────────

def run_permutation(dataset, model_name, new_only=False):
    """Compute and save permutation importance for a (dataset, model) combination."""
    print(f"\n{'#'*70}")
    print(f"# Permutation Importance: dataset={dataset}, model={model_name}")
    print(f"{'#'*70}\n")

    # 0. Load config
    config = build_effective_config(dataset, model_name)

    # 0a. Skip if output already exists and --new-only was requested
    if new_only:
        output_path = SCRIPT_DIR / model_name.upper() / dataset / f'permutation_{config["name"]}.csv'
        if output_path.exists():
            print(f"Skipping (already exists): {output_path}")
            return

    # 1. Set seed for full reproducibility (same as run_experiment)
    if 'seed' not in config:
        raise ValueError(
            f"Dataset config for '{dataset}' is missing the required 'seed' field."
        )
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2. Validate model compatibility
    if model_name not in config.get('compatible_models', []):
        print(f"WARNING: model '{model_name}' not compatible with dataset '{dataset}'. Skipping.")
        return

    # 3. Load data
    print("Loading data...")
    Xcalclass, Xpredclass, ycalclass, ypredclass = load_data(config)
    print(f"  Xcal: {Xcalclass.shape}, Xpred: {Xpredclass.shape}")

    # 4. Preprocess
    print("Preprocessing...")
    Xcalclass_prep, Xpredclass_prep, prep_info = preprocess(config, Xcalclass, Xpredclass)

    # 5. Train model (identical to run_experiment to guarantee same weights)
    print(f"Training {model_name}...")
    result = train_model(model_name, config, Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass)

    # 6. Extract fitted model object
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)

    # 7. Compute permutation importance
    print("Computing permutation importance...")
    scoring_fn = PERMUTATION_SCORING_FN[model_name]
    if scoring_fn is not None:
        scoring_fn = scoring_fn(model)

    perm_df = permutation_importance(
        estimator=model,
        X=Xcalclass_prep,
        n_repeats=10,
        random_state=42,
        scoring_fn=scoring_fn,
    )

    # 7b. Map energy to spectral zone
    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]
    energy_to_zone = {}
    for zone_name, start, end in spectral_cuts:
        for e in perm_df['energy']:
            if start <= float(e) <= end:
                energy_to_zone[e] = zone_name
    perm_df['Zone'] = perm_df['energy'].map(energy_to_zone)

    # 8. Save to the same path that run_analysis.py expects
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'permutation_{config["name"]}.csv'
    perm_df.to_csv(output_path, index=False, sep=';')
    print(f"Permutation importance saved to {output_path}")
    return perm_df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SMX Permutation Importance Generator')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--new-only', action='store_true',
                        help='Skip dataset/model combinations whose permutation_*.csv already exists')
    args = parser.parse_args()

    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                start_time = time.time()
                run_permutation(ds, mdl, new_only=args.new_only)
                elapsed = time.time() - start_time

                # Save technique metrics
                config = load_dataset_config(ds)
                output_dir = SCRIPT_DIR / mdl.upper() / ds
                output_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = output_dir / 'technique_metrics.csv'

                # Load or create metrics dataframe
                if metrics_path.exists():
                    metrics_df = pd.read_csv(metrics_path, sep=';')
                else:
                    metrics_df = pd.DataFrame(columns=['Technique', 'Runtime_seconds'])

                # Update or add Permutation row
                mask = metrics_df['Technique'] == 'Permutation'
                if mask.any():
                    metrics_df.loc[mask, 'Runtime_seconds'] = elapsed
                else:
                    metrics_df = pd.concat([metrics_df, pd.DataFrame(
                        {'Technique': ['Permutation'], 'Runtime_seconds': [elapsed]}
                    )], ignore_index=True)

                metrics_df.to_csv(metrics_path, index=False, sep=';')
                print(f"  Permutation runtime: {elapsed:.2f}s → {metrics_path}")
            except Exception as e:
                print(f"\nERROR running permutation for {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

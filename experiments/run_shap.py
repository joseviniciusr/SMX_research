"""
SHAP value generator for SMX pipeline.

Computes SHAP global importance per energy feature and saves the CSV file
that is consumed by debugging.shap_per_zone().

Usage:
    python experiments/run_shap.py --dataset tomato --model mlp
    python experiments/run_shap.py --dataset all --model all
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
SMX_DIR = WORKSPACE_ROOT / 'smx'
if str(SMX_DIR) not in sys.path:
    sys.path.insert(0, str(SMX_DIR))

from config import load_dataset_config, list_available_datasets
from run_experiment import load_data, preprocess, train_model, MODEL_CONFIG

# ── SHAP predict-function dispatch ───────────────────────────────────────────
# PLS outputs continuous predictions; MLP/SVM output P(positive class).
SHAP_PREDICT_FN = {
    'pls': lambda model: model.predict,
    'mlp': lambda model: lambda x: model.predict_proba(x)[:, 1],
    'svm': lambda model: lambda x: model.predict_proba(x)[:, 1],
}


# ── Core routine ─────────────────────────────────────────────────────────────

def run_shap(dataset, model_name):
    """Compute and save SHAP values for a (dataset, model) combination."""
    print(f"\n{'#'*70}")
    print(f"# SHAP: dataset={dataset}, model={model_name}")
    print(f"{'#'*70}\n")

    # 0. Load config
    config = load_dataset_config(dataset)

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

    # 7. Compute SHAP values
    print("Computing SHAP values (KernelExplainer)...")
    predict_fn = SHAP_PREDICT_FN[model_name](model)
    explainer = shap.KernelExplainer(predict_fn, Xcalclass_prep)
    shap_exp = explainer(Xcalclass_prep)

    shap_global_importance = pd.DataFrame({
        'energy': Xcalclass_prep.columns,
        'Mean_Abs_SHAP': np.abs(shap_exp.values).mean(axis=0)
    })
    shap_global_importance.sort_values(by='Mean_Abs_SHAP', ascending=False, inplace=True)

    # 8. Save to the same path that run_experiment / debugging.py expects
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'shap_{config["name"]}.csv'
    shap_global_importance.to_csv(output_path, index=False, sep=';')
    print(f"SHAP values saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SMX SHAP Value Generator')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    args = parser.parse_args()

    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                run_shap(ds, mdl)
            except Exception as e:
                print(f"\nERROR running SHAP for {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

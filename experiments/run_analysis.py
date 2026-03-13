"""
Post-experiment analysis for SMX research.

Available analyses (select via flags):
  --rbo            Build feature_importance.csv and rbo_rank.csv from the
                   individual technique output files produced by run_experiment,
                   run_shap, and run_permutation.
  --faithfulness   Faithfulness evaluation: progressively mask top-k ranked
                   zones and measure F1 / Accuracy degradation for each XAI
                   method (SMX, SHAP, Permutation, and for PLS: VIP, reg_vet).
  --instability    Stability evaluation: vary the global split seed to produce
                   different stratified train/test partitions, train the model
                   for each split, run the full SMX pipeline (via the ``SMX``
                   facade class) on the training set, and collect predicate
                   rankings across seeds for later pairwise RBO analysis.

Only techniques whose output files exist are included – missing techniques
are silently skipped so you can run analysis at any stage.

Usage:
    # RBO ranking comparison (original behaviour)
    python experiments/run_analysis.py --rbo --dataset soil --model mlp
    python experiments/run_analysis.py --rbo --dataset all --model all

    # Faithfulness evaluation (masking top-k zones, default mask=zero)
    python experiments/run_analysis.py --faithfulness --dataset bank_notes --model mlp
    python experiments/run_analysis.py --faithfulness --dataset soil --model pls --mask_mode median
    python experiments/run_analysis.py --faithfulness --dataset all --model all --mask_mode zero

    # Run both analyses at once
    python experiments/run_analysis.py --rbo --faithfulness --dataset bank_notes --model mlp

    # Instability evaluation (vary split seed, extract SMX rankings per seed)
    python experiments/run_analysis.py --instability --dataset soil --model mlp --seed_number 5 --smx_seed_number 4
    python experiments/run_analysis.py --instability --dataset bank_notes --model pls --seed_number 20 --smx_seed_number 4
    python experiments/run_analysis.py --instability --dataset all --model all --seed_number 10 --smx_seed_number 3 --method smx_perturbation
    python experiments/run_analysis.py --instability --dataset all --model all --seed_number 10 --method shap permutation
    python experiments/run_analysis.py --instability --dataset all --model all --seed_number 10 --smx_seed_number 3 --method all

"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, accuracy_score

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sklearn.model_selection import train_test_split as sklearn_train_test_split

from config import build_effective_config, list_available_datasets
import debugging as dbg
import smx
from smx import SMX
from run_experiment import (
    load_data, preprocess, train_model, extract_y_continuous,
    MODEL_CONFIG,
)
from smx.datasets.synthetic import generate_synthetic_spectral_data


# ── Helpers ──────────────────────────────────────────────────────────────────

def _zone_ranking_from_lrc(lrc_csv_path):
    """Read an LRC natural CSV and return the zone ranking (deduplicated).

    Returns a list of zone names ordered by descending Local_Reaching_Centrality.
    """
    df = pd.read_csv(lrc_csv_path, sep=';')
    df = df.sort_values('Local_Reaching_Centrality', ascending=False)
    df = df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
    return list(df['Zone'])


def _zone_ranking_from_per_energy_csv(csv_path, score_column, spectral_cuts):
    """Read a per-energy importance CSV and return zone ranking.

    Applies energy→zone mapping, deduplicates by zone (keeps highest score),
    and returns a list of zone names ordered by descending score.
    """
    df = pd.read_csv(csv_path, sep=';')
    df['Zone'] = df['energy'].map(
        dbg._map_energy_to_zone(df['energy'], spectral_cuts)
    )
    df = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
    return list(df['Zone'])


def _zone_ranking_from_zone_csv(csv_path, zone_column='Zone'):
    """Read a CSV that already has a Zone column and return the zone ranking.

    Assumes rows are already in rank order (descending importance).
    """
    df = pd.read_csv(csv_path, sep=';')
    return list(df[zone_column])


# ── Instability method helpers ───────────────────────────────────────────────

INSTABILITY_METHOD_ORDER = [
    'smx_perturbation',
    'smx_covariance',
    'shap',
    'permutation',
]


def _normalize_instability_methods(method_selection):
    """Normalize method selection to canonical instability method names."""
    if isinstance(method_selection, str):
        requested = [method_selection]
    else:
        requested = list(method_selection)

    normalized = []
    for m in requested:
        if m == 'all':
            return list(INSTABILITY_METHOD_ORDER)
        if m in INSTABILITY_METHOD_ORDER and m not in normalized:
            normalized.append(m)

    return normalized


def _normalize_smx_lrc_methods(method_selection):
    """Normalize selection to the SMX LRC subset used by RBO/Faithfulness."""
    normalized = _normalize_instability_methods(method_selection)
    return [
        m for m in normalized
        if m in ('smx_perturbation', 'smx_covariance')
    ]


def _prediction_fn_for_model(model_name, model):
    """Return prediction function used by SHAP/Permutation for each model."""
    if model_name == 'pls':
        return model.predict
    return lambda X: model.predict_proba(X)[:, 1]


def _permutation_importance_from_model(model_name, model, X,
                                       n_repeats=10, random_state=42):
    """Compute permutation importance per energy using an already-fitted model."""
    rng = np.random.RandomState(random_state)
    predict_fn = _prediction_fn_for_model(model_name, model)
    baseline_pred = predict_fn(X)

    importance_list = []
    for col in X.columns:
        diffs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_pred = predict_fn(X_perm)
            diffs.append(np.mean(np.abs(baseline_pred - perm_pred)))
        importance_list.append(float(np.mean(diffs)))

    return pd.DataFrame({
        'energy': X.columns,
        'Permutation_importance': importance_list,
    }).sort_values('Permutation_importance', ascending=False).reset_index(drop=True)


def _shap_importance_from_model(model_name, model, X):
    """Compute SHAP global importance per energy using an already-fitted model."""
    predict_fn = _prediction_fn_for_model(model_name, model)
    explainer = shap.KernelExplainer(predict_fn, X, njobs=20)
    shap_explanation = explainer(X)

    vals = np.abs(shap_explanation.values)
    if vals.ndim == 3:
        score = vals.mean(axis=(0, 2))
    else:
        score = vals.mean(axis=0)

    return pd.DataFrame({
        'energy': X.columns,
        'Mean_Abs_SHAP': score,
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)


def _format_instability_energy_output(df_importance, score_col, spectral_cuts,
                                      split_seed, model_name, method_name):
    """Attach metadata and zone mapping while keeping all sorted energies."""
    out = df_importance.copy()
    out['Zone'] = out['energy'].map(dbg._map_energy_to_zone(out['energy'], spectral_cuts))
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out.insert(0, 'split_seed', split_seed)
    out.insert(1, 'method', method_name)
    out.insert(2, 'model', model_name)
    out.insert(3, 'rank', np.arange(1, len(out) + 1))
    return out


# ── Analysis routines ────────────────────────────────────────────────────────

# Canonical column order per model type
COLUMN_ORDER = {
    'pls': ['VIP_Score', 'Reg_Coefficient', 'Shap', 'Permutation',
             'LRC_perturbation', 'LRC_covariance'],
    'svm': ['SVM_pvector', 'Shap', 'Permutation',
             'LRC_perturbation', 'LRC_covariance'],
    'mlp': ['Shap', 'Permutation', 'LRC_perturbation', 'LRC_covariance'],
}


def build_feature_importance_table(model_name, output_dir, dataset_name,
                                   spectral_cuts, method='all'):
    """Build feature importance table from available output files.

    Parameters
    ----------
    model_name : str
        One of 'pls', 'mlp', 'svm'.
    output_dir : Path
        Directory containing the technique output files.
    dataset_name : str
        config['name'] value, used to locate technique CSVs.
    spectral_cuts : list of tuples
        Each tuple is (zone_name, start, end).
    method : str | list[str]
        Method selection using canonical names:
        'smx_perturbation', 'smx_covariance', 'shap', 'permutation', or 'all'.

    Returns
    -------
    pd.DataFrame
        Feature importance table with one column per available technique.
    """
    zone_lists = {}
    smx_lrc_methods = _normalize_smx_lrc_methods(method)

    # LRC results
    if 'smx_covariance' in smx_lrc_methods:
        cov_path = output_dir / 'lrc_cov_natural.csv'
        if cov_path.exists():
            zone_lists['LRC_covariance'] = _zone_ranking_from_lrc(cov_path)

    if 'smx_perturbation' in smx_lrc_methods:
        pert_path = output_dir / 'lrc_pert_natural.csv'
        if pert_path.exists():
            zone_lists['LRC_perturbation'] = _zone_ranking_from_lrc(pert_path)

    # SHAP
    shap_path = output_dir / f'shap_{dataset_name}.csv'
    if shap_path.exists():
        zone_lists['Shap'] = _zone_ranking_from_per_energy_csv(
            shap_path, 'Mean_Abs_SHAP', spectral_cuts
        )

    # Permutation
    perm_path = output_dir / f'permutation_{dataset_name}.csv'
    if perm_path.exists():
        zone_lists['Permutation'] = _zone_ranking_from_per_energy_csv(
            perm_path, 'Permutation_importance', spectral_cuts
        )

    # Model-specific importance
    vip_path = output_dir / f'vip_{dataset_name}.csv'
    if vip_path.exists():
        zone_lists['VIP_Score'] = _zone_ranking_from_per_energy_csv(
            vip_path, 'VIP_Score', spectral_cuts
        )

    reg_path = output_dir / f'reg_coef_{dataset_name}.csv'
    if reg_path.exists():
        zone_lists['Reg_Coefficient'] = _zone_ranking_from_per_energy_csv(
            reg_path, 'Abs_Reg_coef', spectral_cuts
        )

    pvector_path = output_dir / f'pvector_{dataset_name}.csv'
    if pvector_path.exists():
        zone_lists['SVM_pvector'] = _zone_ranking_from_per_energy_csv(
            pvector_path, 'Pvector', spectral_cuts
        )

    if not zone_lists:
        print("  No technique output files found — nothing to analyse.")
        return pd.DataFrame()

    # Pad to equal length
    max_len = max(len(v) for v in zone_lists.values())

    def pad(lst):
        return lst + [None] * (max_len - len(lst))

    fi_data = {k: pad(v) for k, v in zone_lists.items()}
    df = pd.DataFrame(fi_data)

    # Apply canonical column ordering
    ordered_cols = [c for c in COLUMN_ORDER.get(model_name, []) if c in zone_lists]
    return df[ordered_cols] if ordered_cols else df


def run_analysis(dataset, model_name, method='all'):
    """Run RBO analysis for a single (dataset, model) combination."""
    print(f"\n{'#'*70}")
    print(f"# RBO Analysis: dataset={dataset}, model={model_name}, method={method}")
    print(f"{'#'*70}\n")

    config = build_effective_config(dataset, model_name)
    dataset_name = config['name']
    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]
    output_dir = SCRIPT_DIR / model_name.upper() / dataset

    if not output_dir.exists():
        print(f"  Output directory does not exist: {output_dir}")
        print("  Run the experiment first.")
        return

    # Build feature importance table
    features_importance = build_feature_importance_table(
        model_name, output_dir, dataset_name, spectral_cuts, method=method
    )

    if features_importance.empty:
        return

    # Save feature importance
    fi_path = output_dir / 'feature_importance.csv'
    features_importance.to_csv(fi_path, index=False, sep=';')
    print("Feature importance table:")
    print(features_importance)
    print(f"\n  Saved to {fi_path}")

    # RBO rank comparison
    rbo_path = output_dir / 'rbo_rank.csv'
    dbg.rbo_rank_comparison(features_importance, rbo_path)
    print(f"  RBO rank saved to {rbo_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Faithfulness evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _get_zone_columns(zone_name, spectral_cuts, all_columns):
    """Return column names that belong to a spectral zone.

    Parameters
    ----------
    zone_name : str
        Name of the zone (must match a name in *spectral_cuts*).
    spectral_cuts : list of tuples
        Each tuple is ``(zone_name, start, end)``.
    all_columns : pd.Index
        All column names of the spectral DataFrame.

    Returns
    -------
    list of str
        Column names whose numeric value falls within [start, end].
    """
    zone_start = zone_end = None
    for cut in spectral_cuts:
        name, start, end = cut[0], float(cut[1]), float(cut[2])
        if name == zone_name:
            zone_start, zone_end = start, end
            break

    if zone_start is None:
        return []

    col_numeric = pd.to_numeric(all_columns.astype(str), errors='coerce')
    mask = (~np.isnan(col_numeric)) & (col_numeric >= zone_start) & (col_numeric <= zone_end)
    return list(all_columns[mask])


def _collect_zone_rankings(output_dir, dataset_name, spectral_cuts,
                           model_name, method='all'):
    """Read pre-computed ranking CSVs and return a dict of zone rankings.

    Returns ``{method_label: [zone_name_rank1, zone_name_rank2, ...]}``.
    Only methods whose files exist are included.  NaN / None entries are
    dropped so that every element is a valid zone name string.
    """
    rankings = {}

    smx_lrc_methods = _normalize_smx_lrc_methods(method)

    # SMX (LRC perturbation / covariance)
    if 'smx_perturbation' in smx_lrc_methods:
        lrc_path = output_dir / 'lrc_pert_natural.csv'
        if lrc_path.exists():
            rankings['SMX_perturbation'] = _zone_ranking_from_lrc(lrc_path)

    if 'smx_covariance' in smx_lrc_methods:
        lrc_cov_path = output_dir / 'lrc_cov_natural.csv'
        if lrc_cov_path.exists():
            rankings['SMX_covariance'] = _zone_ranking_from_lrc(lrc_cov_path)

    # SHAP
    shap_path = output_dir / f'shap_{dataset_name}.csv'
    if shap_path.exists():
        rankings['SHAP'] = _zone_ranking_from_per_energy_csv(
            shap_path, 'Mean_Abs_SHAP', spectral_cuts,
        )

    # Permutation
    perm_path = output_dir / f'permutation_{dataset_name}.csv'
    if perm_path.exists():
        rankings['Permutation'] = _zone_ranking_from_per_energy_csv(
            perm_path, 'Permutation_importance', spectral_cuts,
        )

    # PLS-only model-specific explainers
    if model_name == 'pls':
        vip_path = output_dir / f'vip_{dataset_name}.csv'
        if vip_path.exists():
            rankings['VIP'] = _zone_ranking_from_per_energy_csv(
                vip_path, 'VIP_Score', spectral_cuts,
            )

        reg_path = output_dir / f'reg_coef_{dataset_name}.csv'
        if reg_path.exists():
            rankings['Reg_vet'] = _zone_ranking_from_per_energy_csv(
                reg_path, 'Abs_Reg_coef', spectral_cuts,
            )

    # Drop NaN / None entries that may arise from unmapped energies
    for key in rankings:
        rankings[key] = [z for z in rankings[key] if isinstance(z, str)]

    return rankings


def _mask_zones(X, zones_to_mask, spectral_cuts, mask_mode, Xcal_prep):
    """Return a copy of *X* with the columns of *zones_to_mask* replaced.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (preprocessed) to mask.
    zones_to_mask : list of str
        Zone names to mask.
    spectral_cuts : list of tuples
        Zone boundary definitions.
    mask_mode : str
        ``'zero'`` — replace with 0.
        ``'median'`` — replace with per-column median of *Xcal_prep*.
        ``'mean'`` — replace with per-column mean of *Xcal_prep*.
    Xcal_prep : pd.DataFrame
        Calibration (training) data used to compute median/mean statistics.

    Returns
    -------
    pd.DataFrame
        Masked copy of *X*.
    """
    X_masked = X.copy()
    for zone_name in zones_to_mask:
        cols = _get_zone_columns(zone_name, spectral_cuts, X.columns)
        if not cols:
            continue
        if mask_mode == 'zero':
            X_masked[cols] = 0.0
        elif mask_mode == 'median':
            X_masked[cols] = Xcal_prep[cols].median().values
        elif mask_mode == 'mean':
            X_masked[cols] = Xcal_prep[cols].mean().values
        else:
            raise ValueError(f"Unknown mask_mode: '{mask_mode}'")
    return X_masked


def _predict_classes(model_name, model, X):
    """Predict class labels (strings 'A'/'B') for *X*.

    Handles PLS-DA (continuous → binarise at 0.5) and sklearn classifiers
    (SVM, MLP) that return numeric 0/1.
    """
    if model_name == 'pls':
        y_cont = model.predict(X).flatten()
        y_bin = (y_cont >= 0.5).astype(int)
    else:
        y_bin = model.predict(X)
        # Ensure integer array
        y_bin = np.asarray(y_bin, dtype=int)
    return y_bin


def run_faithfulness(dataset, model_name, mask_mode='zero', method='all'):
    """Run faithfulness evaluation for a single (dataset, model) combination.

    For every available XAI method, progressively mask top-k zones and
    measure the degradation of F1 and Accuracy on the prediction set.

    Included methods depend on available files and model type:
    * all models: SMX (LRC perturbation/covariance), SHAP, Permutation
    * PLS only: VIP and reg_vet (from ``vip_*.csv`` and ``reg_coef_*.csv``)

    Parameters
    ----------
    dataset : str
        Dataset identifier.
    model_name : str
        One of ``'pls'``, ``'mlp'``, ``'svm'``.
    mask_mode : str
        Masking strategy: ``'zero'``, ``'median'``, or ``'mean'``.
    method : str | list[str]
        Method selection using canonical names:
        ``'smx_perturbation'``, ``'smx_covariance'``, ``'shap'``,
        ``'permutation'``, or ``'all'``.
    """
    print(f"\n{'#'*70}")
    print(f"# Faithfulness: dataset={dataset}, model={model_name}, mask={mask_mode}")
    print(f"{'#'*70}\n")

    # ── 0. Load config & validate ────────────────────────────────────────
    config = build_effective_config(dataset, model_name)
    dataset_name = config['name']
    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]
    output_dir = SCRIPT_DIR / model_name.upper() / dataset

    if not output_dir.exists():
        print(f"  Output directory does not exist: {output_dir}")
        print("  Run the experiment first.")
        return

    if model_name not in config.get('compatible_models', []):
        print(f"  Model '{model_name}' not compatible with dataset '{dataset}'. Skipping.")
        return

    # ── 1. Collect available zone rankings ───────────────────────────────
    rankings = _collect_zone_rankings(
        output_dir, dataset_name, spectral_cuts,
        model_name=model_name, method=method,
    )
    if not rankings:
        print("  No XAI ranking files found — nothing to evaluate.")
        return
    print(f"  Methods found: {list(rankings.keys())}")

    # ── 2. Reproduce identical model (same seed, data, preprocessing) ────
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("  Loading data...")
    Xcalclass, Xpredclass, ycalclass, ypredclass = load_data(config)

    print("  Preprocessing...")
    Xcalclass_prep, Xpredclass_prep, _ = preprocess(config, Xcalclass, Xpredclass)

    print(f"  Training {model_name}...")
    result = train_model(model_name, config, Xcalclass_prep, ycalclass,
                         Xpredclass_prep, ypredclass)
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)

    # ── 3. Encode true labels as numeric 0/1 ────────────────────────────
    ycal_series = pd.Series(ycalclass).reset_index(drop=True)
    unique_labels = sorted(ycal_series.unique())
    label_to_num = {lab: idx for idx, lab in enumerate(unique_labels)}
    ypred_numeric = np.array([label_to_num[l] for l in ypredclass])

    # ── 4. Baseline (unmasked) performance ──────────────────────────────
    y_hat_orig = _predict_classes(model_name, model, Xpredclass_prep)
    f1_orig = f1_score(ypred_numeric, y_hat_orig, average='weighted', zero_division=0)
    acc_orig = accuracy_score(ypred_numeric, y_hat_orig)

    # Continuous baseline: mean_abs_diff (PLS) or probability_shift (SVM/MLP)
    if model_name == 'pls':
        y_cont_orig = model.predict(Xpredclass_prep).flatten()
    else:
        prob_orig = model.predict_proba(Xpredclass_prep)

    print(f"  Baseline — F1: {f1_orig:.4f}, Accuracy: {acc_orig:.4f}")

    # ── 5. Progressive masking ──────────────────────────────────────────
    rows = []
    for method_label, zone_list in rankings.items():
        n_zones = len(zone_list)
        print(f"\n  [{method_label}] {n_zones} zones: {zone_list}")
        for k in range(1, n_zones + 1):
            top_k_zones = zone_list[:k]
            X_masked = _mask_zones(
                Xpredclass_prep, top_k_zones, spectral_cuts,
                mask_mode, Xcal_prep=Xcalclass_prep,
            )
            y_hat_masked = _predict_classes(model_name, model, X_masked)
            f1_masked = f1_score(ypred_numeric, y_hat_masked,
                                 average='weighted', zero_division=0)
            acc_masked = accuracy_score(ypred_numeric, y_hat_masked)

            # Continuous metric: mean_abs_diff (PLS) or probability_shift (SVM/MLP)
            if model_name == 'pls':
                y_cont_masked = model.predict(X_masked).flatten()
                mean_abs_diff = float(np.mean(np.abs(y_cont_orig - y_cont_masked)))
                cont_metric_name = 'mean_abs_diff'
                cont_metric_val = mean_abs_diff
            else:
                prob_masked = model.predict_proba(X_masked)
                prob_shift = float(np.mean(
                    np.sum(np.abs(prob_orig - prob_masked), axis=1) / 2.0
                ))
                cont_metric_name = 'probability_shift'
                cont_metric_val = prob_shift

            rows.append({
                'model': model_name,
                'method': method_label,
                'k': k,
                'mask_mode': mask_mode,
                'f1_original': round(f1_orig, 6),
                'f1_masked': round(f1_masked, 6),
                'f1_drop': round(f1_orig - f1_masked, 6),
                'acc_original': round(acc_orig, 6),
                'acc_masked': round(acc_masked, 6),
                'acc_drop': round(acc_orig - acc_masked, 6),
                cont_metric_name: round(cont_metric_val, 6),
                'zones_removed': '; '.join(top_k_zones),
            })
            print(f"    k={k:2d}  F1_drop={f1_orig - f1_masked:+.4f}  "
                  f"ACC_drop={acc_orig - acc_masked:+.4f}  "
                  f"{cont_metric_name}={cont_metric_val:.4f}  "
                  f"removed={top_k_zones}")

    # ── 6. Export results ────────────────────────────────────────────────
    df_faith = pd.DataFrame(rows)
    out_path = output_dir / f'faithfulness_{dataset_name}.csv'
    df_faith.to_csv(out_path, index=False, sep=';')
    print(f"\n  Faithfulness results saved to {out_path}")
    return df_faith


# ═══════════════════════════════════════════════════════════════════════════════
# Instability (stability) evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_data_stratified(config, split_seed):
    """Load dataset and split via stratified random sampling.

    Unlike :func:`run_experiment.load_data` which uses Kennard-Stone, this
    function uses ``sklearn.model_selection.train_test_split`` with
    ``stratify`` to preserve class proportions, controlled by *split_seed*.

    Parameters
    ----------
    config : dict
        Dataset JSON configuration.
    split_seed : int
        Random state for the stratified split.

    Returns
    -------
    Xcal, Xpred : pd.DataFrame
        Spectral features for calibration and prediction sets.
    ycal, ypred : pd.Series
        Class labels for calibration and prediction sets.
    """
    class_col = config.get('class_column', 'Class')
    test_size = config.get('test_size', 0.30)

    if config.get('is_synthetic'):
        syn_cfg = config['synthetic_config']
        data_complete = generate_synthetic_spectral_data(
            configuracao_classes=syn_cfg['classes'],
            n_pontos=syn_cfg['n_pontos'],
            x_min=syn_cfg['x_min'],
            x_max=syn_cfg['x_max'],
            seed=syn_cfg['seed'],
        )
        spectral_cols = [c for c in data_complete.columns if c != class_col]
        X_all = data_complete[spectral_cols]
    else:
        csv_file = config['csv_file']
        sep = config.get('separator', ';')
        if config.get('csv_file_is_external'):
            csv_path = WORKSPACE_ROOT / csv_file
        else:
            csv_path = WORKSPACE_ROOT / 'real_datasets' / 'xrf' / csv_file
        data_complete = pd.read_csv(str(csv_path), sep=sep)
        start_col, end_col = config['spectral_range']
        X_all = data_complete.loc[:, start_col:end_col]

    y_all = data_complete[class_col]

    Xcal, Xpred, ycal, ypred = sklearn_train_test_split(
        X_all, y_all,
        test_size=test_size,
        stratify=y_all,
        random_state=split_seed,
    )

    Xcal = Xcal.reset_index(drop=True)
    Xpred = Xpred.reset_index(drop=True)
    ycal = ycal.reset_index(drop=True)
    ypred = ypred.reset_index(drop=True)

    return Xcal, Xpred, ycal, ypred


def run_instability(dataset, model_name, seed_number, smx_seed_number,
                    method='all'):
    """Run instability (stability) evaluation for a (dataset, model) pair.

    For each *split_seed* in ``range(seed_number)``:

    1. Stratified random split (controlled by *split_seed*).
    2. Preprocess and train the model (parameters from JSON config).
    3. Collect full performance metrics (identical to ``run_experiment``).
    4. Run the full SMX pipeline (via the ``SMX`` facade class from the
       ``smx`` library) on the **training** set with **progressively
       growing** subsets of internal bagging seeds.  For
       ``smx_seed_number = M`` the pipeline is executed ``M`` times per
       split per LRC method, once for each cumulative subset:
       ``[0]``, ``[0,1]``, ``[0,1,2]``, … , ``[0,…,M-1]``.
       Each ``SMX.fit()`` call internally performs zone extraction, PCA
       aggregation, predicate generation, bagging, metric computation,
       graph construction, LRC, cross-seed aggregation, and natural-scale
       threshold mapping.  Since the deterministic steps (zone extraction,
       PCA, predicate generation) depend only on the data and cuts, the
       results are identical across calls within the same split.
    5. Collect the aggregated LRC predicate ranking for every combination
       of (split_seed, smx_seeds_used).

    All results are concatenated across seeds and saved as two CSV files:

    * ``instability_smx_{dataset_name}.csv`` — predicate rankings.  Each
      row carries ``split_seed`` (which data split), ``smx_seeds_used``
      (how many internal seeds were aggregated, 1 … M), and
      ``smx_seed_number`` (the configured maximum M).
    * ``instability_performance_{dataset_name}.csv`` — model metrics per
      split seed (one entry per split; independent of SMX internal seeds).

    Parameters
    ----------
    dataset : str
        Dataset identifier.
    model_name : str
        One of ``'pls'``, ``'mlp'``, ``'svm'``.
    seed_number : int
        Number of global split seeds to evaluate (``0 .. seed_number-1``).
    smx_seed_number : int
        Maximum number of internal SMX bagging seeds.  The pipeline runs
        progressively for ``1, 2, …, smx_seed_number`` internal seeds.
    method : str | list[str]
        Instability method(s): ``'smx_perturbation'``, ``'smx_covariance'``,
        ``'shap'``, ``'permutation'``, or ``'all'``.
    """
    print(f"\n{'#'*70}")
    print(f"# Instability: dataset={dataset}, model={model_name}, "
          f"seeds={seed_number}, smx_seeds={smx_seed_number}, method={method}")
    print(f"{'#'*70}\n")

    # ── 0. Load config & validate ────────────────────────────────────────
    # Use the same merged config logic as other analyses so model defaults
    # (e.g., real_datasets/xrf/models/svm.json) are always available.
    config = build_effective_config(dataset, model_name)
    dataset_name = config['name']
    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in config.get('compatible_models', []):
        print(f"  Model '{model_name}' not compatible with dataset '{dataset}'. Skipping.")
        return

    selected_methods = _normalize_instability_methods(method)
    if not selected_methods:
        print("  No valid instability methods selected. Skipping.")
        return

    smx_methods_to_run = []
    if 'smx_perturbation' in selected_methods:
        smx_methods_to_run.append(('smx_perturbation', 'perturbation'))
    if 'smx_covariance' in selected_methods:
        smx_methods_to_run.append(('smx_covariance', 'covariance'))

    all_performance = []
    all_method_outputs = {m: [] for m in selected_methods}

    for split_seed in range(seed_number):
        print(f"\n{'='*70}")
        print(f"  Split seed {split_seed}/{seed_number - 1}")
        print(f"{'='*70}\n")

        # ── 1. Stratified split ──────────────────────────────────────────
        print("  Loading data (stratified split)...")
        Xcal, Xpred, ycal, ypred = _load_data_stratified(config, split_seed)
        print(f"    Xcal: {Xcal.shape}, Xpred: {Xpred.shape}")

        # ── 2. Preprocess ────────────────────────────────────────────────
        print("  Preprocessing...")
        Xcal_prep, Xpred_prep, _ = preprocess(config, Xcal, Xpred)

        # ── 3. Train model (once per split_seed) ────────────────────────
        print(f"  Training {model_name}...")
        result = train_model(model_name, config, Xcal_prep, ycal,
                             Xpred_prep, ypred)

        # ── 4. Collect performance metrics ───────────────────────────────
        df_perf = result[0].copy()
        df_perf.insert(0, 'split_seed', split_seed)
        all_performance.append(df_perf)
        print(f"  Performance (split_seed={split_seed}):")
        print(df_perf.to_string(index=False))

        # ── 5. Explanations on training set (reuse same model per split) ──
        mc = MODEL_CONFIG[model_name]
        model = mc['model_extractor'](result)

        if smx_methods_to_run:
            y_pred_cont = extract_y_continuous(model_name, result)

            # Common SMX kwargs shared across all metric types and seed subsets
            _base_kwargs = dict(
                spectral_cuts=spectral_cuts,
                quantiles=[0.2, 0.4, 0.6, 0.8],
                n_bags=10,
                n_samples_fraction=0.8,
                min_samples_fraction=0.2,
                var_exp=True,
            )

            # Progressive SMX internal seeds: [0], [0,1], [0,1,2], ...
            for method_label, metric_type in smx_methods_to_run:
                for n_smx_seeds in range(1, smx_seed_number + 1):
                    smx_seeds_list = list(range(n_smx_seeds))
                    print(f"\n  --- SMX pipeline ({method_label}) | "
                          f"split_seed={split_seed} | "
                          f"smx_seeds={smx_seeds_list} ---")

                    _iter_kwargs = dict(_base_kwargs, seeds=smx_seeds_list)

                    if metric_type == 'covariance':
                        _iter_kwargs.update(
                            metric='covariance',
                            covariance_threshold=0.01,
                        )
                    else:
                        _iter_kwargs.update(
                            metric='perturbation',
                            estimator=model,
                            perturbation_mode='median',
                            perturbation_metric=mc['perturbation_metric'],
                            normalize_by_zone_size=True,
                            zone_size_exponent=1.0,
                        )

                    try:
                        explainer = SMX(**_iter_kwargs)
                        explainer.fit(Xcal_prep, y_pred_cont, X_cal_natural=Xcal)
                        lrc_out = explainer.lrc_summed_.copy()
                        lrc_out.insert(0, 'split_seed', split_seed)
                        lrc_out['smx_seeds_used'] = n_smx_seeds
                        lrc_out['smx_seed_number'] = smx_seed_number
                        lrc_out['method'] = method_label
                        lrc_out['model'] = model_name
                        all_method_outputs[method_label].append(lrc_out)
                    except RuntimeError as exc:
                        print(f"    WARNING: {exc}")
                        continue

        if 'shap' in selected_methods:
            print(f"\n  --- SHAP explanation | split_seed={split_seed} ---")
            shap_df = _shap_importance_from_model(model_name, model, Xcal_prep)
            shap_out = _format_instability_energy_output(
                shap_df,
                score_col='Mean_Abs_SHAP',
                spectral_cuts=spectral_cuts,
                split_seed=split_seed,
                model_name=model_name,
                method_name='shap',
            )
            all_method_outputs['shap'].append(shap_out)

        if 'permutation' in selected_methods:
            print(f"\n  --- Permutation explanation | split_seed={split_seed} ---")
            perm_df = _permutation_importance_from_model(model_name, model, Xcal_prep)
            perm_out = _format_instability_energy_output(
                perm_df,
                score_col='Permutation_importance',
                spectral_cuts=spectral_cuts,
                split_seed=split_seed,
                model_name=model_name,
                method_name='permutation',
            )
            all_method_outputs['permutation'].append(perm_out)

    # ── 6. Export results ────────────────────────────────────────────────
    if all_performance:
        df_perf_all = pd.concat(all_performance, ignore_index=True)
        perf_path = output_dir / f'instability_performance_{dataset_name}.csv'
        df_perf_all.to_csv(perf_path, index=False, sep=';')
        print(f"\n  Performance metrics saved to {perf_path}")
        print(df_perf_all)
    else:
        print("\n  No performance results collected.")

    for method_name in selected_methods:
        rows = all_method_outputs.get(method_name, [])
        if not rows:
            print(f"\n  No results collected for method '{method_name}'.")
            continue
        df_method_all = pd.concat(rows, ignore_index=True)
        method_path = output_dir / f'instability_{method_name}_{dataset_name}.csv'
        df_method_all.to_csv(method_path, index=False, sep=';')
        print(f"\n  Instability results saved to {method_path}")
        print(df_method_all)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SMX Post-Experiment Analysis (RBO ranking + Faithfulness + Instability)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RBO ranking comparison
  python experiments/run_analysis.py --rbo --dataset soil --model mlp

  # Faithfulness with zero-masking (default)
  python experiments/run_analysis.py --faithfulness --dataset bank_notes --model mlp

  # Faithfulness with median-masking
  python experiments/run_analysis.py --faithfulness --dataset soil --model pls --mask_mode median

  # Both RBO and Faithfulness for all datasets and models
  python experiments/run_analysis.py --rbo --faithfulness --dataset all --model all

  # Instability: 5 global split seeds, 4 internal SMX bagging seeds
  python experiments/run_analysis.py --instability --dataset soil --model mlp --seed_number 5 --smx_seed_number 4

    # Instability: 20 global seeds, selected methods
    python experiments/run_analysis.py --instability --dataset bank_notes --model pls --seed_number 20 --smx_seed_number 4 --method smx_perturbation shap permutation

  # Instability for all datasets and all models
  python experiments/run_analysis.py --instability --dataset all --model all --seed_number 10 --smx_seed_number 3
""",
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--method', nargs='+', default=['all'],
                        choices=['smx_perturbation', 'smx_covariance',
                                 'shap', 'permutation', 'all'],
                        help='Methods to include. Accepts one or more values.')
    parser.add_argument('--rbo', action='store_true',
                        help='Build feature_importance.csv and rbo_rank.csv')
    parser.add_argument('--faithfulness', action='store_true',
                        help='Run faithfulness evaluation (progressive zone masking; '
                            'includes VIP/reg_vet for PLS when available)')
    parser.add_argument('--mask_mode', default='zero',
                        choices=['zero', 'median', 'mean'],
                        help='Masking strategy for faithfulness (default: zero)')
    parser.add_argument('--instability', action='store_true',
                        help='Run instability (stability) evaluation: vary split '
                             'seed, train model, extract SMX rankings per seed')
    parser.add_argument('--seed_number', type=int, default=5,
                        help='Number of global split seeds for --instability '
                             '(seeds 0..N-1, default: 5)')
    parser.add_argument('--smx_seed_number', type=int, default=4,
                        help='Number of internal SMX bagging seeds for '
                             '--instability (seeds 0..M-1, default: 4)')
    args = parser.parse_args()

    if not args.rbo and not args.faithfulness and not args.instability:
        parser.error("At least one of --rbo, --faithfulness, or --instability is required.")

    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                if args.rbo:
                    run_analysis(ds, mdl, method=args.method)
                if args.faithfulness:
                    run_faithfulness(ds, mdl, mask_mode=args.mask_mode,
                                    method=args.method)
                if args.instability:
                    run_instability(ds, mdl,
                                    seed_number=args.seed_number,
                                    smx_seed_number=args.smx_seed_number,
                                    method=args.method)
            except Exception as e:
                print(f"\nERROR running analysis for {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

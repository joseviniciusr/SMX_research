"""
Generic experiment runner for SMX pipeline.

Usage:
    python experiments/run_experiment.py --dataset soil --model pls --method all --debugging
    python experiments/run_experiment.py --dataset all --model all --method covariance
"""

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import kennard_stone as ks

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import preprocessings as prepr
from modeling import pls_optimized, svm_optimized, mlp_optimized
import smx
from explaining import permutation_importance_per_zone
import debugging as dbg
from config import load_dataset_config, list_available_datasets
from synthetic import generate_synthetic_spectral_data

# ── Model dispatch table ─────────────────────────────────────────────────────
MODEL_CONFIG = {
    'pls': {
        'train_fn': pls_optimized,
        'train_kwargs': lambda cfg: {
            'LVmax': cfg['model_params']['pls']['LVmax'],
            'aim': 'classification',
            'cv': cfg['model_params']['pls'].get('cv', 10),
        },
        'y_pred_extractor': lambda result: result[5].iloc[:, -1],
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'regression',
        'perturbation_metric': 'mean_abs_diff',
        'permutation_scoring_fn': None,
        'importance_methods': ['vip', 'reg_coef'],
        'importance_columns': ['VIP_Score', 'Reg_Coefficient'],
    },
    'mlp': {
        'train_fn': mlp_optimized,
        'train_kwargs': lambda cfg: {
            'aim': 'classification',
            **cfg['model_params']['mlp'],
            'random_state': cfg['seed'],
        },
        'y_pred_extractor': lambda result: result[4]['MLP'],
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'classification',
        'perturbation_metric': 'probability_shift',
        'permutation_scoring_fn': lambda model: lambda X: model.predict_proba(X)[:, 1],
        'importance_methods': [],
        'importance_columns': [],
    },
    'svm': {
        'train_fn': svm_optimized,
        'train_kwargs': lambda cfg: {
            'aim': 'classification',
            **cfg['model_params']['svm'],
            'random_state': cfg['seed'],
        },
        'y_pred_extractor': lambda result: result[4]['SVC'],
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'classification',
        'perturbation_metric': 'probability_shift',
        'permutation_scoring_fn': lambda model: lambda X: model.predict_proba(X)[:, 1],
        'importance_methods': ['pvector'],
        'importance_columns': ['SVM_pvector'],
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(config):
    """Load (or generate) dataset and split via Kennard-Stone.

    Returns: Xcalclass, Xpredclass, ycalclass, ypredclass, data_complete
    """
    spectral_range = config['spectral_range']
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
        # For synthetic, spectral data is all columns except 'Class'
        spectral_cols = [c for c in data_complete.columns if c != class_col]
        data = data_complete[spectral_cols]
    else:
        csv_file = config['csv_file']
        sep = config.get('separator', ';')
        if config.get('csv_file_is_external'):
            csv_path = WORKSPACE_ROOT / csv_file
        else:
            csv_path = WORKSPACE_ROOT / 'real_datasets' / 'xrf' / csv_file
        data_complete = pd.read_csv(str(csv_path), sep=sep)
        start_col, end_col = spectral_range
        data = data_complete.loc[:, start_col:end_col]

    data_A = data_complete[data_complete[class_col] == 'A'].reset_index(drop=True)
    data_B = data_complete[data_complete[class_col] == 'B'].reset_index(drop=True)

    if config.get('is_synthetic'):
        spectral_cols_list = [c for c in data_complete.columns if c != class_col]
        XA_all = data_A[spectral_cols_list]
        XB_all = data_B[spectral_cols_list]
    else:
        start_col, end_col = spectral_range
        XA_all = data_A.loc[:, start_col:end_col]
        XB_all = data_B.loc[:, start_col:end_col]

    XA_cal, XA_pred = ks.train_test_split(XA_all, test_size=test_size)
    XA_cal = XA_cal.reset_index(drop=True)
    XA_pred = XA_pred.reset_index(drop=True)

    XB_cal, XB_pred = ks.train_test_split(XB_all, test_size=test_size)
    XB_cal = XB_cal.reset_index(drop=True)
    XB_pred = XB_pred.reset_index(drop=True)

    Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)
    Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
    ycalclass = pd.Series(['A'] * XA_cal.shape[0] + ['B'] * XB_cal.shape[0])
    ypredclass = pd.Series(['A'] * XA_pred.shape[0] + ['B'] * XB_pred.shape[0])

    return Xcalclass, Xpredclass, ycalclass, ypredclass


# ── Preprocessing ────────────────────────────────────────────────────────────

def _apply_single_preprocessing(method, params, Xcal, Xpred):
    """Fit one preprocessing step on *Xcal* and apply the learned transform to *Xpred*.

    Returns (Xcal_new, Xpred_new, info_dict).
    """
    if method == 'poisson':
        mc = params.get('mc', True)
        Xcal_new, mean_original, mean_poisson = prepr.poisson(Xcal, mc=mc)
        Xpred_new = (Xpred / np.sqrt(mean_original)) - mean_poisson
        return Xcal_new, Xpred_new, {'mean_original': mean_original, 'mean_poisson': mean_poisson}

    elif method == 'mc':
        Xcal_new, mean_original = prepr.mc(Xcal)
        Xpred_new = pd.DataFrame(Xpred.values - mean_original.values, columns=Xpred.columns)
        return Xcal_new, Xpred_new, {'mean_original': mean_original}

    elif method == 'savgol':
        from scipy.signal import savgol_filter
        wl = params.get('window_length', 11)
        po = params.get('polyorder', 3)
        deriv = params.get('deriv', 0)
        Xcal_new = pd.DataFrame(
            savgol_filter(Xcal.values, window_length=wl, polyorder=po, deriv=deriv, axis=1),
            columns=Xcal.columns
        )
        Xpred_new = pd.DataFrame(
            savgol_filter(Xpred.values, window_length=wl, polyorder=po, deriv=deriv, axis=1),
            columns=Xpred.columns
        )
        return Xcal_new, Xpred_new, {}

    elif method == 'auto_scaling':
        Xcal_new, sd_original, mean_original = prepr.auto_scaling(Xcal)
        Xpred_new = pd.DataFrame(
            (Xpred.values / sd_original.values) - mean_original.values,
            columns=Xpred.columns
        )
        return Xcal_new, Xpred_new, {'sd_original': sd_original, 'mean_original': mean_original}

    elif method == 'pareto':
        mc = params.get('mc', True)
        Xcal_new, mean_pareto = prepr.pareto(Xcal, mc=mc)
        sd = np.std(Xcal, axis=0)
        escala_pareto = 1 / np.sqrt(sd)
        if mc:
            Xpred_new = pd.DataFrame(
                (Xpred.values * escala_pareto.values) - mean_pareto.values,
                columns=Xpred.columns
            )
        else:
            Xpred_new = pd.DataFrame(Xpred.values * escala_pareto.values, columns=Xpred.columns)
        return Xcal_new, Xpred_new, {'mean_pareto': mean_pareto}

    else:
        raise ValueError(f"Unknown preprocessing method: '{method}'")


def preprocess(config, Xcal, Xpred):
    """Apply preprocessing based on config. Returns (Xcal_prep, Xpred_prep, preprocess_info).

    ``config['preprocessing']`` accepts two forms:

    **Single string** (legacy, fully backward-compatible)::

        "preprocessing": "poisson"          # uses preprocessing_mc flag
        "preprocessing": "savgol"           # uses savgol_params + preprocessing_mc

    **Ordered list of steps** — each step is a method name string *or* a dict with a
    ``"method"`` key plus any per-step parameters::

        "preprocessing": ["savgol", "mc"]
        "preprocessing": [
            {"method": "savgol", "window_length": 11, "polyorder": 3, "deriv": 1},
            "mc"
        ]

    Steps are applied sequentially: the output of each step becomes the input of the next.
    For every step the transform is *fitted on Xcal* and the learned parameters are then
    applied to Xpred.
    """
    preprocessing = config.get('preprocessing', 'poisson')

    if isinstance(preprocessing, str):
        # ── Legacy single-string path (fully backward-compatible) ──────────
        mc = config.get('preprocessing_mc', True)
        if preprocessing == 'savgol':
            savgol_p = dict(config.get('savgol_params', {}))
            # mc becomes an explicit next step so savgol itself does no centering
            steps = [{'method': 'savgol', **savgol_p}]
            if mc:
                steps.append({'method': 'mc'})
        elif preprocessing in ('poisson', 'pareto'):
            steps = [{'method': preprocessing, 'mc': mc}]
        else:
            steps = [{'method': preprocessing}]
    else:
        # ── New list-of-steps path ─────────────────────────────────────────
        steps = []
        for item in preprocessing:
            if isinstance(item, str):
                steps.append({'method': item})
            else:
                steps.append(dict(item))  # copy to avoid mutating config

    # Apply steps sequentially, fitting each on the current Xcal
    Xcal_curr, Xpred_curr = Xcal, Xpred
    combined_info = {}
    for i, step in enumerate(steps):
        method = step['method']
        step_params = {k: v for k, v in step.items() if k != 'method'}
        Xcal_curr, Xpred_curr, info = _apply_single_preprocessing(
            method, step_params, Xcal_curr, Xpred_curr
        )
        combined_info[f'step_{i}_{method}'] = info

    return Xcal_curr, Xpred_curr, combined_info


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model_name, config, Xcal_prep, ycal, Xpred_prep, ypred):
    """Dispatch to pls_optimized / mlp_optimized / svm_optimized."""
    mc = MODEL_CONFIG[model_name]
    train_fn = mc['train_fn']
    kwargs = mc['train_kwargs'](config)
    result = train_fn(Xcal_prep, ycal, Xpred=Xpred_prep, ypred=ypred, **kwargs)
    return result


def extract_y_continuous(model_name, result):
    """Extract continuous predictions from model result tuple."""
    return pd.Series(MODEL_CONFIG[model_name]['y_pred_extractor'](result).values)


# ── LRC Pipelines ────────────────────────────────────────────────────────────

def _run_lrc_pipeline(config, metric_type, zone_scores_df, y_pred, predicates_df,
                      pca_info_dict, Xcalclass, Xcalclass_prep, spectral_cuts,
                      model=None, model_name=None):
    """Run covariance or perturbation LRC pipeline across seeds.

    Returns: (lrc_summed_df, lrc_summed_unique_df, lrc_natural_df,
              spectral_zones_original, pca_info_dict_original)
    """
    random_seeds = config.get('random_seeds', [0, 1, 2, 3])
    training_samples = len(zone_scores_df)
    y_pred_series = pd.Series(y_pred.values) if hasattr(y_pred, 'values') else pd.Series(y_pred)

    # Phase 1: compute metrics per seed
    all_results = {}
    for seed in random_seeds:
        print(f"\n{'='*70}")
        print(f"Processing seed ({metric_type}): {seed}")
        print(f"{'='*70}\n")

        bagger = smx.PredicateBagger(
            n_bags=10,
            n_samples_per_bag=int(training_samples * 0.8),
            min_samples_per_predicate=int(training_samples * 0.2),
            replace=False,
            sample_bagging=True,
            predicate_bagging=False,
            random_seed=seed,
        )
        bags_result_seed = bagger.run(zone_scores_df, y_pred_series, predicates_df)

        for bag_name, pred_dict in bags_result_seed.items():
            for pred_rule, df_info in pred_dict.items():
                df_info['Class_Predicted'] = np.where(
                    df_info['Predicted_Y'] >= 0.5, 'A', 'B'
                )

        if metric_type == 'covariance':
            metric_obj = smx.CovarianceMetric(
                metric='covariance', threshold=0.01, n_neighbors=5
            )
            metric_results = metric_obj.compute(bags_result_seed)
        else:  # perturbation
            mc = MODEL_CONFIG[model_name]
            metric_obj = smx.PerturbationMetric(
                estimator=model,
                Xcalclass_prep=Xcalclass_prep,
                predicates_df=predicates_df,
                spectral_cuts=spectral_cuts,
                perturbation_mode='median',
                stats_source='full',
                metric=mc['perturbation_metric'],
                verbose=True,
                normalize_by_zone_size=True,
                zone_size_exponent=1.0,
            )
            metric_results = metric_obj.compute(bags_result_seed)

        all_results[seed] = {
            'bags_result': bags_result_seed,
            'metric_results': metric_results
        }

    # Phase 2: build graphs per seed
    metric_column = 'Covariance' if metric_type == 'covariance' else 'Perturbation'
    graphs_by_seed = {}
    for seed in random_seeds:
        print(f"\n{'='*70}")
        print(f"Processing graph ({metric_type}) - Seed: {seed}")
        print(f"{'='*70}\n")
        builder = smx.PredicateGraphBuilder(
            random_state=seed,
            show_details=True,
            var_exp=True,
            pca_info_dict=pca_info_dict,
        )
        DG = builder.build(
            bags_result=all_results[seed]['bags_result'],
            predicate_ranking_dict=all_results[seed]['metric_results'],
            metric_column=metric_column,
        )
        graphs_by_seed[seed] = DG

    # Phase 3: LRC per seed
    lrc_by_seed = {}
    for seed in random_seeds:
        DG = graphs_by_seed[seed]
        predicate_nodes = [n for n, attr in DG.nodes(data=True)
                           if attr.get('node_type') == 'predicate']
        if len(predicate_nodes) == 0:
            print(f"  WARNING: seed {seed} produced an empty graph ({metric_type}), skipping.")
            continue
        lrc_df_seed = smx.compute_lrc(DG, predicates_df)
        lrc_df_seed['Seed'] = seed
        lrc_by_seed[seed] = lrc_df_seed

    if not lrc_by_seed:
        raise RuntimeError(
            f"All seeds produced empty graphs for {metric_type}. "
            "The model may be degenerate (e.g., all predictions on one side)."
        )

    # Phase 4: aggregate across seeds
    valid_seeds = list(lrc_by_seed.keys())
    lrc_summed_df, lrc_summed_unique_df = smx.aggregate_lrc_across_seeds(
        lrc_by_seed, valid_seeds
    )
    print(lrc_summed_unique_df)

    # Phase 5: map to natural scale
    spectral_zones_original = smx.extract_spectral_zones(Xcalclass, spectral_cuts)
    agg_original = smx.ZoneAggregator(method='pca')
    zone_scores_natural = agg_original.fit_transform(spectral_zones_original)

    lrc_natural_df = smx.map_thresholds_to_natural(
        lrc_df=lrc_summed_df,
        zone_sums_preprocessed=zone_scores_df,
        zone_sums_natural=zone_scores_natural,
    )
    print(lrc_natural_df)

    pca_info_dict_original = agg_original.pca_info_

    return lrc_summed_df, lrc_summed_unique_df, lrc_natural_df, spectral_zones_original, pca_info_dict_original


# ── Visualization helpers ────────────────────────────────────────────────────

def save_visualization_data(lrc_natural_df, spectral_zones_original,
                            pca_info_dict_original, ycalclass,
                            output_dir, prefix='pert'):
    """Export all data needed to rebuild the threshold-spectrum graph.

    Saved artifacts
    ---------------
    ``viz_{prefix}_lrc_natural.csv``
        LRC results with natural-scale thresholds (already saved elsewhere, but
        duplicated here for self-contained access).
    ``viz_{prefix}_zones/zone_{name}.csv``
        Spectral data for every zone that appears in *lrc_natural_df*, with a
        prepended ``Class`` column drawn from *ycalclass*.
    ``viz_{prefix}_pca_info.json``
        Serialised PCA info (mean, loadings, variance_explained, columns) for
        every zone in *pca_info_dict_original*.
    """
    import json

    viz_dir = output_dir / f'viz_{prefix}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    zones_dir = viz_dir / 'zones'
    zones_dir.mkdir(exist_ok=True)

    # 1. LRC table
    lrc_natural_df.to_csv(viz_dir / 'lrc_natural.csv', index=False, sep=';')

    # 2. Spectral data per zone (with class labels + threshold rows)
    zones_in_lrc = lrc_natural_df['Zone'].dropna().unique().tolist()
    for zone_name in zones_in_lrc:
        if zone_name not in spectral_zones_original:
            continue
        zone_df = spectral_zones_original[zone_name].copy()
        zone_df.insert(0, 'Class', ycalclass.values)

        # Append one Multivariate_threshold row per predicate for this zone
        zone_rows = lrc_natural_df[lrc_natural_df['Zone'] == zone_name]
        if zone_name in pca_info_dict_original:
            for _, lrc_row in zone_rows.iterrows():
                thresh_val = float(lrc_row['Threshold_Natural'])
                node_natural = lrc_row.get('Node_Natural', lrc_row.get('Node', ''))
                thresh_spectrum = smx.reconstruct_threshold_to_spectrum(
                    threshold_value=thresh_val,
                    zone_name=zone_name,
                    pca_info_dict=pca_info_dict_original,
                )
                thresh_row = {'Class': f'Multivariate_threshold|{node_natural}'}
                thresh_row.update({str(c): v for c, v in
                                   zip(thresh_spectrum.index, thresh_spectrum.values)})
                zone_df = pd.concat(
                    [zone_df, pd.DataFrame([thresh_row])], ignore_index=True
                )

        safe_name = zone_name.replace(' ', '_').replace('/', '_').replace('+', 'plus')
        zone_df.to_csv(zones_dir / f'zone_{safe_name}.csv', index=False, sep=';')

    # 3. PCA info (serialisable subset)
    pca_export = {}
    for zone_name, info in pca_info_dict_original.items():
        pca_export[zone_name] = {
            'mean': info['mean'].tolist(),
            'loadings': info['loadings'].tolist(),
            'variance_explained': float(info['variance_explained']),
            'columns': [str(c) for c in info['columns']],
        }
    with open(viz_dir / 'pca_info.json', 'w') as fh:
        json.dump(pca_export, fh, indent=2)

    print(f"  Visualization data exported to {viz_dir}")


def build_and_export_figure(lrc_natural_df, spectral_zones_original,
                            pca_info_dict_original, ycalclass,
                            output_dir, prefix='pert'):
    """Generate and save one HTML figure per row of *lrc_natural_df*.

    Each figure shows the spectral zone coloured by class (A=gold, B=blue) with
    the reconstructed multivariate threshold highlighted in red.
    """
    import plotly.graph_objects as go

    fig_dir = output_dir / f'figures_{prefix}'
    fig_dir.mkdir(parents=True, exist_ok=True)

    CLASS_COLORS = {'A': 'gold', 'B': 'blue'}

    for n, row in lrc_natural_df.iterrows():
        zone_name = row.get('Zone')
        if zone_name not in spectral_zones_original or zone_name not in pca_info_dict_original:
            continue

        threshold_score = float(row['Threshold_Natural'])
        node_natural = row.get('Node_Natural', row.get('Node', ''))

        # Reconstruct threshold spectrum
        threshold_spectrum = smx.reconstruct_threshold_to_spectrum(
            threshold_value=threshold_score,
            zone_name=zone_name,
            pca_info_dict=pca_info_dict_original
        )

        zone_df = spectral_zones_original[zone_name]
        x_values = pd.to_numeric(zone_df.columns, errors='coerce')

        fig = go.Figure()

        # Sample spectra coloured by class
        seen_classes = set()
        for idx, spec_row in zone_df.iterrows():
            class_label = ycalclass.iloc[idx] if idx < len(ycalclass) else 'Unknown'
            show_leg = class_label not in seen_classes
            seen_classes.add(class_label)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=spec_row.values,
                mode='lines',
                line=dict(color=CLASS_COLORS.get(class_label, 'gray'), width=0.5),
                name=f'Class {class_label}',
                legendgroup=class_label,
                showlegend=show_leg,
                hoverinfo='skip',
            ))

        # Threshold spectrum
        fig.add_trace(go.Scatter(
            x=x_values,
            y=threshold_spectrum.values,
            mode='lines',
            line=dict(color='red', width=4, dash='dash'),
            name=f'Threshold Spectrum ({threshold_spectrum.name})',
        ))

        fig.update_layout(
            title=f"Zone '{zone_name}' — Multivariate Threshold "
                  f"(Predicate: {node_natural})",
            xaxis_title='Energy / Wavelength',
            yaxis_title='Intensity',
            template='plotly_white',
            showlegend=True,
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        )

        safe_name = zone_name.replace(' ', '_').replace('/', '_').replace('+', 'plus')
        out_path = fig_dir / f'threshold_{safe_name}_rank{n}.html'
        fig.write_html(str(out_path))
        print(f"  Figure saved: {out_path}")


def load_visualization_data(output_dir, prefix):
    """Load visualization data previously saved by save_visualization_data.

    Returns
    -------
    lrc_natural_df : pd.DataFrame
    spectral_zones : dict[str, pd.DataFrame]   spectral data only (no Class column,
                                                no threshold rows)
    pca_info_dict  : dict                       with numpy arrays restored
    ycalclass      : pd.Series                  class labels aligned to zone rows
    """
    import json

    viz_dir = output_dir / f'viz_{prefix}'
    if not viz_dir.exists():
        raise FileNotFoundError(
            f"Visualization data directory not found: {viz_dir}\n"
            "Run without --visualization-only first to generate the data."
        )

    # 1. LRC table
    lrc_natural_df = pd.read_csv(viz_dir / 'lrc_natural.csv', sep=';')

    # 2. PCA info (restore numpy arrays)
    with open(viz_dir / 'pca_info.json') as fh:
        pca_raw = json.load(fh)
    pca_info_dict = {
        zone_name: {
            'mean': np.array(info['mean']),
            'loadings': np.array(info['loadings']),
            'variance_explained': info['variance_explained'],
            'columns': info['columns'],
        }
        for zone_name, info in pca_raw.items()
    }

    # 3. Zone CSVs → spectral_zones + ycalclass
    zones_dir = viz_dir / 'zones'
    zones_in_lrc = lrc_natural_df['Zone'].dropna().unique().tolist()
    safe_to_zone = {
        z.replace(' ', '_').replace('/', '_').replace('+', 'plus'): z
        for z in zones_in_lrc
    }

    spectral_zones = {}
    ycalclass = None
    for csv_path in sorted(zones_dir.glob('zone_*.csv')):
        safe_name = csv_path.stem[len('zone_'):]
        zone_name = safe_to_zone.get(safe_name)
        if zone_name is None:
            continue
        df = pd.read_csv(csv_path, sep=';')
        # Filter out threshold rows, keep only real samples
        mask_threshold = df['Class'].astype(str).str.startswith('Multivariate_threshold')
        samples_df = df[~mask_threshold].reset_index(drop=True)
        if ycalclass is None:
            ycalclass = samples_df['Class'].reset_index(drop=True)
        spectral_zones[zone_name] = samples_df.drop(columns=['Class'])

    if ycalclass is None:
        raise RuntimeError(
            f"No zone CSV files found under {zones_dir} for prefix '{prefix}'."
        )

    print(f"  Loaded visualization data from {viz_dir} "
          f"({len(spectral_zones)} zones, {len(lrc_natural_df)} LRC rows)")
    return lrc_natural_df, spectral_zones, pca_info_dict, ycalclass


def run_visualization_only(dataset, model_name, method):
    """Load pre-computed visualization data from disk and export figures.

    No SMX computation is performed; all inputs come from the ``viz_{prefix}/``
    directories previously written by save_visualization_data.
    """
    print(f"\n{'#'*70}")
    print(f"# Visualization-only: dataset={dataset}, model={model_name}, method={method}")
    print(f"{'#'*70}\n")

    output_dir = SCRIPT_DIR / model_name.upper() / dataset

    prefixes = []
    if method in ('covariance', 'all'):
        prefixes.append('cov')
    if method in ('perturbation', 'all'):
        prefixes.append('pert')

    for prefix in prefixes:
        print(f"\n--- Loading and rendering figures ({prefix}) ---")
        lrc_natural_df, spectral_zones, pca_info_dict, ycalclass = \
            load_visualization_data(output_dir, prefix)
        build_and_export_figure(
            lrc_natural_df, spectral_zones, pca_info_dict,
            ycalclass, output_dir, prefix=prefix,
        )

    print(f"\nDone – figures saved under {output_dir}")


# ── Debugging helpers ────────────────────────────────────────────────────────

def run_debugging(model_name, result, config, Xcalclass_prep, spectral_cuts, output_dir):
    """Run model-specific importance methods + SHAP zone + permutation importance.

    Returns dict with zone-deduplicated DataFrames for each method.
    """
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)
    results = {}

    # Model-specific importance
    if 'vip' in mc['importance_methods']:
        vip_scores_mat = result[4]  # PLS vip
        results['vip'] = dbg.vip_scores_per_zone(vip_scores_mat, spectral_cuts)
        print("VIP scores per zone:")
        print(results['vip'])

    if 'reg_coef' in mc['importance_methods']:
        results['reg_coef'] = dbg.regression_coefficients_per_zone(model, spectral_cuts)
        print("Regression coefficients per zone:")
        print(results['reg_coef'])

    if 'pvector' in mc['importance_methods']:
        results['pvector'] = dbg.svm_pvector_per_zone(
            model, Xcalclass_prep.columns, spectral_cuts
        )
        print("SVM P-vector per zone:")
        print(results['pvector'])

    # SHAP from pre-computed CSV
    dataset_name = config['name']
    shap_csv = output_dir / f'shap_{dataset_name}.csv'
    if shap_csv.exists():
        results['shap'] = dbg.shap_per_zone(str(shap_csv), spectral_cuts)
    else:
        warnings.warn(f"SHAP CSV not found at {shap_csv}, skipping SHAP.")

    # Permutation importance
    scoring_fn = mc['permutation_scoring_fn']
    if scoring_fn is not None:
        scoring_fn = scoring_fn(model)

    perm_unique, perm_full = permutation_importance_per_zone(
        estimator=model,
        X=Xcalclass_prep,
        spectral_cuts=spectral_cuts,
        n_repeats=10,
        random_state=42,
        scoring_fn=scoring_fn,
    )
    results['permutation'] = perm_unique
    print("Permutation importance per zone:")
    print(perm_unique)

    # exporting the performance metrics of the models
    dataset_name = config['name']
    metrics_csv = output_dir / 'performance_metrics.csv'
    dbg.export_performance_metrics(result[0], str(metrics_csv))
    results['performance_metrics'] = result[0]

    return results


def build_feature_importance_table(model_name, debugging_results,
                                   lrc_cov_unique=None, lrc_pert_unique=None):
    """Assemble feature_importance DataFrame with model-appropriate columns."""
    mc = MODEL_CONFIG[model_name]

    # Collect all zone lists
    zone_lists = {}

    if lrc_cov_unique is not None:
        zone_lists['LRC_covariance'] = list(lrc_cov_unique['Zone'])
    if lrc_pert_unique is not None:
        zone_lists['LRC_perturbation'] = list(lrc_pert_unique['Zone'])

    if 'permutation' in debugging_results:
        zone_lists['Permutation'] = list(debugging_results['permutation']['Zone'])

    if 'vip' in debugging_results:
        zone_lists['VIP_Score'] = list(debugging_results['vip']['Zone'])
    if 'reg_coef' in debugging_results:
        zone_lists['Reg_Coefficient'] = list(debugging_results['reg_coef']['Zone'])
    if 'pvector' in debugging_results:
        zone_lists['SVM_pvector'] = list(debugging_results['pvector']['Zone'])
    if 'shap' in debugging_results:
        zone_lists['Shap'] = list(debugging_results['shap']['Zone'])

    if not zone_lists:
        return pd.DataFrame()

    max_len = max(len(v) for v in zone_lists.values())

    def pad(lst):
        return lst + [None] * (max_len - len(lst))

    # Canonical column order per model type (model-specific first, then common)
    COLUMN_ORDER = {
        'pls': ['VIP_Score', 'Reg_Coefficient', 'Shap', 'Permutation',
                 'LRC_perturbation', 'LRC_covariance'],
        'svm': ['SVM_pvector', 'Shap', 'Permutation',
                 'LRC_perturbation', 'LRC_covariance'],
        'mlp': ['Shap', 'Permutation', 'LRC_perturbation', 'LRC_covariance'],
    }
    ordered_cols = [c for c in COLUMN_ORDER.get(model_name, [])
                    if c in zone_lists]

    fi_data = {k: pad(v) for k, v in zone_lists.items()}
    df = pd.DataFrame(fi_data)
    return df[ordered_cols] if ordered_cols else df


# ── Main experiment orchestrator ─────────────────────────────────────────────

def run_single_experiment(dataset, model_name, method, debugging, visualization=False):
    """Orchestrate a single (dataset, model) experiment."""
    print(f"\n{'#'*70}")
    print(f"# Experiment: dataset={dataset}, model={model_name}, method={method}")
    print(f"{'#'*70}\n")

    # 0. Load config
    config = load_dataset_config(dataset)

    # 1. Read seed from config (required)
    if 'seed' not in config:
        raise ValueError(
            f"Dataset config for '{dataset}' is missing the required 'seed' field."
        )
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]

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

    # 5. Train model
    print(f"Training {model_name}...")
    result = train_model(model_name, config, Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass)

    # 6. Extract predictions
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)
    y_pred_cont = extract_y_continuous(model_name, result)

    # 7. PCA aggregation
    print("PCA aggregation...")
    spectral_zones_class = smx.extract_spectral_zones(Xcalclass_prep, spectral_cuts)
    aggregator = smx.ZoneAggregator(method='pca')
    zone_scores_df = aggregator.fit_transform(spectral_zones_class)
    pca_info_dict = aggregator.pca_info_
    print(f"  Zone scores shape: {zone_scores_df.shape}")

    # 8. Predicates
    pred_gen = smx.PredicateGenerator(quantiles=[0.2, 0.4, 0.6, 0.8])
    pred_gen.fit(zone_scores_df)
    predicates_df = pred_gen.predicates_df_

    # 9. Output directory
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # 10. Run LRC pipelines
    lrc_cov_unique = None
    lrc_pert_unique = None
    lrc_cov_natural = None
    lrc_pert_natural = None

    if method in ('covariance', 'all'):
        print("\n--- Covariance Pipeline ---")
        _, lrc_cov_unique, lrc_cov_natural, cov_zones_orig, cov_pca_orig = _run_lrc_pipeline(
            config, 'covariance', zone_scores_df, y_pred_cont,
            predicates_df, pca_info_dict,
            Xcalclass, Xcalclass_prep, spectral_cuts,
            model=model, model_name=model_name
        )
        save_visualization_data(
            lrc_cov_natural, cov_zones_orig, cov_pca_orig,
            ycalclass, output_dir, prefix='cov'
        )
        if visualization:
            build_and_export_figure(
                lrc_cov_natural, cov_zones_orig, cov_pca_orig,
                ycalclass, output_dir, prefix='cov'
            )

    if method in ('perturbation', 'all'):
        print("\n--- Perturbation Pipeline ---")
        _, lrc_pert_unique, lrc_pert_natural, pert_zones_orig, pert_pca_orig = _run_lrc_pipeline(
            config, 'perturbation', zone_scores_df, y_pred_cont,
            predicates_df, pca_info_dict,
            Xcalclass, Xcalclass_prep, spectral_cuts,
            model=model, model_name=model_name
        )
        save_visualization_data(
            lrc_pert_natural, pert_zones_orig, pert_pca_orig,
            ycalclass, output_dir, prefix='pert'
        )
        if visualization:
            build_and_export_figure(
                lrc_pert_natural, pert_zones_orig, pert_pca_orig,
                ycalclass, output_dir, prefix='pert'
            )

    # 11. Debugging extras
    debugging_results = {}
    if debugging:
        print("\n--- Debugging ---")
        debugging_results = run_debugging(
            model_name, result, config, Xcalclass_prep, spectral_cuts, output_dir
        )

    # 12. Feature importance table
    if debugging:
        features_importance = build_feature_importance_table(
            model_name, debugging_results,
            lrc_cov_unique=lrc_cov_unique,
            lrc_pert_unique=lrc_pert_unique
        )
        if not features_importance.empty:
            features_importance.to_csv(
                output_dir / 'feature_importance.csv', index=False, sep=';'
            )
            print("\nFeature importance table:")
            print(features_importance)

            # RBO
            dbg.rbo_rank_comparison(features_importance, output_dir / 'rbo_rank.csv')

    # 13. Save LRC artifacts
    if lrc_cov_natural is not None:
        lrc_cov_natural.to_csv(output_dir / 'lrc_cov_natural.csv', index=False, sep=';')
    if lrc_pert_natural is not None:
        lrc_pert_natural.to_csv(output_dir / 'lrc_pert_natural.csv', index=False, sep=';')

    print(f"\nDone – artifacts saved to {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SMX Experiment Runner')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--method', default='all',
                        choices=['covariance', 'perturbation', 'all'],
                        help='LRC method(s) to run')
    parser.add_argument('--debugging', action='store_true',
                        help='Enable non-core extras (permutation, model importance, SHAP, RBO)')
    parser.add_argument('--visualization', action='store_true',
                        help='Export threshold-spectrum HTML figures (one per LRC row)')
    parser.add_argument('--visualization-only', action='store_true',
                        help='Load pre-computed viz data from file and export figures; '
                             'skips all SMX computation')
    args = parser.parse_args()

    # Expand 'all'
    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                if args.visualization_only:
                    run_visualization_only(ds, mdl, args.method)
                else:
                    run_single_experiment(ds, mdl, args.method, args.debugging,
                                          visualization=args.visualization)
            except Exception as e:
                print(f"\nERROR running {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

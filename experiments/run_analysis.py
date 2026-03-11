"""
Post-experiment analysis: build feature_importance.csv and rbo_rank.csv from
the individual technique output files produced by run_experiment, run_shap,
and run_permutation.

Only techniques whose output files exist are included – missing techniques
are silently skipped so you can run analysis at any stage.

Usage:
    python experiments/run_analysis.py --dataset soil --model mlp
    python experiments/run_analysis.py --dataset all --model all --method all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import load_dataset_config, list_available_datasets
import debugging as dbg


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
    method : str
        Which LRC methods to include: 'covariance', 'perturbation', or 'all'.

    Returns
    -------
    pd.DataFrame
        Feature importance table with one column per available technique.
    """
    zone_lists = {}

    # LRC results
    if method in ('covariance', 'all'):
        cov_path = output_dir / 'lrc_cov_natural.csv'
        if cov_path.exists():
            zone_lists['LRC_covariance'] = _zone_ranking_from_lrc(cov_path)

    if method in ('perturbation', 'all'):
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
    """Run analysis for a single (dataset, model) combination."""
    print(f"\n{'#'*70}")
    print(f"# Analysis: dataset={dataset}, model={model_name}, method={method}")
    print(f"{'#'*70}\n")

    config = load_dataset_config(dataset)
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SMX Post-Experiment Analysis (feature importance + RBO)'
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--method', default='all',
                        choices=['covariance', 'perturbation', 'all'],
                        help='Which LRC method(s) to include in the comparison')
    args = parser.parse_args()

    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                run_analysis(ds, mdl, method=args.method)
            except Exception as e:
                print(f"\nERROR running analysis for {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

# alpha_by_regime.py | Alpha Performance by Regime
# ------------------------------------------------------------------------------
# Analyzes backtest performance by regime using EWMA-based regime shift detection.
# Supports phase labeling (stable / elevated / crisis onset / resolution) or
# legacy transition vs stable. Focuses on:
# - Fraction of total returns contributed by each regime (cumulative return contribution)
# - Sharpe ratio within each regime
# - Sample size (months) for reliability assessment
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from typing import Optional, Dict, List
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

PHASE_REGIMES = ['stable', 'elevated', 'crisis_onset', 'resolution']
LEGACY_REGIMES = ['transition', 'stable']

REGIME_DISPLAY_NAMES = {
    'stable': 'Stable',
    'elevated': 'Elevated',
    'crisis_onset': 'Crisis Onset',
    'resolution': 'Resolution',
    'transition': 'Transition',
}

REGIME_COLORS = {
    'stable': '#5B9BD5',
    'elevated': '#FFC000',
    'crisis_onset': '#C00000',
    'resolution': '#70AD47',
    'transition': '#C00000',
}


def get_regime_order(method: str, regime_labels: pd.Series) -> List[str]:
    """Return regime labels in display order, filtered to those present."""
    order = PHASE_REGIMES if method == 'phase' else LEGACY_REGIMES
    present = set(regime_labels.dropna().unique())
    return [r for r in order if r in present]

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config() -> dict:
    """Read parameters from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

cfg = load_config()
CACHE_DIR = Path(cfg["paths"]["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------#
# 1  Load Data Functions
# -----------------------------------------------------------------------------#
def load_ewma_regime_shifts(use_cache: bool = True) -> pd.DataFrame:
    """
    Load EWMA regime shifts from cache or calculate if not available.
    
    Parameters
    ----------
    use_cache : bool, default True
        Whether to use cached EWMA results
        
    Returns
    -------
    pd.DataFrame
        EWMA regime shifts DataFrame with columns ['1-year', '2-year', '3-year', '4-year', 'mean']
    """
    cache_file = CACHE_DIR / "ewma_regime_shifts.pkl"
    
    if use_cache and cache_file.exists():
        print(f"Loading EWMA regime shifts from cache: {cache_file}")
        return joblib.load(cache_file)
    else:
        print("EWMA regime shifts not found in cache. Calculating...")
        try:
            from .regime_shift import run_regime_shift_analysis
        except ImportError:
            from regime_shift import run_regime_shift_analysis
        return run_regime_shift_analysis(use_cache=use_cache, create_visualization=False)


def load_backtest_returns(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1
) -> pd.DataFrame:
    """
    Load or calculate backtest returns.
    
    Parameters
    ----------
    n_buckets : int, default 5
        Number of buckets/quintiles
    back_test_start_date : str, default "1985-01-31"
        Start date for backtest
    forward_look_months : int, default 1
        Forward look months for returns
    similarity_window : int, default 1
        Similarity window size
        
    Returns
    -------
    pd.DataFrame
        Backtest returns DataFrame
    """
    try:
        from ..backtest.back_test import run_backtest
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backtest.back_test import run_backtest
    return run_backtest(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window,
        show_alignment_message=False
    )


# -----------------------------------------------------------------------------#
# 2  Regime Labeling Functions
# -----------------------------------------------------------------------------#
def label_regimes(
    ewma_df: pd.DataFrame,
    method: str = "percentile",
    threshold_percentile: Optional[float] = None,
    threshold_absolute: Optional[float] = None,
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None
) -> pd.Series:
    """
    Label each month by regime based on EWMA values and (for phase method) direction.
    
    Parameters
    ----------
    ewma_df : pd.DataFrame
        EWMA regime shifts DataFrame with 'mean' column
    method : str, default "percentile"
        Method for labeling: "phase", "percentile", or "absolute"
    threshold_percentile : float, optional
        Percentile threshold (0-1) used when method="percentile"
    threshold_absolute : float, optional
        Absolute EWMA value threshold used when method="absolute"
    low_threshold_percentile : float, optional
        Lower percentile (0-1) for phase method (stable vs elevated)
    high_threshold_percentile : float, optional
        Upper percentile (0-1) for phase method (elevated vs crisis)
        
    Returns
    -------
    pd.Series
        Regime labels indexed by date
    """
    mean_ewma = ewma_df['mean'].dropna()
    
    if method == "phase":
        if low_threshold_percentile is None or high_threshold_percentile is None:
            raise ValueError(
                "low_threshold_percentile and high_threshold_percentile required when method='phase'"
            )
        for pct, name in [
            (low_threshold_percentile, "low_threshold_percentile"),
            (high_threshold_percentile, "high_threshold_percentile"),
        ]:
            if not (0 <= pct <= 1):
                raise ValueError(f"{name} must be between 0 and 1 (got {pct})")
        if low_threshold_percentile >= high_threshold_percentile:
            raise ValueError("low_threshold_percentile must be less than high_threshold_percentile")

        low_threshold = mean_ewma.quantile(low_threshold_percentile)
        high_threshold = mean_ewma.quantile(high_threshold_percentile)
        delta = mean_ewma.diff()

        labels = pd.Series('stable', index=mean_ewma.index, name='regime')
        labels[(mean_ewma > high_threshold) & (delta > 0)] = 'crisis_onset'
        labels[(mean_ewma > high_threshold) & (delta <= 0)] = 'resolution'
        labels[(mean_ewma > low_threshold) & (mean_ewma <= high_threshold)] = 'elevated'

        print(
            f"Regime labeling complete (method: {method}, "
            f"low: {low_threshold_percentile:.2f} -> {low_threshold:.4f}, "
            f"high: {high_threshold_percentile:.2f} -> {high_threshold:.4f})"
        )
        for regime in PHASE_REGIMES:
            count = (labels == regime).sum()
            print(f"  {REGIME_DISPLAY_NAMES[regime]} months: {count} ({count / len(labels) * 100:.1f}%)")

    elif method == "percentile":
        if threshold_percentile is None:
            raise ValueError("threshold_percentile parameter required when method='percentile'")
        if not (0 <= threshold_percentile <= 1):
            raise ValueError(f"Percentile threshold must be between 0 and 1 (got {threshold_percentile})")
        threshold_value = mean_ewma.quantile(threshold_percentile)
        print(f"Regime labeling complete (method: {method}, percentile: {threshold_percentile:.3f}, threshold value: {threshold_value:.4f})")
        labels = pd.Series(
            index=mean_ewma.index,
            data=['transition' if val > threshold_value else 'stable' for val in mean_ewma.values],
            name='regime'
        )
        print(f"  Transition months: {(labels == 'transition').sum()} ({(labels == 'transition').mean()*100:.1f}%)")
        print(f"  Stable months: {(labels == 'stable').sum()} ({(labels == 'stable').mean()*100:.1f}%)")

    elif method == "absolute":
        if threshold_absolute is None:
            raise ValueError("threshold_absolute parameter required when method='absolute'")
        threshold_value = threshold_absolute
        print(f"Regime labeling complete (method: {method}, threshold: {threshold_value:.4f})")
        labels = pd.Series(
            index=mean_ewma.index,
            data=['transition' if val > threshold_value else 'stable' for val in mean_ewma.values],
            name='regime'
        )
        print(f"  Transition months: {(labels == 'transition').sum()} ({(labels == 'transition').mean()*100:.1f}%)")
        print(f"  Stable months: {(labels == 'stable').sum()} ({(labels == 'stable').mean()*100:.1f}%)")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'phase', 'percentile', or 'absolute'")
    
    return labels


# -----------------------------------------------------------------------------#
# 3  Performance Metrics by Regime
# -----------------------------------------------------------------------------#
def calculate_regime_metrics(
    returns: pd.Series, 
    regime_labels: pd.Series, 
    regime_type: str
) -> Dict[str, float]:
    """
    Calculate Sharpe ratio and sample size for a given regime subset.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    regime_labels : pd.Series
        Regime labels ('transition' or 'stable')
    regime_type : str
        Which regime to filter for: 'transition' or 'stable'
        
    Returns
    -------
    dict
        Dictionary with Sharpe ratio and sample size
    """
    # Align indices
    aligned_data = pd.DataFrame({
        'returns': returns,
        'regime': regime_labels
    }).dropna()
    
    if len(aligned_data) == 0:
        return {
            'ann_sharpe': np.nan,
            'n_months': 0
        }
    
    # Filter for the specified regime
    regime_data = aligned_data[aligned_data['regime'] == regime_type]
    
    if len(regime_data) == 0:
        return {
            'ann_sharpe': np.nan,
            'n_months': 0
        }
    
    regime_returns = regime_data['returns']
    all_returns = aligned_data['returns']
    
    # Annualized Sharpe: regime mean return normalized by full-sample vol
    if all_returns.std() > 0:
        ann_sharpe = regime_returns.mean() / all_returns.std() * np.sqrt(12)
    else:
        ann_sharpe = np.nan
    
    return {
        'ann_sharpe': ann_sharpe,
        'n_months': len(regime_returns)
    }


def calculate_cumulative_return_contribution(
    returns: pd.Series,
    regime_labels: pd.Series,
    regime_types: List[str]
) -> Dict[str, float]:
    """
    Calculate cumulative return contribution by regime.
    Uses log returns for additive decomposition of cumulative returns.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    regime_labels : pd.Series
        Regime labels
    regime_types : list of str
        Regime names to compute contributions for
        
    Returns
    -------
    dict
        Per-regime contribution, cumulative return, and total cumulative return
    """
    aligned_data = pd.DataFrame({
        'returns': returns,
        'regime': regime_labels
    }).dropna()
    
    result: Dict[str, float] = {'total_cumulative_return': np.nan}
    for regime in regime_types:
        result[f'{regime}_contribution'] = np.nan
        result[f'{regime}_cumulative_return'] = np.nan

    if len(aligned_data) == 0:
        return result

    aligned_data['log_return'] = np.log(1 + aligned_data['returns'])
    total_cumret = (1 + aligned_data['returns']).prod() - 1
    total_log_return = aligned_data['log_return'].sum()
    total_sum = aligned_data['returns'].sum()

    result['total_cumulative_return'] = total_cumret

    for regime in regime_types:
        regime_data = aligned_data[aligned_data['regime'] == regime]
        if len(regime_data) == 0:
            continue

        regime_returns = regime_data['returns']
        regime_log_returns = regime_data['log_return']
        result[f'{regime}_cumulative_return'] = (1 + regime_returns).prod() - 1

        if abs(total_log_return) > 1e-10:
            result[f'{regime}_contribution'] = regime_log_returns.sum() / total_log_return
        elif abs(total_sum) > 1e-10:
            result[f'{regime}_contribution'] = regime_returns.sum() / total_sum
        else:
            result[f'{regime}_contribution'] = 0.0

    return result


def compute_regime_performance(
    backtest_returns: pd.DataFrame,
    regime_labels: pd.Series,
    regime_types: List[str]
) -> pd.DataFrame:
    """
    Compute performance metrics separately for each regime.
    
    Parameters
    ----------
    backtest_returns : pd.DataFrame
        Backtest returns for all strategies
    regime_labels : pd.Series
        Regime labels for each month
    regime_types : list of str
        Regime names to compute metrics for
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each strategy
    """
    results = []
    
    for strategy in backtest_returns.columns:
        strategy_returns = backtest_returns[strategy]
        result = {'strategy': strategy}

        for regime in regime_types:
            metrics = calculate_regime_metrics(strategy_returns, regime_labels, regime_type=regime)
            result[f'{regime}_sharpe'] = metrics['ann_sharpe']
            result[f'{regime}_n_months'] = metrics['n_months']

        contribution_metrics = calculate_cumulative_return_contribution(
            strategy_returns, regime_labels, regime_types
        )
        for regime in regime_types:
            result[f'{regime}_return_contribution'] = contribution_metrics[f'{regime}_contribution']
            result[f'{regime}_cumulative_return'] = contribution_metrics[f'{regime}_cumulative_return']
        result['total_cumulative_return'] = contribution_metrics['total_cumulative_return']

        results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df.set_index('strategy')


# -----------------------------------------------------------------------------#
# 4  Summary Table Generation
# -----------------------------------------------------------------------------#
def create_summary_table(
    regime_performance: pd.DataFrame,
    regime_types: List[str]
) -> pd.DataFrame:
    """
    Create a formatted summary table comparing performance across regimes.
    
    Parameters
    ----------
    regime_performance : pd.DataFrame
        Output from compute_regime_performance
    regime_types : list of str
        Regime names in display order
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    column_order = []
    display_names = []
    for regime in regime_types:
        display = REGIME_DISPLAY_NAMES.get(regime, regime.title())
        column_order.extend([
            f'{regime}_sharpe',
            f'{regime}_return_contribution',
            f'{regime}_cumulative_return',
        ])
        display_names.extend([
            f'{display}_Sharpe',
            f'{display}_Return_Fraction',
            f'{display}_CumReturn',
        ])
    column_order.append('total_cumulative_return')
    display_names.append('Total_CumReturn')

    available_cols = [col for col in column_order if col in regime_performance.columns]
    summary_df = regime_performance[available_cols].copy()
    summary_df.columns = display_names[:len(available_cols)]
    return summary_df


# -----------------------------------------------------------------------------#
# 5  Visualization Functions
# -----------------------------------------------------------------------------#
def _regime_metric_frame(
    summary_table: pd.DataFrame,
    regime_types: List[str],
    metric_suffix: str,
) -> pd.DataFrame:
    """Extract a strategy x regime DataFrame for Sharpe or Return_Fraction columns."""
    columns = {}
    for regime in regime_types:
        display = REGIME_DISPLAY_NAMES.get(regime, regime.title())
        col_name = f'{display}_{metric_suffix}'
        if col_name in summary_table.columns:
            columns[display] = summary_table[col_name]
    return pd.DataFrame(columns)


def create_alpha_by_regime_exhibit(
    summary_table: pd.DataFrame,
    regime_types: List[str],
    regime_method: str,
    regime_labels: Optional[pd.Series] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Create a two-panel exhibit: Sharpe heatmap and return-contribution stacked bars.
    """
    if save_path is None:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "alpha_by_regime_exhibit.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    sharpe_df = _regime_metric_frame(summary_table, regime_types, 'Sharpe')
    fraction_df = _regime_metric_frame(summary_table, regime_types, 'Return_Fraction')

    n_strategies = len(sharpe_df)
    fig, (ax_sharpe, ax_fraction) = plt.subplots(
        2, 1,
        figsize=(10, max(7, n_strategies * 0.55 + 4)),
        gridspec_kw={'height_ratios': [1.2, 1]},
    )

    if regime_method == 'phase':
        title = 'Alpha Performance by Regime'
        subtitle = 'Sharpe by regime (top) | Share of cumulative returns by regime (bottom)'
    else:
        title = 'Alpha Performance by Regime: Transition vs Stable'
        subtitle = 'Sharpe by regime (top) | Share of cumulative returns by regime (bottom)'

    vmax = max(1.0, np.nanmax(np.abs(sharpe_df.values))) if sharpe_df.size else 1.0
    sns.heatmap(
        sharpe_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Sharpe Ratio'},
        ax=ax_sharpe,
    )
    ax_sharpe.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold', pad=8)
    ax_sharpe.set_xlabel('')
    ax_sharpe.set_ylabel('Strategy')

    if regime_labels is not None:
        month_labels = [
            f"{REGIME_DISPLAY_NAMES.get(r, r)}\n(n={(regime_labels == r).sum()})"
            for r in regime_types
            if REGIME_DISPLAY_NAMES.get(r, r) in sharpe_df.columns
        ]
        ax_sharpe.set_xticklabels(month_labels, rotation=0, ha='center')

    strategies = fraction_df.index.tolist()
    left = np.zeros(len(strategies))
    for regime in regime_types:
        display = REGIME_DISPLAY_NAMES.get(regime, regime.title())
        if display not in fraction_df.columns:
            continue
        values = fraction_df[display].fillna(0).values
        ax_fraction.barh(
            strategies,
            values,
            left=left,
            label=display,
            color=REGIME_COLORS.get(regime, '#888888'),
            edgecolor='white',
            linewidth=0.5,
        )
        left += values

    ax_fraction.set_xlim(0, 1.05)
    ax_fraction.set_xlabel('Fraction of Total Cumulative Return')
    ax_fraction.set_title('Return Contribution by Regime', fontsize=12, fontweight='bold', pad=8)
    ax_fraction.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(regime_types),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exhibit saved to: {save_path}")
    plt.close()


# -----------------------------------------------------------------------------#
# 6  Main Execution Function
# -----------------------------------------------------------------------------#
def _print_interpretation(summary_table: pd.DataFrame, regime_method: str, regime_types: List[str]) -> None:
    """Print regime performance interpretation."""
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)

    for strategy in summary_table.index:
        print(f"\n{strategy}:")

        if regime_method == 'phase':
            sharpe_cols = {
                r: f'{REGIME_DISPLAY_NAMES[r]}_Sharpe'
                for r in regime_types
                if f'{REGIME_DISPLAY_NAMES[r]}_Sharpe' in summary_table.columns
            }
            if 'crisis_onset' in sharpe_cols and 'resolution' in sharpe_cols:
                onset = summary_table.loc[strategy, sharpe_cols['crisis_onset']]
                resolution = summary_table.loc[strategy, sharpe_cols['resolution']]
                if not np.isnan(onset) and not np.isnan(resolution):
                    if onset < resolution:
                        print(f"  -> Ash hypothesis: crisis onset weaker than resolution ({onset:.3f} vs {resolution:.3f})")
                    else:
                        print(f"  -> Ash hypothesis NOT supported: crisis onset >= resolution ({onset:.3f} vs {resolution:.3f})")

            valid_sharpes = {
                REGIME_DISPLAY_NAMES[r]: summary_table.loc[strategy, sharpe_cols[r]]
                for r in regime_types if r in sharpe_cols and not np.isnan(summary_table.loc[strategy, sharpe_cols[r]])
            }
            if valid_sharpes:
                best = max(valid_sharpes, key=valid_sharpes.get)
                print(f"  -> Highest Sharpe in {best} ({valid_sharpes[best]:.3f})")
        else:
            trans_col = 'Transition_Sharpe'
            stable_col = 'Stable_Sharpe'
            trans_frac_col = 'Transition_Return_Fraction'
            stable_frac_col = 'Stable_Return_Fraction'

            if trans_col in summary_table.columns and stable_col in summary_table.columns:
                trans_sharpe = summary_table.loc[strategy, trans_col]
                stable_sharpe = summary_table.loc[strategy, stable_col]
                if not np.isnan(trans_sharpe) and not np.isnan(stable_sharpe):
                    if trans_sharpe > stable_sharpe:
                        print(f"  -> Higher Sharpe in TRANSITION regimes ({trans_sharpe:.3f} vs {stable_sharpe:.3f})")
                    else:
                        print(f"  -> Higher Sharpe in STABLE regimes ({stable_sharpe:.3f} vs {trans_sharpe:.3f})")

            if trans_frac_col in summary_table.columns and stable_frac_col in summary_table.columns:
                trans_frac = summary_table.loc[strategy, trans_frac_col]
                stable_frac = summary_table.loc[strategy, stable_frac_col]
                if not np.isnan(trans_frac) and not np.isnan(stable_frac):
                    if abs(trans_frac) > abs(stable_frac):
                        print(f"  -> More returns from TRANSITION regimes ({trans_frac:.1%} vs {stable_frac:.1%})")
                    else:
                        print(f"  -> More returns from STABLE regimes ({stable_frac:.1%} vs {trans_frac:.1%})")


def run_alpha_by_regime_analysis(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    regime_method: str = "percentile",
    regime_threshold_percentile: Optional[float] = None,
    regime_threshold_absolute: Optional[float] = None,
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None,
    use_cache: bool = True,
    create_exhibit: bool = True
) -> pd.DataFrame:
    """
    Run complete alpha-by-regime analysis end-to-end.
    Focuses on identifying whether most alpha (Sharpe ratio) is generated during
    stable or transitionary regimes, showing cumulative return contribution and
    Sharpe ratio by regime.
    
    Parameters
    ----------
    n_buckets : int, default 5
        Number of buckets for backtest
    back_test_start_date : str, default "1985-01-31"
        Start date for backtest
    forward_look_months : int, default 1
        Forward look months
    similarity_window : int, default 1
        Similarity window size
    regime_method : str, default "percentile"
        Method for regime labeling: "phase", "percentile", or "absolute"
    regime_threshold_percentile : float, optional
        Percentile threshold (0-1) used when method="percentile"
    regime_threshold_absolute : float, optional
        Absolute EWMA value threshold used when method="absolute"
    low_threshold_percentile : float, optional
        Lower percentile for phase method (stable vs elevated)
    high_threshold_percentile : float, optional
        Upper percentile for phase method (elevated vs crisis)
    use_cache : bool, default True
        Whether to use cached data
    create_exhibit : bool, default True
        Whether to create and save a visualization exhibit
        
    Returns
    -------
    pd.DataFrame
        Summary table with Sharpe ratio, return contribution, and sample size by regime
    """
    print("="*70)
    print("Alpha Performance by Regime Analysis")
    if regime_method == 'phase':
        print("Focus: Alpha across stable / elevated / crisis onset / resolution")
    else:
        print("Focus: Where is most alpha generated? (Stable vs Transitionary)")
    print("="*70)
    
    print("\n1. Loading EWMA regime shifts...")
    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)
    
    print("\n2. Labeling regimes...")
    regime_labels = label_regimes(
        ewma_df,
        method=regime_method,
        threshold_percentile=regime_threshold_percentile,
        threshold_absolute=regime_threshold_absolute,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile,
    )
    regime_types = get_regime_order(regime_method, regime_labels)
    
    # Load backtest returns
    print("\n3. Loading backtest returns...")
    backtest_returns = load_backtest_returns(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window
    )
    
    # Align regime labels with backtest returns
    print("\n4. Aligning regime labels with backtest returns...")
    common_dates = backtest_returns.index.intersection(regime_labels.index)
    backtest_returns_aligned = backtest_returns.loc[common_dates]
    regime_labels_aligned = regime_labels.loc[common_dates]
    
    print(f"   Aligned {len(common_dates)} months of data")
    print(f"   Date range: {common_dates.min()} to {common_dates.max()}")
    
    # Compute performance by regime
    print("\n5. Computing performance metrics by regime...")
    regime_performance = compute_regime_performance(
        backtest_returns_aligned,
        regime_labels_aligned,
        regime_types,
    )
    
    print("\n6. Creating summary table...")
    summary_table = create_summary_table(regime_performance, regime_types)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Alpha Performance by Regime")
    print("="*70)
    print("\nKey Metrics:")
    print("- Sharpe Ratio: Risk-adjusted return within each regime")
    print("- Return Fraction: Fraction of total cumulative returns from each regime")
    print("="*70)
    
    # Format numbers for better readability
    summary_display = summary_table.copy()
    for col in summary_display.columns:
        if 'Sharpe' in col:
            summary_display[col] = summary_display[col].round(3)
        elif 'Fraction' in col or 'CumReturn' in col:
            summary_display[col] = summary_display[col].round(4)
    
    print("\n" + summary_display.to_string())
    
    _print_interpretation(summary_table, regime_method, regime_types)
    
    if create_exhibit:
        print("\n7. Creating exhibit visualization...")
        create_alpha_by_regime_exhibit(
            summary_table, regime_types, regime_method, regime_labels_aligned
        )
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return summary_table


# -----------------------------------------------------------------------------#
# 6  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze alpha performance by regime.")
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--method', type=str, default=None,
                       choices=['phase', 'percentile', 'absolute'],
                       help='Method for regime labeling (overrides config)')
    parser.add_argument('--low-threshold-percentile', type=float, default=None,
                       help='Lower percentile for phase method (overrides config)')
    parser.add_argument('--high-threshold-percentile', type=float, default=None,
                       help='Upper percentile for phase method (overrides config)')
    parser.add_argument('--threshold-percentile', type=float, default=None,
                       help='Percentile threshold (0-1) used when method="percentile" (overrides config)')
    parser.add_argument('--threshold-absolute', type=float, default=None,
                       help='Absolute EWMA value threshold used when method="absolute" (overrides config)')
    args = parser.parse_args()
    
    alpha_config = cfg.get("regime_shifts", {}).get("alpha_by_regime", {})
    params = dict(
        n_buckets=cfg["backtest"].get("n_buckets", 5),
        back_test_start_date=cfg["backtest"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["backtest"].get("forward_look_months", 1),
        similarity_window=cfg["state_variables"]["similarity_score"].get("similarity_window", 1),
        regime_method=args.method if args.method is not None else alpha_config.get("regime_method", "percentile"),
        regime_threshold_percentile=args.threshold_percentile if args.threshold_percentile is not None else alpha_config.get("regime_threshold_percentile", None),
        regime_threshold_absolute=args.threshold_absolute if args.threshold_absolute is not None else alpha_config.get("regime_threshold_absolute", None),
        low_threshold_percentile=args.low_threshold_percentile if args.low_threshold_percentile is not None else alpha_config.get("low_threshold_percentile", None),
        high_threshold_percentile=args.high_threshold_percentile if args.high_threshold_percentile is not None else alpha_config.get("high_threshold_percentile", None),
        use_cache=not args.no_cache
    )
    
    summary = run_alpha_by_regime_analysis(**params)

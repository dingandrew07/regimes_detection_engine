# alpha_by_regime.py | Alpha Performance by Regime
# ------------------------------------------------------------------------------
# Analyzes backtest performance separately for transition vs stable months
# using EWMA-based regime shift detection. Focuses on:
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
from typing import Optional, Dict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

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
        from ...back_test import run_backtest
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from back_test import run_backtest
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
    threshold_absolute: Optional[float] = None
) -> pd.Series:
    """
    Label each month as 'transition' or 'stable' based on EWMA values.
    
    Parameters
    ----------
    ewma_df : pd.DataFrame
        EWMA regime shifts DataFrame with 'mean' column
    method : str, default "percentile"
        Method for labeling: "percentile" or "absolute"
    threshold_percentile : float, optional
        Percentile threshold (0-1) used when method="percentile" (e.g., 0.5 = median, 0.75 = 75th percentile)
    threshold_absolute : float, optional
        Absolute EWMA value threshold used when method="absolute"
        
    Returns
    -------
    pd.Series
        Series with 'transition' or 'stable' labels, indexed by date
    """
    mean_ewma = ewma_df['mean'].dropna()
    
    if method == "percentile":
        if threshold_percentile is None:
            raise ValueError("threshold_percentile parameter required when method='percentile'")
        if not (0 <= threshold_percentile <= 1):
            raise ValueError(f"Percentile threshold must be between 0 and 1 (got {threshold_percentile})")
        threshold_value = mean_ewma.quantile(threshold_percentile)
        print(f"Regime labeling complete (method: {method}, percentile: {threshold_percentile:.3f}, threshold value: {threshold_value:.4f})")
    elif method == "absolute":
        if threshold_absolute is None:
            raise ValueError("threshold_absolute parameter required when method='absolute'")
        threshold_value = threshold_absolute
        print(f"Regime labeling complete (method: {method}, threshold: {threshold_value:.4f})")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'absolute'")
    
    labels = pd.Series(
        index=mean_ewma.index,
        data=['transition' if val > threshold_value else 'stable' for val in mean_ewma.values],
        name='regime'
    )
    
    print(f"  Transition months: {(labels == 'transition').sum()} ({(labels == 'transition').mean()*100:.1f}%)")
    print(f"  Stable months: {(labels == 'stable').sum()} ({(labels == 'stable').mean()*100:.1f}%)")
    
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
    
    # Annualized Sharpe
    if regime_returns.std() > 0:
        ann_sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(12)
    else:
        ann_sharpe = np.nan
    
    return {
        'ann_sharpe': ann_sharpe,
        'n_months': len(regime_returns)
    }


def calculate_cumulative_return_contribution(
    returns: pd.Series,
    regime_labels: pd.Series
) -> Dict[str, float]:
    """
    Calculate cumulative return contribution by regime.
    Uses log returns for additive decomposition of cumulative returns.
    Returns the fraction of total cumulative returns contributed by each regime.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    regime_labels : pd.Series
        Regime labels ('transition' or 'stable')
        
    Returns
    -------
    dict
        Dictionary with cumulative return contributions for each regime
    """
    # Align indices
    aligned_data = pd.DataFrame({
        'returns': returns,
        'regime': regime_labels
    }).dropna()
    
    if len(aligned_data) == 0:
        return {
            'transition_contribution': np.nan,
            'stable_contribution': np.nan,
            'total_cumulative_return': np.nan
        }
    
    # Calculate log returns (additive for cumulative decomposition)
    aligned_data['log_return'] = np.log(1 + aligned_data['returns'])
    
    # Calculate cumulative returns for each regime
    transition_returns = aligned_data[aligned_data['regime'] == 'transition']['returns']
    stable_returns = aligned_data[aligned_data['regime'] == 'stable']['returns']
    transition_log_returns = aligned_data[aligned_data['regime'] == 'transition']['log_return']
    stable_log_returns = aligned_data[aligned_data['regime'] == 'stable']['log_return']
    
    # Total cumulative return
    total_cumret = (1 + aligned_data['returns']).prod() - 1
    
    # Cumulative returns for each regime (if we only had those months)
    transition_cumret = (1 + transition_returns).prod() - 1 if len(transition_returns) > 0 else 0.0
    stable_cumret = (1 + stable_returns).prod() - 1 if len(stable_returns) > 0 else 0.0
    
    # Calculate contributions using log returns (additive decomposition)
    total_log_return = aligned_data['log_return'].sum()
    transition_log_sum = transition_log_returns.sum() if len(transition_log_returns) > 0 else 0.0
    stable_log_sum = stable_log_returns.sum() if len(stable_log_returns) > 0 else 0.0
    
    # Contribution as fraction of total log return
    if abs(total_log_return) > 1e-10:  # Avoid division by zero
        transition_contribution = transition_log_sum / total_log_return
        stable_contribution = stable_log_sum / total_log_return
    else:
        # If total is near zero, use simple return sums
        total_sum = aligned_data['returns'].sum()
        transition_sum = transition_returns.sum() if len(transition_returns) > 0 else 0.0
        stable_sum = stable_returns.sum() if len(stable_returns) > 0 else 0.0
        if abs(total_sum) > 1e-10:
            transition_contribution = transition_sum / total_sum
            stable_contribution = stable_sum / total_sum
        else:
            transition_contribution = 0.0 if len(transition_returns) > 0 else np.nan
            stable_contribution = 0.0 if len(stable_returns) > 0 else np.nan
    
    return {
        'transition_contribution': transition_contribution,
        'stable_contribution': stable_contribution,
        'transition_cumulative_return': transition_cumret,
        'stable_cumulative_return': stable_cumret,
        'total_cumulative_return': total_cumret
    }


def compute_regime_performance(
    backtest_returns: pd.DataFrame,
    regime_labels: pd.Series
) -> pd.DataFrame:
    """
    Compute performance metrics separately for transition and stable regimes.
    Focuses on Sharpe ratio, cumulative return contribution, and sample size.
    
    Parameters
    ----------
    backtest_returns : pd.DataFrame
        Backtest returns for all strategies
    regime_labels : pd.Series
        Regime labels for each month
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each strategy
    """
    results = []
    
    for strategy in backtest_returns.columns:
        strategy_returns = backtest_returns[strategy]
        
        # Calculate Sharpe and sample size for each regime
        transition_metrics = calculate_regime_metrics(
            strategy_returns,
            regime_labels,
            regime_type='transition'
        )
        stable_metrics = calculate_regime_metrics(
            strategy_returns,
            regime_labels,
            regime_type='stable'
        )
        
        # Calculate cumulative return contribution
        contribution_metrics = calculate_cumulative_return_contribution(
            strategy_returns,
            regime_labels
        )
        
        # Combine all metrics
        result = {
            'strategy': strategy,
            'transition_sharpe': transition_metrics['ann_sharpe'],
            'stable_sharpe': stable_metrics['ann_sharpe'],
            'transition_n_months': transition_metrics['n_months'],
            'stable_n_months': stable_metrics['n_months'],
            'transition_return_contribution': contribution_metrics['transition_contribution'],
            'stable_return_contribution': contribution_metrics['stable_contribution'],
            'transition_cumulative_return': contribution_metrics['transition_cumulative_return'],
            'stable_cumulative_return': contribution_metrics['stable_cumulative_return'],
            'total_cumulative_return': contribution_metrics['total_cumulative_return']
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('strategy')
    
    return results_df


# -----------------------------------------------------------------------------#
# 4  Summary Table Generation
# -----------------------------------------------------------------------------#
def create_summary_table(regime_performance: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted summary table comparing transition vs stable performance.
    Shows Sharpe ratio and cumulative return contribution.
    
    Parameters
    ----------
    regime_performance : pd.DataFrame
        Output from compute_regime_performance
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    # Reorder columns for better readability
    column_order = [
        'transition_sharpe',
        'stable_sharpe',
        'transition_return_contribution',
        'stable_return_contribution',
        'transition_cumulative_return',
        'stable_cumulative_return',
        'total_cumulative_return'
    ]
    
    # Select and reorder columns
    available_cols = [col for col in column_order if col in regime_performance.columns]
    summary_df = regime_performance[available_cols].copy()
    
    # Rename columns for better readability
    summary_df.columns = [
        'Transition_Sharpe',
        'Stable_Sharpe',
        'Transition_Return_Fraction',
        'Stable_Return_Fraction',
        'Transition_CumReturn',
        'Stable_CumReturn',
        'Total_CumReturn'
    ]
    
    return summary_df


# -----------------------------------------------------------------------------#
# 5  Visualization Functions
# -----------------------------------------------------------------------------#
def create_alpha_by_regime_exhibit(
    summary_table: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Create a professional-looking exhibit table showing alpha performance by regime.
    
    Parameters
    ----------
    summary_table : pd.DataFrame
        Summary table from create_summary_table
    regime_labels : pd.Series, optional
        Regime labels to calculate month counts. If None, counts won't be displayed.
    save_path : Path, optional
        Path to save the exhibit. If None, uses reports/regime_shifts/
    """
    if save_path is None:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "alpha_by_regime_exhibit.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, max(6, len(summary_table) * 0.8 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for display
    display_df = summary_table.copy()
    
    # Format numbers
    for col in display_df.columns:
        if 'Sharpe' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A")
        elif 'Fraction' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A")
        elif 'CumReturn' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if not np.isnan(x) else "N/A")
    
    # Rename columns for better display
    display_df.columns = [
        'Transition\nSharpe',
        'Stable\nSharpe',
        'Transition\nReturn\nFraction',
        'Stable\nReturn\nFraction',
        'Transition\nCumulative\nReturn',
        'Stable\nCumulative\nReturn',
        'Total\nCumulative\nReturn'
    ]
    
    # Create table
    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header row styling
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.08)
    
    # Row label styling
    for i in range(len(display_df.index)):
        cell = table[(i + 1, -1)]
        cell.set_facecolor('#E7E6E6')
        cell.set_text_props(weight='bold')
    
    # Alternate row colors for readability
    for i in range(len(display_df.index)):
        for j in range(len(display_df.columns)):
            cell = table[(i + 1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(0.5)
    
    # Highlight higher Sharpe values (use original data, not formatted)
    trans_col_idx = list(display_df.columns).index('Transition\nSharpe')
    stable_col_idx = list(display_df.columns).index('Stable\nSharpe')
    
    for i, strategy in enumerate(display_df.index):
        # Get original values from summary_table
        trans_sharpe = summary_table.loc[strategy, 'Transition_Sharpe']
        stable_sharpe = summary_table.loc[strategy, 'Stable_Sharpe']
        
        if not np.isnan(trans_sharpe) and not np.isnan(stable_sharpe):
            # Highlight the higher Sharpe
            if trans_sharpe > stable_sharpe:
                cell = table[(i + 1, trans_col_idx)]
                cell.set_facecolor('#C5E0B4')  # Light green
            elif stable_sharpe > trans_sharpe:
                cell = table[(i + 1, stable_col_idx)]
                cell.set_facecolor('#C5E0B4')  # Light green
    
    # Calculate month counts if regime_labels provided
    month_counts_text = ""
    if regime_labels is not None:
        n_transition = (regime_labels == 'transition').sum()
        n_stable = (regime_labels == 'stable').sum()
        month_counts_text = f"Transition months: {n_transition} | Stable months: {n_stable}"
    
    # Add title
    plt.suptitle(
        'Alpha Performance by Regime: Transition vs Stable',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # Add month counts in top right corner
    if month_counts_text:
        fig.text(0.98, 0.97, month_counts_text, ha='right', fontsize=9, style='normal', color='black')
    
    # Add subtitle with explanation
    subtitle_text = (
        'Sharpe Ratio: Risk-adjusted return within each regime | '
        'Return Fraction: Fraction of total cumulative returns'
    )
    fig.text(0.5, 0.94, subtitle_text, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exhibit saved to: {save_path}")
    plt.close()


# -----------------------------------------------------------------------------#
# 6  Main Execution Function
# -----------------------------------------------------------------------------#
def run_alpha_by_regime_analysis(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    regime_method: str = "percentile",
    regime_threshold_percentile: Optional[float] = None,
    regime_threshold_absolute: Optional[float] = None,
    use_cache: bool = True,
    save_table: bool = True,
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
        Method for regime labeling: "percentile" or "absolute"
    regime_threshold_percentile : float, optional
        Percentile threshold (0-1) used when method="percentile"
    regime_threshold_absolute : float, optional
        Absolute EWMA value threshold used when method="absolute"
    use_cache : bool, default True
        Whether to use cached data
    save_table : bool, default True
        Whether to save summary table to reports folder
    create_exhibit : bool, default True
        Whether to create and save a visualization exhibit
        
    Returns
    -------
    pd.DataFrame
        Summary table with Sharpe ratio, return contribution, and sample size by regime
    """
    print("="*70)
    print("Alpha Performance by Regime Analysis")
    print("Focus: Where is most alpha generated? (Stable vs Transitionary)")
    print("="*70)
    
    # Load EWMA regime shifts
    print("\n1. Loading EWMA regime shifts...")
    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)
    
    # Label regimes
    print("\n2. Labeling regimes...")
    regime_labels = label_regimes(
        ewma_df, 
        method=regime_method, 
        threshold_percentile=regime_threshold_percentile,
        threshold_absolute=regime_threshold_absolute
    )
    
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
        regime_labels_aligned
    )
    
    # Create summary table
    print("\n6. Creating summary table...")
    summary_table = create_summary_table(regime_performance)
    
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
    
    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    for strategy in summary_table.index:
        trans_sharpe = summary_table.loc[strategy, 'Transition_Sharpe']
        stable_sharpe = summary_table.loc[strategy, 'Stable_Sharpe']
        trans_frac = summary_table.loc[strategy, 'Transition_Return_Fraction']
        stable_frac = summary_table.loc[strategy, 'Stable_Return_Fraction']
        
        print(f"\n{strategy}:")
        if not np.isnan(trans_sharpe) and not np.isnan(stable_sharpe):
            if trans_sharpe > stable_sharpe:
                print(f"  → Higher Sharpe in TRANSITION regimes ({trans_sharpe:.3f} vs {stable_sharpe:.3f})")
            else:
                print(f"  → Higher Sharpe in STABLE regimes ({stable_sharpe:.3f} vs {trans_sharpe:.3f})")
        
        if not np.isnan(trans_frac) and not np.isnan(stable_frac):
            if abs(trans_frac) > abs(stable_frac):
                print(f"  → More returns from TRANSITION regimes ({trans_frac:.1%} vs {stable_frac:.1%})")
            else:
                print(f"  → More returns from STABLE regimes ({stable_frac:.1%} vs {trans_frac:.1%})")
    
    # Save table
    if save_table:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_file = reports_dir / "alpha_by_regime_summary.csv"
        summary_table.to_csv(output_file)
        print(f"\n\nSummary table saved to: {output_file}")
    
    # Create exhibit visualization
    if create_exhibit:
        print("\n7. Creating exhibit visualization...")
        create_alpha_by_regime_exhibit(summary_table, regime_labels_aligned)
    
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
                       choices=['percentile', 'absolute'],
                       help='Method for regime labeling: "percentile" or "absolute" (overrides config)')
    parser.add_argument('--threshold-percentile', type=float, default=None,
                       help='Percentile threshold (0-1) used when method="percentile" (overrides config)')
    parser.add_argument('--threshold-absolute', type=float, default=None,
                       help='Absolute EWMA value threshold used when method="absolute" (overrides config)')
    args = parser.parse_args()
    
    # Get parameters from config, with CLI args as overrides
    alpha_config = cfg.get("analysis", {}).get("alpha_by_regime", {})
    params = dict(
        n_buckets=cfg["back_test"].get("n_buckets", 5),
        back_test_start_date=cfg["back_test"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["back_test"].get("forward_look_months", 1),
        similarity_window=cfg["similarity_score"].get("similarity_window", 1),
        regime_method=args.method if args.method is not None else alpha_config.get("regime_method", "percentile"),
        regime_threshold_percentile=args.threshold_percentile if args.threshold_percentile is not None else alpha_config.get("regime_threshold_percentile", None),
        regime_threshold_absolute=args.threshold_absolute if args.threshold_absolute is not None else alpha_config.get("regime_threshold_absolute", None),
        use_cache=not args.no_cache
    )
    
    summary = run_alpha_by_regime_analysis(**params)

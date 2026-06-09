# back_test.py | Section 4 – Main Backtesting Engine
# ------------------------------------------------------------------------------
# Implements the main backtesting engine for regime-based factor strategies.
# Follows the expanding window, bucketed similarity, and per-factor signal rules.
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
from typing import Optional, Dict, Any
import yaml
import matplotlib.pyplot as plt

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

# -----------------------------------------------------------------------------#
# 1  Data Loading
# -----------------------------------------------------------------------------#
# Flag to track if factor data info has been printed
_factors_info_printed = False

def load_similarity_scores(similarity_window: int = 1) -> pd.DataFrame:
    return joblib.load(CACHE_DIR / f"similarity_scores_window{similarity_window}.pkl")

def load_factors() -> pd.DataFrame:
    global _factors_info_printed
    df = pd.read_pickle(CACHE_DIR / "df_factors.pkl")
    # Shift by -1 so that row T contains returns from T to T+1
    # This aligns signals formed at month-end T with P&L earned during T+1
    df = df.shift(-1)
    # Only print this information once, the first time factors are loaded
    if not _factors_info_printed:
        print("\nFactor data after -1 shift (tail):")
        print(df.tail())
        print(f"\nLast row has NaN values: {df.iloc[-1].isna().all()}")
        _factors_info_printed = True
    return df

# -----------------------------------------------------------------------------#
# 2  Main Backtesting Engine
# -----------------------------------------------------------------------------#
def run_backtest(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    verbose: bool = False,
    show_alignment_message: bool = True,
    use_efficacy: bool = False,
    efficacy_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Main backtesting engine. Returns DataFrame of bucketed strategy returns and summary stats.
    
    Parameters:
    -----------
    n_buckets : int
        Number of buckets/quintiles for similarity-based grouping
    back_test_start_date : str
        Start date for the backtest period
    forward_look_months : int
        Number of months to look forward for returns (default: 1 month)
    similarity_window : int
        Rolling window size used for similarity calculation (default: 1 month)
    verbose : bool
        Whether to print verbose output
    use_efficacy : bool, default False
        Whether to use efficacy score extension for confidence scaling
    efficacy_config : dict or None, default None
        Configuration for efficacy extension (bootstrap_iterations, etc.)
    """
    # Load data
    similarity_scores = load_similarity_scores(similarity_window=similarity_window)
    df_factors = load_factors()

    # Align indices: both series start at the latest of their earliest available dates
    min_date = max(similarity_scores.index.min(), df_factors.index.min())
    max_date = min(similarity_scores.index.max(), df_factors.index.max())
    if show_alignment_message:
        print(f"Aligning data: earliest date used in backtesting is {min_date.strftime('%Y-%m-%d')}")
    similarity_scores = similarity_scores.loc[min_date:max_date]
    df_factors = df_factors.loc[min_date:max_date]
    similarity_scores = similarity_scores.iloc[:-1]
    df_factors = df_factors.iloc[:-1]

    # Ensure chronological order
    similarity_scores = similarity_scores.sort_index()
    df_factors = df_factors.sort_index()

    # Prepare output
    all_bucket_returns = {}
    factor_names = df_factors.columns
    dates = df_factors.index
    
    # Initialize efficacy extension data structures
    if use_efficacy:
        try:
            from src.extensions.efficacy_score import (
                compute_efficacy_with_realized,
                efficacy_to_multiplier,
            )
        except ImportError:
            from extensions.efficacy_score import (
                compute_efficacy_with_realized,
                efficacy_to_multiplier,
            )
        efficacy_config = efficacy_config or {}
        bootstrap_iterations = efficacy_config.get('bootstrap_iterations', 200)
        random_seed = cfg.get('analysis', {}).get('random_seed', 42)
        
        efficacy_scores = pd.Series(index=dates, dtype=float)
        efficacy_stds = pd.Series(index=dates, dtype=float)
        multipliers = pd.Series(index=dates, dtype=float)
        
        if verbose:
            print(f"Efficacy extension enabled: {bootstrap_iterations} bootstrap iterations")
    else:
        efficacy_scores = None
        efficacy_stds = None
        multipliers = None

    for quintile in range(1, n_buckets + 1):
        bucket_returns = []
        for t, T in enumerate(dates):
            # Expanding window: use all history up to T (excluding T)
            if t == 0:
                bucket_returns.append(np.nan)
                continue
            hist_months = dates[:t]
            dists = similarity_scores[T].loc[hist_months]
            # Remove NaN values (masked months) before ranking
            # This ensures masked months are not included in quintile buckets
            dists = dists.dropna()
            if len(dists) == 0:
                # No valid historical data, skip this iteration
                bucket_returns.append(np.nan)
                continue
            # Rank distances, break ties deterministically
            ranks = dists.rank(method="first")
            # Method 1: Equal-frequency buckets using numpy array splitting
            sorted_idx = ranks.sort_values().index
            n_hist = len(sorted_idx)
            bucket_sizes = [n_hist // n_buckets + (1 if x < n_hist % n_buckets else 0) for x in range(n_buckets)]
            bucket_edges = np.cumsum([0] + bucket_sizes)
            # Select the months in the current quintile (bucket)
            bucket_start = bucket_edges[quintile - 1]
            bucket_end = bucket_edges[quintile]
            bucket_idx = sorted_idx[bucket_start:bucket_end]
            
            # Compute efficacy score for Quintile 1 (if extension is enabled and we're on Q1)
            if use_efficacy and quintile == 1:
                # S(T) = Quintile 1 months (fixed for this month T)
                S_T = bucket_idx
                
                # Compute efficacy score
                efficacy_score, efficacy_std = compute_efficacy_with_realized(
                    S_T=S_T,
                    T=T,
                    df_factors=df_factors,
                    factor_names=factor_names,
                    bootstrap_iterations=bootstrap_iterations,
                    random_seed=random_seed,
                )
                
                # Store efficacy metrics
                efficacy_scores.loc[T] = efficacy_score
                efficacy_stds.loc[T] = efficacy_std
                
                # Compute multiplier from efficacy
                mult_T = efficacy_to_multiplier(efficacy_score)
                multipliers.loc[T] = mult_T
            elif use_efficacy and quintile == 1 and len(bucket_idx) == 0:
                # Edge case: no valid S(T) for efficacy calculation
                efficacy_scores.loc[T] = np.nan
                efficacy_stds.loc[T] = np.nan
                multipliers.loc[T] = 0.5  # Default to neutral multiplier
            
            # Get multiplier for this month (1.0 if efficacy not enabled)
            # For quintiles 2-5, read the multiplier computed during Q1 processing
            if use_efficacy:
                if T in multipliers.index:
                    mult_T = multipliers.loc[T]
                    if np.isnan(mult_T):
                        mult_T = 0.5  # Default to neutral if NaN
                else:
                    mult_T = 0.5  # Default if not yet computed (shouldn't happen)
            else:
                mult_T = 1.0
            
            # Per-factor signal
            factor_signals = []
            for f in factor_names:
                bucket_months = bucket_idx
                if len(bucket_months) == 0:
                    factor_signals.append(0)
                    continue
                mean_ret = df_factors.loc[bucket_months, f].mean()
                # Position rule: no flat, only long or short
                if mean_ret > 0:
                    signal = 1
                else:
                    signal = -1
                factor_signals.append(signal)
            
            # Apply efficacy multiplier to factor signals (for all quintiles)
            if use_efficacy:
                factor_signals = [s * mult_T for s in factor_signals]
            # Calculate forward-looking returns based on forward_look_months parameter
            if forward_look_months == 1:
                # Default behavior: use next month's returns
                forward_returns = df_factors.loc[T].values
            else:
                # Look forward multiple months and calculate cumulative returns
                future_dates = dates[t:t + forward_look_months]
                if len(future_dates) >= forward_look_months:
                    # Calculate cumulative returns over the forward period
                    future_factor_returns = df_factors.loc[future_dates].values
                    # Convert to cumulative returns: (1 + r1) * (1 + r2) * ... - 1
                    cumulative_returns = np.prod(1 + future_factor_returns, axis=0) - 1
                    forward_returns = cumulative_returns
                else:
                    # Not enough future data, use available data or skip
                    forward_returns = np.full(len(factor_names), np.nan)
            
            realized = np.nansum(np.array(factor_signals) * forward_returns) / len(factor_names)
            bucket_returns.append(realized)
        all_bucket_returns[f"Q{quintile}"] = pd.Series(bucket_returns, index=dates)

    # Build DataFrame
    quintile_returns = pd.DataFrame(all_bucket_returns)
    # Long-short: Q1 - Q{n_buckets}
    quintile_returns[f"Q1_minus_Q{n_buckets}"] = quintile_returns["Q1"] - quintile_returns[f"Q{n_buckets}"]
    # Long-only benchmark (mean of factors - 1/6 exposure to each)
    quintile_returns["long_only"] = df_factors.mean(axis=1)

    # Restrict output to back_test_start_date onward
    quintile_returns = quintile_returns.loc[back_test_start_date:]
    
    # Store efficacy data as attributes of the return DataFrame (for later access)
    if use_efficacy:
        quintile_returns.attrs['efficacy_scores'] = efficacy_scores.loc[back_test_start_date:]
        quintile_returns.attrs['efficacy_stds'] = efficacy_stds.loc[back_test_start_date:]
        quintile_returns.attrs['multipliers'] = multipliers.loc[back_test_start_date:]
    
    return quintile_returns

# -----------------------------------------------------------------------------#
# 3  Performance Metrics
# -----------------------------------------------------------------------------#
def compute_performance_metrics(
    returns: pd.DataFrame,
    risk_free: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute AnnSharpe, CorrWithLO, Mean, Std for each strategy column.
    """
    metrics = {}
    if risk_free is not None:
        excess = returns.sub(risk_free, axis=0)
    else:
        excess = returns.copy()
    for col in returns.columns:
        strat = returns[col]
        ex = excess[col]
        ann_sharpe = ex.mean() / ex.std() * np.sqrt(12)
        corr_with_lo = strat.corr(returns["long_only"])
        metrics[col] = {
            "AnnSharpe": ann_sharpe,
            "CorrWithLO": corr_with_lo,
            "Mean": strat.mean(),
            "Std": strat.std(),
        }
    return pd.DataFrame(metrics).T

# -----------------------------------------------------------------------------#
# 4  Drawdown Analysis Functions
# -----------------------------------------------------------------------------#
def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Monthly returns series
        
    Returns:
    --------
    pd.Series
        Drawdown series (negative values represent losses from peak)
    """
    # Calculate cumulative returns
    cumret = (1 + returns).cumprod()
    
    # Calculate running maximum (peak)
    peak = cumret.expanding().max()
    
    # Calculate drawdown as percentage from peak
    drawdown = (cumret / peak - 1) * 100
    
    return drawdown

def calculate_max_drawdown(returns: pd.Series) -> tuple:
    """
    Calculate maximum drawdown and its date.
    
    Parameters:
    -----------
    returns : pd.Series
        Monthly returns series
        
    Returns:
    --------
    tuple
        (max_drawdown, date_of_max_drawdown)
    """
    drawdown = calculate_drawdowns(returns)
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    return max_dd, max_dd_date

# -----------------------------------------------------------------------------#
# 5  Volatility Targeting and Exhibit 1 Functions
# -----------------------------------------------------------------------------#
def apply_volatility_targeting(returns: pd.Series, vol_target: float = 0.15, vol_window: int = 36) -> pd.Series:
    """
    Apply volatility targeting to scale returns to target annualized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Monthly returns series
    vol_target : float
        Target annualized volatility (default 15%)
    vol_window : int
        Rolling window for realized volatility calculation (default 36 months)
    
    Returns:
    --------
    pd.Series
        Volatility-targeted returns
    """
    # Calculate monthly target volatility
    monthly_target = vol_target / np.sqrt(12)
    
    # Calculate rolling realized volatility (monthly)
    rolling_std = returns.rolling(window=vol_window, min_periods=vol_window).std()
    
    # Calculate scaling factor (both in monthly units)
    scale_t = monthly_target / rolling_std
    
    # Shift by +1 to prevent look-ahead bias (use σ up to T-1 to scale return T to T+1)
    scale_t_shifted = scale_t.shift(1)
    
    # Apply scaling (with forward fill for first few observations)
    scale_t_shifted = scale_t_shifted.fillna(1.0)  # Use 1.0 (no scaling) for initial periods
    
    # Scale returns
    vol_targeted_returns = returns * scale_t_shifted
    
    return vol_targeted_returns

def generate_exhibit1(quintile_returns: pd.DataFrame, vol_target: float = 0.15, vol_window: int = 36):
    """
    Generate Exhibit 1: Bar chart of annualized volatility-targeting returns.
    """
    # Get the Q1-Q5 long-short returns
    ls_returns = quintile_returns["Q1_minus_Q5"].dropna()
    
    # Apply volatility targeting
    vol_targeted_returns = apply_volatility_targeting(ls_returns, vol_target, vol_window)
    
    # Convert to annual returns (sum of 12 months)
    annual_returns = vol_targeted_returns.resample('Y').sum() * 100  # Convert to percentage
    
    # Create bar chart with 40 equal-width bars
    plt.figure(figsize=(15, 8))
    
    # Create 40 equal-width bars
    n_bars = 40
    bar_width = len(annual_returns) / n_bars
    
    # Group returns into 40 equal-width bins
    returns_array = annual_returns.values
    n_returns = len(returns_array)
    bin_size = n_returns // n_bars
    
    binned_returns = []
    for i in range(n_bars):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_returns)
        if start_idx < n_returns:
            bin_mean = np.mean(returns_array[start_idx:end_idx])
            binned_returns.append(bin_mean)
    
    # Create x-axis labels (actual years from backtest start to end)
    start_year = annual_returns.index[0].year
    end_year = annual_returns.index[-1].year
    year_range = np.linspace(start_year, end_year, len(binned_returns))
    
    # Create bar chart
    bars = plt.bar(year_range, binned_returns, width=0.8, alpha=0.7, color='steelblue')
    
    # Color bars based on positive/negative returns
    for i, bar in enumerate(bars):
        if binned_returns[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Calculate metrics
    positive_years = sum(1 for r in binned_returns if r > 0)
    percentage_positive = (positive_years / len(binned_returns)) * 100
    
    avg_positive = np.mean([r for r in binned_returns if r > 0]) if any(r > 0 for r in binned_returns) else 0
    avg_negative = np.mean([r for r in binned_returns if r < 0]) if any(r < 0 for r in binned_returns) else 0
    
    # Add metrics as text
    plt.text(0.02, 0.98, f'Positive years: {percentage_positive:.1f}%', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.92, f'Avg positive return: {avg_positive:.1f}%', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.86, f'Avg negative return: {avg_negative:.1f}%', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Exhibit 1: Annualized Volatility-Targeting Returns (Q1-Q5 Long-Short)', fontsize=14, fontweight='bold')
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('% Excess Returns', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to reports directory
    try:
        from src.extensions.efficacy_score import get_reports_dir
    except ImportError:
        from extensions.efficacy_score import get_reports_dir
    reports_dir = get_reports_dir(Path(cfg["paths"]["reports_dir"]), cfg, "backtest")
    plt.savefig(reports_dir / "exhibit1_volatility_targeting.png", dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to prevent automatic plot display
    
    # Print metrics
    print(f"\nExhibit 1 Metrics:")
    print(f"Percentage of years with positive returns: {percentage_positive:.1f}%")
    print(f"Average return when positive: {avg_positive:.1f}%")
    print(f"Average return when negative: {avg_negative:.1f}%")
    
    return binned_returns, percentage_positive, avg_positive, avg_negative

# -----------------------------------------------------------------------------#
# 6  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Run backtest with defaults from config
    params = dict(
        n_buckets=cfg["back_test"].get("n_buckets", 5),
        back_test_start_date=cfg["back_test"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["back_test"].get("forward_look_months", 1),
        similarity_window=cfg["similarity_score"].get("similarity_window", 1),
    )
    
    # Check if efficacy extension is enabled
    efficacy_config = cfg.get("extensions", {}).get("efficacy_score", {})
    use_efficacy = efficacy_config.get("enabled", False)
    if use_efficacy:
        print(f"Efficacy extension enabled: {efficacy_config.get('bootstrap_iterations', 200)} bootstrap iterations")
    
    quintile_returns = run_backtest(
        **params,
        use_efficacy=use_efficacy,
        efficacy_config=efficacy_config,
    )

    # Check if there is data at the backtest start date
    start_date = pd.to_datetime(params["back_test_start_date"])
    print(f"\nBacktest returns at start date ({start_date.date()}):")
    print(quintile_returns.loc[start_date])
    # Find and print the first date with any available backtest return
    first_valid = quintile_returns.dropna(how='all').index[0]
    # print("First date with any backtest return:", first_valid)

    # Find and print the first date with any available backtest return for Q1-Q5 (excluding long_only and Q1_minus_Qn) that is non-NaN and non-zero
    quintile_cols = [col for col in quintile_returns.columns if col.startswith('Q') and '_' not in col]
    mask_nonzero = (quintile_returns[quintile_cols].notna() & (quintile_returns[quintile_cols] != 0)).any(axis=1)
    if mask_nonzero.any():
        first_nonzero_quintile = quintile_returns[quintile_cols].index[mask_nonzero][0]
        print("First date with any non-NaN, non-zero Q1-Q5 backtest return:", first_nonzero_quintile)
    else:
        print("No non-NaN, non-zero Q1-Q5 backtest returns found.")

    summary = compute_performance_metrics(quintile_returns)
    print("\nSummary Performance Metrics:")
    print(summary.round(4))
    # Save results - always overwrite cache to ensure parameter changes take effect
    # Note: quintile_returns.pkl is not used elsewhere in the codebase
    # quintile_returns.to_pickle(CACHE_DIR / "quintile_returns.pkl")
    summary.to_csv(CACHE_DIR / "backtest_summary.csv")

    # -----------------------------------------------------------------------------#
    # 7  Exhibit 10: Quintile Performance Plot
    # -----------------------------------------------------------------------------#
    print("Generating Exhibit 10: Quintile Performance Plot...")

    # Left plot (all quintiles + long_only) - cumprod()-1 scaling (start at 0)
    cumrets_geometric = ((1 + quintile_returns).cumprod() - 1) * 100

    # Right plot (Q1-Q5) - cumprod()-1 scaling (start at 0)
    cumrets_arithmetic = ((1 + quintile_returns).cumprod() - 1) * 100

    # Prepare legend labels with SR and corr
    legend_labels = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sr = summary.loc[q, "AnnSharpe"]
        corr = summary.loc[q, "CorrWithLO"]
        legend_labels.append(f"{q} (SR: {sr:.2f}, corr: {corr:.2f})")
    lo_sr = summary.loc["long_only", "AnnSharpe"]
    legend_labels.append(f"LO model (SR: {lo_sr:.2f})")

    # Left: all quintiles + long_only
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    ax1, ax2 = axes
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
        ax1.plot(cumrets_geometric.index, cumrets_geometric[q], label=legend_labels[i])
    ax1.plot(cumrets_geometric.index, cumrets_geometric["long_only"], label=legend_labels[-1], linestyle="--", color="gray")
    ax1.set_title("Fama-French factors: all quintiles")
    ax1.set_ylabel("Cumulative return")
    ax1.set_xlabel("Date")
    ax1.legend(fontsize=9)

    # Right: Q1-Q5 (long-short)
    ls_sr = summary.loc[f"Q1_minus_Q5", "AnnSharpe"]
    ls_corr = summary.loc[f"Q1_minus_Q5", "CorrWithLO"]
    ls_label = f"1st - 5th (SR: {ls_sr:.2f}, corr to LO: {ls_corr:.2f})"
    # Then plot Q1-Q5 using arithmetic returns
    ax2.plot(cumrets_arithmetic.index, cumrets_arithmetic[f"Q1_minus_Q5"], label=ls_label)
    ax2.set_title("Fama-French: 1st - 5th quintile")
    ax2.set_ylabel("Cumulative return")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    # Save Exhibit 10 plot to reports/
    try:
        from src.extensions.efficacy_score import get_reports_dir
    except ImportError:
        from extensions.efficacy_score import get_reports_dir
    reports_dir = get_reports_dir(Path(cfg["paths"]["reports_dir"]), cfg, "backtest")
    fig.savefig(reports_dir / "exhibit10_quintile_performance.png", dpi=150)
    # plt.show()  # (disabled: report is saved instead)

    # -----------------------------------------------------------------------------#
    # 7A  Equal-Weighted Long-Only Across All Quintiles
    # -----------------------------------------------------------------------------#
    # Check if equal-weighted exhibit should be generated (optional analysis)
    if cfg.get("extensions", {}).get("equal_weighted", {}).get("enabled", False):
        try:
            from .extensions.equal_weighted_exhibit import generate_equal_weighted_exhibit
        except ImportError:
            from extensions.equal_weighted_exhibit import generate_equal_weighted_exhibit
        generate_equal_weighted_exhibit(quintile_returns)

    # -----------------------------------------------------------------------------#
    # 7B  Random Long Bias Investigation
    # -----------------------------------------------------------------------------#
    # Check if random long bias investigation should be generated (optional extension)
    random_long_bias_config = cfg.get("extensions", {}).get("random_long_bias", {})
    if isinstance(random_long_bias_config, dict) and random_long_bias_config.get("enabled", False):
        try:
            from .extensions.random_long_bias import generate_random_long_bias_exhibit
        except ImportError:
            from extensions.random_long_bias import generate_random_long_bias_exhibit
        random_long_bias = random_long_bias_config.get("random_long_bias", 0.75)
        random_seed = random_long_bias_config.get("random_seed", 42)
        generate_random_long_bias_exhibit(
            original_quintile_returns=quintile_returns,
            n_buckets=params["n_buckets"],
            back_test_start_date=params["back_test_start_date"],
            forward_look_months=params["forward_look_months"],
            similarity_window=params["similarity_window"],
            random_seed=random_seed,
            long_bias=random_long_bias,
        )

    # -----------------------------------------------------------------------------#
    # 8  Exhibit 11: Drawdown Comparison
    # -----------------------------------------------------------------------------#
    print("Generating Exhibit 11: Drawdown Comparison...")
    
    # Get the returns data
    # For Exhibit 11, use LO with full exposure (sum of all factors)
    df_factors_ex11 = load_factors()
    # Align dates with quintile_returns
    df_factors_ex11 = df_factors_ex11.loc[quintile_returns.index]
    long_only_returns_ex11 = df_factors_ex11.sum(axis=1).dropna()
    long_short_returns = quintile_returns["Q1_minus_Q5"].dropna()
    
    # Align indices for comparison
    common_dates = long_only_returns_ex11.index.intersection(long_short_returns.index)
    long_only_returns = long_only_returns_ex11.loc[common_dates]
    long_short_returns = long_short_returns.loc[common_dates]
    
    # Calculate drawdowns
    lo_drawdown = calculate_drawdowns(long_only_returns)
    ls_drawdown = calculate_drawdowns(long_short_returns)
    
    # Calculate max drawdowns
    lo_max_dd, lo_max_dd_date = calculate_max_drawdown(long_only_returns)
    ls_max_dd, ls_max_dd_date = calculate_max_drawdown(long_short_returns)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot drawdowns
    plt.plot(lo_drawdown.index, lo_drawdown.values, label=f"LO Drawdown (Max: {lo_max_dd:.1f}%)", 
             color='blue', linewidth=1.5)
    plt.plot(ls_drawdown.index, ls_drawdown.values, label=f"Model Drawdown (Max: {ls_max_dd:.1f}%)", 
             color='red', linewidth=1.5)
    
    # Add title and labels
    plt.title("Exhibit 11: Drawdown comparison: Similarity model vs. long-only factor model\nDrawdown profile", 
              fontsize=14, fontweight='bold')
    plt.ylabel("% of capital", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    
    # Format y-axis to show percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    # Add horizontal line at 0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Highlight max drawdown points
    plt.scatter([lo_max_dd_date], [lo_max_dd], color='blue', s=100, zorder=5)
    plt.scatter([ls_max_dd_date], [ls_max_dd], color='red', s=100, zorder=5)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    try:
        from src.extensions.efficacy_score import get_reports_dir
    except ImportError:
        from extensions.efficacy_score import get_reports_dir
    reports_dir = get_reports_dir(Path(cfg["paths"]["reports_dir"]), cfg, "backtest")
    plt.savefig(reports_dir / "exhibit11_drawdown_comparison.png", dpi=150)
    # plt.show()  # (disabled: report is saved instead)
    
    print(f"Long-Only Max Drawdown: {lo_max_dd:.1f}% on {lo_max_dd_date.strftime('%Y-%m-%d')}")
    print(f"Long-Short Model Max Drawdown: {ls_max_dd:.1f}% on {ls_max_dd_date.strftime('%Y-%m-%d')}")

    # -----------------------------------------------------------------------------#
    # 9  Exhibit 12: Quantile Sweeps Plot
    # -----------------------------------------------------------------------------#
    print("Generating Exhibit 12: Quantile Sweeps Plot...")
    quantile_list = [2, 3, 4, 5, 10, 20]
    plt.figure(figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Prepare DataFrame to store cumulative returns and Sharpe ratios for each n_buckets
    cumrets_dict = {}
    sharpe_dict = {}
    for n in quantile_list:
        qrets = run_backtest(n_buckets=n, back_test_start_date=params["back_test_start_date"], forward_look_months=params["forward_look_months"], show_alignment_message=False)
        col = f"Q1_minus_Q{n}"
        metrics = compute_performance_metrics(qrets)
        if col in qrets.columns and col in metrics.index:
            # cumprod()-1 scaling (start at 0)
            cum = ((1 + qrets[col]).cumprod() - 1) * 100
            sr = metrics.loc[col, "AnnSharpe"]
            cumrets_dict[n] = cum
            sharpe_dict[n] = sr
        else:
            print(f"Column {col} not found for n_buckets={n}. Skipping.")
    # Align all cumulative returns by index
    cumrets_df = pd.DataFrame(cumrets_dict)
    sharpe_series = pd.Series(sharpe_dict, name="AnnSharpe")
    # Save to cache for later use - always overwrite to ensure parameter changes take effect
    # Note: These cache files are not used elsewhere in the codebase
    # cumrets_df.to_pickle(CACHE_DIR / "exhibit12_cumrets.pkl")
    # sharpe_series.to_pickle(CACHE_DIR / "exhibit12_sharpe.pkl")
    for i, n in enumerate(cumrets_df.columns):
        plt.plot(cumrets_df.index, cumrets_df[n], label=f"{n} quantiles, SR: {sharpe_series[n]:.2f}", color=colors[i % len(colors)])
    plt.title("Fama-French quantile sweeps (top minus bottom)")
    plt.ylabel("Cumulative return")
    plt.xlabel("Date")
    plt.legend(fontsize=11)
    plt.tight_layout()
    try:
        from src.extensions.efficacy_score import get_reports_dir
    except ImportError:
        from extensions.efficacy_score import get_reports_dir
    reports_dir = get_reports_dir(Path(cfg["paths"]["reports_dir"]), cfg, "backtest")
    plt.savefig(reports_dir / "exhibit12_quantile_sweeps.png", dpi=150)
    # plt.show()  # (disabled: report is saved instead)

    # Generate Exhibit 1
    print("\nGenerating Exhibit 1: Volatility Targeting Analysis...")
    vol_target = cfg["back_test"]["vol_target"]
    vol_window = cfg["back_test"]["vol_window"]

    exhibit1_results = generate_exhibit1(quintile_returns, vol_target, vol_window)
    
    # Save efficacy and multiplier series if extension is enabled
    if use_efficacy and 'efficacy_scores' in quintile_returns.attrs:
        try:
            from src.extensions.efficacy_score import get_reports_dir
        except ImportError:
            from extensions.efficacy_score import get_reports_dir
        reports_dir = get_reports_dir(Path(cfg["paths"]["reports_dir"]), cfg, "backtest")
        
        efficacy_scores = quintile_returns.attrs['efficacy_scores']
        efficacy_stds = quintile_returns.attrs['efficacy_stds']
        multipliers = quintile_returns.attrs['multipliers']
        
        # Save to CSV
        efficacy_df = pd.DataFrame({
            'efficacy_score': efficacy_scores,
            'efficacy_std': efficacy_stds,
            'multiplier': multipliers,
        })
        efficacy_df.to_csv(reports_dir / "efficacy_series.csv")
        print(f"\nSaved efficacy series to: {reports_dir / 'efficacy_series.csv'}")
        
        # Generate plot of efficacy and multiplier time series
        if efficacy_config.get('save_series', True):
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Top plot: Efficacy score
            ax1 = axes[0]
            ax1.plot(efficacy_scores.index, efficacy_scores.values, label='Efficacy Score', color='steelblue', linewidth=1.5)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
            ax1.fill_between(efficacy_scores.index, 
                            efficacy_scores.values - efficacy_stds.values,
                            efficacy_scores.values + efficacy_stds.values,
                            alpha=0.2, color='steelblue', label='±1 Std')
            ax1.set_ylabel('Efficacy Score (Correlation)', fontsize=12)
            ax1.set_title('Efficacy Score Time Series', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Multiplier
            ax2 = axes[1]
            ax2.plot(multipliers.index, multipliers.values, label='Exposure Multiplier', color='orange', linewidth=1.5)
            ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1, label='Neutral (0.5)')
            ax2.set_ylabel('Multiplier', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_title('Exposure Multiplier Time Series', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(reports_dir / "efficacy_series_plot.png", dpi=300, bbox_inches='tight')
            print(f"Saved efficacy series plot to: {reports_dir / 'efficacy_series_plot.png'}")
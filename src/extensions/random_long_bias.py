# random_long_bias.py | Random Long Bias Investigation
# ------------------------------------------------------------------------------
# Generates exhibit comparing Q1-Q5 spread with random long bias vs original strategy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config() -> dict:
    """Read parameters from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

cfg = load_config()
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Import backtest functions
try:
    from ..backtest.back_test import compute_performance_metrics, load_similarity_scores, load_factors
except ImportError:
    from backtest.back_test import compute_performance_metrics, load_similarity_scores, load_factors

# -----------------------------------------------------------------------------#
# 0  Random Long Bias Backtest
# -----------------------------------------------------------------------------#
def run_random_long_bias_backtest(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    random_seed: int = 42,
    long_bias: float = 0.75,
) -> pd.DataFrame:
    """
    Run backtest with random long bias signals instead of mean-return-based signals.
    
    Parameters:
    -----------
    n_buckets : int
        Number of buckets/quintiles for similarity-based grouping
    back_test_start_date : str
        Start date for the backtest period
    forward_look_months : int
        Number of months to look forward for returns
    similarity_window : int
        Rolling window size used for similarity calculation
    random_seed : int
        Random seed for reproducibility
    long_bias : float
        Probability of long signal (default 0.75 = 75% long bias)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing Q1-Q5 returns and Q1_minus_Q5 spread
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load data
    similarity_scores = load_similarity_scores(similarity_window=similarity_window)
    df_factors = load_factors()
    
    # Align indices
    min_date = max(similarity_scores.index.min(), df_factors.index.min())
    max_date = min(similarity_scores.index.max(), df_factors.index.max())
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
            dists = dists.dropna()
            if len(dists) == 0:
                bucket_returns.append(np.nan)
                continue
            # Rank distances, break ties deterministically
            ranks = dists.rank(method="first")
            sorted_idx = ranks.sort_values().index
            n_hist = len(sorted_idx)
            bucket_sizes = [n_hist // n_buckets + (1 if x < n_hist % n_buckets else 0) for x in range(n_buckets)]
            bucket_edges = np.cumsum([0] + bucket_sizes)
            bucket_start = bucket_edges[quintile - 1]
            bucket_end = bucket_edges[quintile]
            bucket_idx = sorted_idx[bucket_start:bucket_end]
            
            # Random long bias signals (per-factor)
            factor_signals = []
            for f in factor_names:
                if len(bucket_idx) == 0:
                    factor_signals.append(0)
                    continue
                # Generate random number between 0 and 1 for this factor
                random_val = np.random.random()
                # If random < long_bias: signal = +1 (long), else signal = -1 (short)
                signal = 1 if random_val < long_bias else -1
                factor_signals.append(signal)
            
            # Calculate forward-looking returns
            if forward_look_months == 1:
                forward_returns = df_factors.loc[T].values
            else:
                future_dates = dates[t:t + forward_look_months]
                if len(future_dates) >= forward_look_months:
                    future_factor_returns = df_factors.loc[future_dates].values
                    cumulative_returns = np.prod(1 + future_factor_returns, axis=0) - 1
                    forward_returns = cumulative_returns
                else:
                    forward_returns = np.full(len(factor_names), np.nan)
            
            realized = np.nansum(np.array(factor_signals) * forward_returns) / len(factor_names)
            bucket_returns.append(realized)
        all_bucket_returns[f"Q{quintile}"] = pd.Series(bucket_returns, index=dates)
    
    # Build DataFrame
    quintile_returns = pd.DataFrame(all_bucket_returns)
    # Long-short: Q1 - Q{n_buckets}
    quintile_returns[f"Q1_minus_Q{n_buckets}"] = quintile_returns["Q1"] - quintile_returns[f"Q{n_buckets}"]
    # Long-only benchmark
    quintile_returns["long_only"] = df_factors.mean(axis=1)
    
    # Restrict output to back_test_start_date onward
    quintile_returns = quintile_returns.loc[back_test_start_date:]
    return quintile_returns

# -----------------------------------------------------------------------------#
# 1  Random Long Bias Exhibit
# -----------------------------------------------------------------------------#
def generate_random_long_bias_exhibit(
    original_quintile_returns: pd.DataFrame,
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    random_seed: int = 42,
    long_bias: float = 0.75,
) -> None:
    """
    Generate exhibit comparing Q1-Q5 spread with random long bias vs original strategy.
    
    Parameters:
    -----------
    original_quintile_returns : pd.DataFrame
        DataFrame containing original Q1-Q5 returns from normal backtest
    n_buckets : int
        Number of buckets/quintiles
    back_test_start_date : str
        Start date for the backtest period
    forward_look_months : int
        Number of months to look forward for returns
    similarity_window : int
        Rolling window size used for similarity calculation
    random_seed : int
        Random seed for reproducibility
    long_bias : float
        Probability of long signal (default 0.75 = 75% long bias)
    """
    print("Generating Random Long Bias Exhibit...")
    
    # Run backtest with random long bias signals
    random_quintile_returns = run_random_long_bias_backtest(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window,
        random_seed=random_seed,
        long_bias=long_bias,
    )
    
    # Get Q1-Q5 spread for both strategies
    original_col = f"Q1_minus_Q{n_buckets}"
    random_col = f"Q1_minus_Q{n_buckets}"
    
    original_ls = original_quintile_returns[original_col].dropna()
    random_ls = random_quintile_returns[random_col].dropna()
    
    # Align indices for comparison
    common_dates = original_ls.index.intersection(random_ls.index)
    original_ls = original_ls.loc[common_dates]
    random_ls = random_ls.loc[common_dates]
    
    # Calculate performance metrics for both
    original_returns_df = pd.DataFrame({
        "original": original_ls,
        "long_only": original_quintile_returns.loc[common_dates, "long_only"]
    })
    random_returns_df = pd.DataFrame({
        "random": random_ls,
        "long_only": original_quintile_returns.loc[common_dates, "long_only"]
    })
    
    original_metrics = compute_performance_metrics(original_returns_df, risk_free=None)
    random_metrics = compute_performance_metrics(random_returns_df, risk_free=None)
    
    # Get SR and corr to LO
    orig_sr = original_metrics.loc["original", "AnnSharpe"]
    orig_corr = original_metrics.loc["original", "CorrWithLO"]
    rand_sr = random_metrics.loc["random", "AnnSharpe"]
    rand_corr = random_metrics.loc["random", "CorrWithLO"]
    
    # Calculate cumulative returns using same method as Exhibit 10
    cumrets_original = ((1 + original_ls).cumprod() - 1) * 100
    cumrets_random = ((1 + random_ls).cumprod() - 1) * 100
    
    # Create the plot (similar to right plot in Exhibit 10)
    plt.figure(figsize=(12, 7))
    
    # Plot both series
    plt.plot(cumrets_original.index, cumrets_original.values, 
             label=f"Original Q1-Q5 (SR: {orig_sr:.2f}, corr to LO: {orig_corr:.2f})",
             linewidth=2, color='blue')
    plt.plot(cumrets_random.index, cumrets_random.values, 
             label=f"Random Long Bias {int(long_bias*100)}% (SR: {rand_sr:.2f}, corr to LO: {rand_corr:.2f})",
             linewidth=2, color='red', linestyle='--')
    
    plt.title(f"Random Long Bias {int(long_bias*100)}% on Q1-Q5 Spread: Comparison with Original Strategy", 
              fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative return (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    reports_dir = REPORTS_DIR / "extensions" / "random_long_bias"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "exhibit_random_long_bias_comparison.png", dpi=150)
    # plt.show()  # (disabled: report is saved instead)
    
    print(f"Original Q1-Q5 SR: {orig_sr:.2f}, Corr to LO: {orig_corr:.2f}")
    print(f"Random Long Bias {int(long_bias*100)}% Q1-Q5 SR: {rand_sr:.2f}, Corr to LO: {rand_corr:.2f}")

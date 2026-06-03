# appendix.py | Section 8: Appendix Replication (A1, A2, A3)
# ------------------------------------------------------------------------------
# Generates Exhibit A1 (quintile performance for individual factors), 
# Exhibit A2 (Q1-Q5 long-short performance for individual factors), and
# Exhibit A3 (position signals over time for individual factors).
# Each exhibit shows all 6 factors (Market, Size, Value, Profitability, Investment, Momentum)
# on a single combined plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import warnings
from typing import Optional

# Import backtest functions
try:
    from .back_test import compute_performance_metrics, load_factors, load_similarity_scores
except ImportError:
    from back_test import compute_performance_metrics, load_factors, load_similarity_scores

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config() -> dict:
    """Read parameters from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

cfg = load_config()
REPORTS_DIR = Path(cfg["paths"]["reports_dir"]) / "backtest"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------#
# 1  Factor mapping
# -----------------------------------------------------------------------------#
FACTOR_MAPPING = {
    "MKT": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "MOM": "Momentum"
}

FACTOR_LIST = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

# -----------------------------------------------------------------------------#
# 2  Single-factor backtest
# -----------------------------------------------------------------------------#
def run_single_factor_backtest(
    factor_name: str,
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run backtest for a single factor.
    
    Parameters:
    -----------
    factor_name : str
        Factor name (MKT, SMB, HML, RMW, CMA, or MOM)
    n_buckets : int
        Number of buckets/quintiles
    back_test_start_date : str
        Start date for backtest
    forward_look_months : int
        Forward look months
    similarity_window : int
        Similarity window size
    verbose : bool
        Verbose output
        
    Returns:
    --------
    pd.DataFrame
        Quintile returns for the single factor
    """
    # Load data
    similarity_scores = load_similarity_scores(similarity_window=similarity_window)
    df_factors_all = load_factors()
    
    # Filter to single factor
    if factor_name not in df_factors_all.columns:
        raise ValueError(f"Factor {factor_name} not found in factor data")
    df_factors = df_factors_all[[factor_name]].copy()
    
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
    dates = df_factors.index
    
    for quintile in range(1, n_buckets + 1):
        bucket_returns = []
        for t, T in enumerate(dates):
            if t == 0:
                bucket_returns.append(np.nan)
                continue
            hist_months = dates[:t]
            dists = similarity_scores[T].loc[hist_months]
            dists = dists.dropna()
            if len(dists) == 0:
                bucket_returns.append(np.nan)
                continue
            # Rank distances
            ranks = dists.rank(method="first")
            sorted_idx = ranks.sort_values().index
            n_hist = len(sorted_idx)
            bucket_sizes = [n_hist // n_buckets + (1 if x < n_hist % n_buckets else 0) for x in range(n_buckets)]
            bucket_edges = np.cumsum([0] + bucket_sizes)
            bucket_start = bucket_edges[quintile - 1]
            bucket_end = bucket_edges[quintile]
            bucket_idx = sorted_idx[bucket_start:bucket_end]
            
            # Single-factor signal
            if len(bucket_idx) == 0:
                signal = 0
            else:
                mean_ret = df_factors.loc[bucket_idx, factor_name].mean()
                signal = 1 if mean_ret > 0 else -1
            
            # Forward returns
            if forward_look_months == 1:
                forward_return = df_factors.loc[T, factor_name]
            else:
                future_dates = dates[t:t + forward_look_months]
                if len(future_dates) >= forward_look_months:
                    future_returns = df_factors.loc[future_dates, factor_name].values
                    forward_return = np.prod(1 + future_returns) - 1
                else:
                    forward_return = np.nan
            
            realized = signal * forward_return
            bucket_returns.append(realized)
        all_bucket_returns[f"Q{quintile}"] = pd.Series(bucket_returns, index=dates)
    
    # Build DataFrame
    quintile_returns = pd.DataFrame(all_bucket_returns)
    # Long-short: Q1 - Q5
    quintile_returns[f"Q1_minus_Q5"] = quintile_returns["Q1"] - quintile_returns["Q5"]
    # Long-only benchmark (just the factor itself)
    quintile_returns["long_only"] = df_factors[factor_name]
    
    # Restrict output to back_test_start_date onward
    quintile_returns = quintile_returns.loc[back_test_start_date:]
    return quintile_returns

# -----------------------------------------------------------------------------#
# 3  Generate exhibits
# -----------------------------------------------------------------------------#
def generate_exhibit_a1(
    factor_returns_dict: dict,
    factor_metrics_dict: dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Generate Exhibit A1: Quintile performance for each factor (cumprod()-1 scaling, start at 0).
    
    Parameters:
    -----------
    factor_returns_dict : dict
        Dictionary mapping factor names to their quintile returns DataFrames
    factor_metrics_dict : dict
        Dictionary mapping factor names to their performance metrics DataFrames
    save_path : Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, factor_code in enumerate(FACTOR_LIST):
        ax = axes[idx]
        factor_name = FACTOR_MAPPING[factor_code]
        
        if factor_code not in factor_returns_dict:
            ax.text(0.5, 0.5, f"No data for {factor_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(factor_name, fontsize=12, fontweight='bold')
            continue
        
        quintile_returns = factor_returns_dict[factor_code]
        metrics = factor_metrics_dict[factor_code]
        
        # cumprod()-1 scaling (start at 0)
        cumrets_geometric = ((1 + quintile_returns).cumprod() - 1) * 100
        
        # Prepare legend labels with SR and corr
        legend_labels = []
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if q in metrics.index:
                sr = metrics.loc[q, "AnnSharpe"]
                corr = metrics.loc[q, "CorrWithLO"]
                legend_labels.append(f"{q} (SR: {sr:.2f}, corr: {corr:.2f})")
            else:
                legend_labels.append(q)
        
        if "long_only" in metrics.index:
            lo_sr = metrics.loc["long_only", "AnnSharpe"]
            legend_labels.append(f"LO (SR: {lo_sr:.2f})")
        else:
            legend_labels.append("LO")
        
        # Plot quintiles
        for i, q in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
            if q in cumrets_geometric.columns:
                ax.plot(cumrets_geometric.index, cumrets_geometric[q], 
                       label=legend_labels[i], linewidth=1.5)
        
        # Plot long-only
        if "long_only" in cumrets_geometric.columns:
            ax.plot(cumrets_geometric.index, cumrets_geometric["long_only"], 
                   label=legend_labels[-1], linestyle="--", color="gray", linewidth=2)
        
        ax.set_title(factor_name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Cumulative return", fontsize=10)
        if idx >= 3:  # Bottom row
            ax.set_xlabel("Date", fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Exhibit A1: Quintile Performance by Factor", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path is None:
        save_path = REPORTS_DIR / "exhibit_a1_quintile_performance.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Exhibit A1 to {save_path}")

def generate_exhibit_a2(
    factor_returns_dict: dict,
    factor_metrics_dict: dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Generate Exhibit A2: Q1-Q5 long-short performance for each factor (cumprod()-1 scaling, start at 0).
    
    Parameters:
    -----------
    factor_returns_dict : dict
        Dictionary mapping factor names to their quintile returns DataFrames
    factor_metrics_dict : dict
        Dictionary mapping factor names to their performance metrics DataFrames
    save_path : Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, factor_code in enumerate(FACTOR_LIST):
        ax = axes[idx]
        factor_name = FACTOR_MAPPING[factor_code]
        
        if factor_code not in factor_returns_dict:
            ax.text(0.5, 0.5, f"No data for {factor_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(factor_name, fontsize=12, fontweight='bold')
            continue
        
        quintile_returns = factor_returns_dict[factor_code]
        metrics = factor_metrics_dict[factor_code]
        
        # cumprod()-1 scaling (start at 0)
        cumrets_arithmetic = ((1 + quintile_returns).cumprod() - 1) * 100
        
        # Get Q1-Q5 metrics
        ls_col = "Q1_minus_Q5"
        if ls_col in cumrets_arithmetic.columns and ls_col in metrics.index:
            ls_sr = metrics.loc[ls_col, "AnnSharpe"]
            ls_corr = metrics.loc[ls_col, "CorrWithLO"]
            ls_label = f"Q1-Q5 (SR: {ls_sr:.2f}, corr: {ls_corr:.2f})"
        else:
            ls_label = "Q1-Q5"
        
        # Plot Q1-Q5
        if ls_col in cumrets_arithmetic.columns:
            ax.plot(cumrets_arithmetic.index, cumrets_arithmetic[ls_col], 
                   label=ls_label, linewidth=2, color='steelblue')
        
        ax.set_title(factor_name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Cumulative return", fontsize=10)
        if idx >= 3:  # Bottom row
            ax.set_xlabel("Date", fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.suptitle("Exhibit A2: Long-Short Performance (Q1-Q5) by Factor", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path is None:
        save_path = REPORTS_DIR / "exhibit_a2_long_short_performance.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Exhibit A2 to {save_path}")

def get_factor_positions(
    factor_name: str,
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
) -> pd.Series:
    """
    Extract position signals (long/short) for a single factor over time.
    Uses Q1 (most similar periods) to determine position.
    
    Parameters:
    -----------
    factor_name : str
        Factor name (MKT, SMB, HML, RMW, CMA, or MOM)
    n_buckets : int
        Number of buckets/quintiles
    back_test_start_date : str
        Start date for backtest
    forward_look_months : int
        Forward look months
    similarity_window : int
        Similarity window size
        
    Returns:
    --------
    pd.Series
        Position signals (+1 for long, -1 for short, 0 for neutral) indexed by date
    """
    # Load data
    similarity_scores = load_similarity_scores(similarity_window=similarity_window)
    df_factors_all = load_factors()
    
    # Filter to single factor
    if factor_name not in df_factors_all.columns:
        raise ValueError(f"Factor {factor_name} not found in factor data")
    df_factors = df_factors_all[[factor_name]].copy()
    
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
    
    dates = df_factors.index
    positions = []
    
    # Extract Q1 signals (most similar periods)
    for t, T in enumerate(dates):
        if t == 0:
            positions.append(np.nan)
            continue
        
        hist_months = dates[:t]
        dists = similarity_scores[T].loc[hist_months]
        dists = dists.dropna()
        
        if len(dists) == 0:
            positions.append(np.nan)
            continue
        
        # Rank distances
        ranks = dists.rank(method="first")
        sorted_idx = ranks.sort_values().index
        n_hist = len(sorted_idx)
        bucket_sizes = [n_hist // n_buckets + (1 if x < n_hist % n_buckets else 0) for x in range(n_buckets)]
        bucket_edges = np.cumsum([0] + bucket_sizes)
        bucket_start = bucket_edges[0]  # Q1
        bucket_end = bucket_edges[1]
        bucket_idx = sorted_idx[bucket_start:bucket_end]
        
        # Extract signal from Q1
        if len(bucket_idx) == 0:
            signal = 0
        else:
            mean_ret = df_factors.loc[bucket_idx, factor_name].mean()
            signal = 1 if mean_ret > 0 else -1
        
        positions.append(signal)
    
    positions_series = pd.Series(positions, index=dates)
    
    # Restrict to back_test_start_date onward
    positions_series = positions_series.loc[back_test_start_date:]
    return positions_series

def generate_exhibit_a3(
    factor_positions_dict: dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Generate Exhibit A3: Position signals over time for each factor.
    
    Parameters:
    -----------
    factor_positions_dict : dict
        Dictionary mapping factor names to their position signals Series
    save_path : Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, factor_code in enumerate(FACTOR_LIST):
        ax = axes[idx]
        factor_name = FACTOR_MAPPING[factor_code]
        
        if factor_code not in factor_positions_dict:
            ax.text(0.5, 0.5, f"No data for {factor_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(factor_name, fontsize=12, fontweight='bold')
            continue
        
        positions = factor_positions_dict[factor_code]
        
        # Calculate statistics
        valid_positions = positions.dropna()
        if len(valid_positions) > 0:
            long_pct = (valid_positions == 1).sum() / len(valid_positions) * 100
            short_pct = (valid_positions == -1).sum() / len(valid_positions) * 100
            
            # Create area plot
            ax.fill_between(positions.index, 0, positions, 
                          where=(positions >= 0), 
                          color='green', alpha=0.5, label=f'Long ({long_pct:.1f}%)')
            ax.fill_between(positions.index, 0, positions, 
                          where=(positions < 0), 
                          color='red', alpha=0.5, label=f'Short ({short_pct:.1f}%)')
            
            # Add step plot for clarity
            ax.plot(positions.index, positions, 'k-', linewidth=0.5, alpha=0.3)
        
        ax.set_title(factor_name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Position", fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Short', 'Neutral', 'Long'])
        if idx >= 3:  # Bottom row
            ax.set_xlabel("Date", fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='best')
    
    plt.suptitle("Exhibit A3: Position Signals Over Time by Factor", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path is None:
        save_path = REPORTS_DIR / "exhibit_a3_positions.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Exhibit A3 to {save_path}")

# -----------------------------------------------------------------------------#
# 4  Main execution
# -----------------------------------------------------------------------------#
def generate_appendix_exhibits(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
) -> None:
    """
    Generate Exhibits A1, A2, and A3 for all individual factors.
    
    Parameters:
    -----------
    n_buckets : int
        Number of buckets/quintiles
    back_test_start_date : str
        Start date for backtest
    forward_look_months : int
        Forward look months
    similarity_window : int
        Similarity window size
    """
    print("Generating Appendix Exhibits A1, A2, and A3...")
    print(f"Running backtests for {len(FACTOR_LIST)} individual factors...")
    
    factor_returns_dict = {}
    factor_metrics_dict = {}
    factor_positions_dict = {}
    
    # Run backtest for each factor
    for factor_code in FACTOR_LIST:
        factor_name = FACTOR_MAPPING[factor_code]
        print(f"  Processing {factor_name} ({factor_code})...")
        
        try:
            quintile_returns = run_single_factor_backtest(
                factor_name=factor_code,
                n_buckets=n_buckets,
                back_test_start_date=back_test_start_date,
                forward_look_months=forward_look_months,
                similarity_window=similarity_window,
                verbose=False
            )
            
            metrics = compute_performance_metrics(quintile_returns)
            
            # Extract positions
            positions = get_factor_positions(
                factor_name=factor_code,
                n_buckets=n_buckets,
                back_test_start_date=back_test_start_date,
                forward_look_months=forward_look_months,
                similarity_window=similarity_window,
            )
            
            factor_returns_dict[factor_code] = quintile_returns
            factor_metrics_dict[factor_code] = metrics
            factor_positions_dict[factor_code] = positions
            
        except Exception as e:
            print(f"    Error processing {factor_name}: {e}")
            continue
    
    # Generate exhibits
    print("\nGenerating Exhibit A1...")
    generate_exhibit_a1(factor_returns_dict, factor_metrics_dict)
    
    print("Generating Exhibit A2...")
    generate_exhibit_a2(factor_returns_dict, factor_metrics_dict)
    
    print("Generating Exhibit A3...")
    generate_exhibit_a3(factor_positions_dict)
    
    print("\n✅ Appendix exhibits generated successfully!")

# -----------------------------------------------------------------------------#
# 5  Wrapper function for CLI
# -----------------------------------------------------------------------------#
def generate_appendix(config: dict) -> None:
    """
    Wrapper function for CLI compatibility.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing back_test and similarity_score parameters
    """
    params = dict(
        n_buckets=config["back_test"].get("n_buckets", 5),
        back_test_start_date=config["back_test"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=config["back_test"].get("forward_look_months", 1),
        similarity_window=config["similarity_score"].get("similarity_window", 1),
    )
    generate_appendix_exhibits(**params)

# -----------------------------------------------------------------------------#
# 6  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Get parameters from config
    params = dict(
        n_buckets=cfg["back_test"].get("n_buckets", 5),
        back_test_start_date=cfg["back_test"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["back_test"].get("forward_look_months", 1),
        similarity_window=cfg["similarity_score"].get("similarity_window", 1),
    )
    
    generate_appendix_exhibits(**params)


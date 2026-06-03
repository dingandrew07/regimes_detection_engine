# equal_weighted_exhibit.py | Equal-Weighted Long-Only Across All Quintiles
# ------------------------------------------------------------------------------
# Generates exhibit showing equal-weighted long-only performance across all quintiles.

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
    from ..back_test import compute_performance_metrics
except ImportError:
    from back_test import compute_performance_metrics

# -----------------------------------------------------------------------------#
# 0  Equal-Weighted Long-Only Across All Quintiles
# -----------------------------------------------------------------------------#
def generate_equal_weighted_exhibit(quintile_returns: pd.DataFrame) -> None:
    """
    Generate Equal-weighted long-only performance across all quintiles.
    
    Parameters:
    -----------
    quintile_returns : pd.DataFrame
        DataFrame containing Q1-Q5 returns and long_only benchmark
    """
    print("Generating Equal-Weighted Long-Only Across All Quintiles...")
    
    # Create equal-weighted portfolio: 1/5 in each quintile
    equal_weighted_returns = (quintile_returns["Q1"] + quintile_returns["Q2"] + 
                              quintile_returns["Q3"] + quintile_returns["Q4"] + 
                              quintile_returns["Q5"]) / 5
    
    # Calculate performance metrics for equal-weighted portfolio
    # Need to include long_only column for correlation calculation
    equal_weighted_returns_df = pd.DataFrame({
        "equal_weighted": equal_weighted_returns,
        "long_only": quintile_returns["long_only"]
    })
    equal_weighted_metrics = compute_performance_metrics(equal_weighted_returns_df, risk_free=None)
    
    # Get SR and corr to LO
    ew_sr = equal_weighted_metrics.loc["equal_weighted", "AnnSharpe"]
    ew_corr = equal_weighted_metrics.loc["equal_weighted", "CorrWithLO"]
    
    # Calculate cumulative returns using same method as Exhibit 10
    cumrets_equal_weighted = ((1 + equal_weighted_returns).cumprod() - 1) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(cumrets_equal_weighted.index, cumrets_equal_weighted.values, 
             label=f"Equal-weighted (1/5 each quintile) (SR: {ew_sr:.2f}, corr to LO: {ew_corr:.2f})",
             linewidth=2)
    
    plt.title("Equal-Weighted Long-Only Performance Across All Quintiles", 
              fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative return (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save exhibit to reports/analysis/
    reports_dir = Path("reports/analysis")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "exhibit_equal_weighted_performance.png", dpi=150)
    # plt.show()  # (disabled: report is saved instead)
    
    print(f"Equal-weighted portfolio SR: {ew_sr:.2f}, Corr to LO: {ew_corr:.2f}")

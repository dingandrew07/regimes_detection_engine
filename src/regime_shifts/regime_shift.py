# regime_shift.py | Section 4 – Regime Shift Detection
# ------------------------------------------------------------------------------
# Implements EWMA-based regime shift detection using similarity scores.
# Calculates exponentially weighted moving averages for multiple lookback periods
# to identify regime transitions in economic state variables.
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Apply a professional plot style
sns.set_style("whitegrid")
warnings.filterwarnings('ignore', category=FutureWarning)

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
# 1  EWMA Calculation Functions
# -----------------------------------------------------------------------------#
def calculate_ewma_for_period(distances: pd.Series, lookback_months: int) -> float:
    """
    Calculate EWMA for a specific lookback period.
    
    Parameters
    ----------
    distances : pd.Series
        Series of distances from all previous months to current month T
        (already masked, so NaN values are excluded)
    lookback_months : int
        Number of months to look back for EWMA calculation
        
    Returns
    -------
    float
        Final EWMA value for this lookback period, or NaN if insufficient data
    """
    # Remove NaN values (masked months)
    valid_distances = distances.dropna()
    
    # Check if we have enough data
    if len(valid_distances) < lookback_months:
        return np.nan
    
    # Take the last n distances
    recent_distances = valid_distances.tail(lookback_months)
    
    # Calculate β = 1 - 1/n
    beta = 1 - (1 / lookback_months)
    
    # Calculate half-life for reference
    half_life = -np.log(2) / np.log(beta)
    
    # Initialize EWMA with first distance
    ewma = recent_distances.iloc[0]
    
    # Iterate through remaining distances
    for i in range(1, len(recent_distances)):
        distance = recent_distances.iloc[i]
        ewma = (1 - beta) * distance + beta * ewma
    
    return ewma


def calculate_ewma_regime_shifts(
    similarity_scores: pd.Series,
    lookback_periods: List[int] = [12, 24, 36, 48],
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate EWMA regime shifts for all months using multiple lookback periods.
    
    Parameters
    ----------
    similarity_scores : pd.Series
        Output from calculate_global_scores (Series of Series)
    lookback_periods : list of int, default [12, 24, 36, 48]
        List of lookback periods in months
    use_cache : bool, default True
        Whether to save the result to cache
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['1-year', '2-year', '3-year', '4-year', 'mean']
        and index as month-end dates. Early rows will have NaN for longer lookbacks.
    """
    cache_file = CACHE_DIR / "ewma_regime_shifts.pkl"
    
    print(f"Calculating EWMA regime shifts for lookback periods: {lookback_periods}")
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=similarity_scores.index)
    
    # Column names for the lookback periods
    column_names = [f"{period//12}-year" for period in lookback_periods]
    
    # Calculate EWMA for each lookback period
    for i, lookback_months in enumerate(lookback_periods):
        print(f"Processing {lookback_months}-month lookback period...")
        
        ewma_values = []
        for month_date, distances in similarity_scores.items():
            ewma_value = calculate_ewma_for_period(distances, lookback_months)
            ewma_values.append(ewma_value)
        
        results[column_names[i]] = ewma_values
    
    # Calculate mean EWMA across all periods (only where all exist)
    results['mean'] = results[column_names].mean(axis=1)
    
    # Cache the results
    if use_cache:
        joblib.dump(results, cache_file, compress=3)
        print(f"Cached EWMA regime shifts to {cache_file}")
    
    print(f"EWMA calculation complete. Shape: {results.shape}")
    print(f"Date range: {results.index.min()} to {results.index.max()}")
    
    return results


# -----------------------------------------------------------------------------#
# 2  Visualization Functions
# -----------------------------------------------------------------------------#
def create_exhibit_9_ewma(
    ewma_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create Exhibit 9: EWMA of global score visualization.
    
    Parameters
    ----------
    ewma_df : pd.DataFrame
        Output from calculate_ewma_regime_shifts
    save_path : str or Path, optional
        Path to save the plot. If None, uses reports/regime_shifts/exhibit9_ewma.png
    """
    if save_path is None:
        reports_dir = Path(cfg["paths"]["reports_dir"]) / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "exhibit9_ewma.png"
    else:
        save_path = Path(save_path)
    
    # Ensure reports directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find plot_start_date: first date where all 4 lookback EWMAs exist
    plot_start_date = ewma_df.dropna(subset=['1-year', '2-year', '3-year', '4-year']).index[0]
    print(f"Plot start date (first date with all 4 lookbacks): {plot_start_date}")
    
    # Filter DataFrame to only include dates >= plot_start_date
    plot_data = ewma_df.loc[plot_start_date:].copy()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and styles to match the reference
    line_specs = {
        '1-year': {'color': 'tab:blue', 'linestyle': '-', 'linewidth': 1.5},
        '2-year': {'color': 'tab:cyan', 'linestyle': '-', 'linewidth': 1.5},
        '3-year': {'color': 'tab:green', 'linestyle': '-', 'linewidth': 1.5},
        '4-year': {'color': 'tab:purple', 'linestyle': '-', 'linewidth': 1.5},
        'mean': {'color': 'tab:pink', 'linestyle': '--', 'linewidth': 1.5}
    }
    
    # Plot each EWMA series
    for col in ['1-year', '2-year', '3-year', '4-year', 'mean']:
        if col in plot_data.columns:
            specs = line_specs[col]
            plt.plot(plot_data.index, plot_data[col], 
                    label=col, 
                    color=specs['color'], 
                    linestyle=specs['linestyle'],
                    linewidth=specs['linewidth'],
                    alpha=0.8)
    
    # Customize the plot to match reference
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('X-year EWMA of global score', fontsize=12)
    plt.title('Plot of EWMAs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    
    # Add horizontal grid lines only
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(False, axis='x')
    
    # Format x-axis to show years (every 5 years)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.xticks(rotation=0)
    
    # Set x-axis to start at plot_start_date
    plt.xlim(left=plot_start_date)
        
    # Clean up the plot appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Exhibit 9 saved to: {save_path}")
    
    # Show plot (optional - comment out if running in headless mode)
    # plt.show()


# -----------------------------------------------------------------------------#
# 3  Main Execution Functions
# -----------------------------------------------------------------------------#
def run_regime_shift_analysis(
    use_cache: bool = True,
    create_visualization: bool = True
) -> pd.DataFrame:
    """
    Run the complete regime shift analysis pipeline.
    
    Parameters
    ----------
    use_cache : bool, default True
        Whether to use cached similarity scores and save EWMA results
    create_visualization : bool, default True
        Whether to create and save Exhibit 9
        
    Returns
    -------
    pd.DataFrame
        EWMA regime shifts DataFrame
    """
    print("Starting regime shift analysis...")
    
    # Load similarity scores
    try:
        similarity_scores = joblib.load(CACHE_DIR / "similarity_scores_window1.pkl")
        print(f"Loaded similarity scores: {len(similarity_scores)} months")
    except FileNotFoundError:
        print("Error: similarity_scores_window1.pkl not found. Run similarity_score.py first.")
        raise
    
    # Get lookback periods from config
    lookback_periods = cfg['regime_shift']['lookback_periods']
    
    # Calculate EWMA regime shifts
    ewma_df = calculate_ewma_regime_shifts(
        similarity_scores, 
        lookback_periods=lookback_periods,
        use_cache=use_cache
    )
    
    # Create visualization if requested
    if create_visualization:
        create_exhibit_9_ewma(ewma_df)
    
    return ewma_df


# -----------------------------------------------------------------------------#
# 4  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate EWMA regime shifts.")
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    args = parser.parse_args()
    
    # Run the analysis
    ewma_df = run_regime_shift_analysis(
        use_cache=not args.no_cache,
        create_visualization=not args.no_plot
    )
    
    print(f"\nRegime shift analysis complete!")
    print(f"Results shape: {ewma_df.shape}")
    print(f"Columns: {list(ewma_df.columns)}")
    print(f"Date range: {ewma_df.index.min()} to {ewma_df.index.max()}")
    
    # Show some sample results
    print(f"\nSample results (last 5 rows):")
    print(ewma_df.tail())

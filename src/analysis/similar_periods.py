# similar_periods.py | Similar Periods Visualization
# ------------------------------------------------------------------------------
# Creates visualizations showing similar historical periods to a target month.
# Highlights the most similar months and shows masked periods to prevent look-ahead bias.
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
from pathlib import Path
import yaml
from typing import Union, Optional
import warnings

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
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])

# -----------------------------------------------------------------------------#
# 1  Similar Periods Visualization
# -----------------------------------------------------------------------------#
def plot_similar_periods(
    target_month: str = None,
    similarity_scores_file: Optional[Path] = None,
    df_winsorized_file: Optional[Path] = CACHE_DIR / "df_winsorized.pkl",
    top_percentile: float = 0.15,
    mask_horizon: int = 36,
    similarity_window: int = 1,
    save_plot: bool = True,
    output_file: Optional[str] = None
) -> None:
    """
    Generate a plot showing similar historical periods to a target month.
    
    The plot displays:
    - Global score (average of state variables) over time as a dashed line
    - Blue highlighted regions for the top_percentile most similar months
    - Grey highlighted region for the masked period (prior 36 months)
    
    Parameters
    ----------
    target_month : str
        Target month in format "YYYY-MM" (e.g., "2009-01")
    similarity_scores_file : Path or None
        Path to cached similarity scores. If None, will try to load from default location
    df_winsorized_file : Path or None
        Path to cached winsorized state variables. If None, will try to load from default location
         top_percentile : float, default 0.15
         Top percentile of most similar months to highlight (0.15 = 15%)
     mask_horizon : int, default 36
         Number of months to mask before the target month
     similarity_window : int, default 1
         Rolling window size used for similarity calculation
     save_plot : bool, default True
         Whether to save the plot to the reports directory
     output_file : str or None, default None
         Custom filename for the plot. If None, uses default naming convention
    """
    
    # If target_month is not provided, read from config.yaml
    if target_month is None:
        # Try to read from config.yaml
        try:
            config_target = cfg.get('similar_periods', {}).get('target_month', None)
            if config_target is None:
                raise ValueError("No target_month specified in config.yaml under similar_periods:target_month")
            target_month = config_target
        except Exception as e:
            raise ValueError(f"No target_month provided and could not read from config.yaml under similar_periods:target_month: {e}")

    # Load similarity scores (masked for selection, unmasked for plotting)
    if similarity_scores_file is None:
        similarity_scores_file = CACHE_DIR / f"similarity_scores_window{similarity_window}.pkl"
    if not similarity_scores_file.exists():
        raise FileNotFoundError(f"Similarity scores file not found: {similarity_scores_file}. Run similarity_score.py with similarity_window={similarity_window} to generate it.")
    print(f"Loading similarity scores from {similarity_scores_file}")
    similarity_scores = joblib.load(similarity_scores_file)

    # Load unmasked similarity scores for plotting global score
    similarity_scores_unmasked_file = CACHE_DIR / f"similarity_scores_unmasked_window{similarity_window}.pkl"
    if not similarity_scores_unmasked_file.exists():
        raise FileNotFoundError(f"Unmasked similarity scores file not found: {similarity_scores_unmasked_file}. Run similarity_score.py with mask_horizon=0 and similarity_window={similarity_window} to generate it.")
    print(f"Loading unmasked similarity scores from {similarity_scores_unmasked_file}")
    similarity_scores_unmasked = joblib.load(similarity_scores_unmasked_file)

    # Load winsorized state variables for global score calculation
    if df_winsorized_file is None:
        df_winsorized_file = CACHE_DIR / "df_winsorized.pkl"
    
    if not df_winsorized_file.exists():
        raise FileNotFoundError(f"Winsorized data file not found: {df_winsorized_file}")
    
    print(f"Loading winsorized state variables from {df_winsorized_file}")
    df_winsorized = joblib.load(df_winsorized_file)
    
    # Convert target_month to month-end datetime (similarity scores use month-end dates)
    target_date = pd.to_datetime(target_month + "-01") + pd.offsets.MonthEnd(0)
    
    # Check if target date exists in similarity scores
    if target_date not in similarity_scores.index:
        available_dates = similarity_scores.index.strftime("%Y-%m").unique()
        raise ValueError(f"Target date {target_month} (month-end: {target_date.strftime('%Y-%m-%d')}) not found in similarity scores. "
                        f"Available dates: {available_dates[:10]}...")
    
    # Get distances for target date (masked for selection, unmasked for plotting)
    distances = similarity_scores[target_date]
    distances_unmasked = similarity_scores_unmasked[target_date]
    if not isinstance(distances, pd.Series):
        raise ValueError(f"Unexpected data type for distances: {type(distances)}")
    
    # Remove NaN values (masked months) and the target month itself
    valid_distances = distances.dropna()
    valid_distances = valid_distances[valid_distances.index != target_date]
    
    # Find the top_percentile most similar months
    n_similar = max(1, int(len(valid_distances) * top_percentile))
    similar_months = valid_distances.nsmallest(n_similar)
    
    print(f"Found {len(similar_months)} most similar months to {target_month}")
    print(f"Similarity threshold: {similar_months.max():.4f}")
    
    # Use the unmasked distances for the global score plot (continuous line)
    global_score = distances_unmasked.copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot global score as dashed line
    ax.plot(global_score.index, global_score.values, '--', color='black', 
            linewidth=1.5, label='Global score')
    
    # Set plot limits to start at first available date and end at target date
    first_date = global_score.index.min()
    ax.set_xlim(first_date, target_date)
    
    # Highlight similar months in blue
    for month in similar_months.index:
        # Find the month boundaries (start and end of month)
        month_start = month.replace(day=1)
        if month.month == 12:
            month_end = month.replace(year=month.year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = month.replace(month=month.month + 1, day=1) - pd.Timedelta(days=1)
        
        # Add blue highlighting for similar months
        ax.axvspan(month_start, month_end, alpha=0.3, facecolor='lightblue', 
                  hatch='////', edgecolor='none', label='Selected similar months' if month == similar_months.index[0] else "")
    
    # Highlight masked period in grey
    mask_start = target_date - pd.DateOffset(months=mask_horizon)
    ax.axvspan(mask_start, target_date, alpha=0.4, facecolor='grey', 
              label='Masked period')
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Global score', fontsize=12)
    ax.set_title(f'{target_month} global score', fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
        # Save the plot if requested
    if save_plot:
        # Always save the plot with a filename based on the target month and similarity window
        if output_file is None:
            output_file = f"similar_periods_{target_month.replace('-', '_')}_window{similarity_window}.png"
        
        # Save to similar periods subfolder
        similar_periods_dir = REPORTS_DIR / "similar periods"
        similar_periods_dir.mkdir(parents=True, exist_ok=True)
        output_path = similar_periods_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary for {target_month}:")
    print(f"Global score: {global_score[target_date]:.4f}")
    print(f"Number of similar months highlighted: {len(similar_months)}")
    print(f"Date range of similar months: {similar_months.index.min()} to {similar_months.index.max()}")
    print(f"Distance range: {similar_months.min():.4f} to {similar_months.max():.4f}")


# -----------------------------------------------------------------------------#
# 2  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    # Example usage
    try:
        # Load config to get default similarity_window
        cfg = load_config()
        similarity_window = cfg['similarity_score']['similarity_window']
        
        # Generate plot for January 2009
        plot_similar_periods(similarity_window=similarity_window)
        
        # You can also try other dates
        # plot_similar_periods("2008-10", similarity_window=similarity_window)
        # plot_similar_periods("2007-07", similarity_window=similarity_window)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run similarity_score.py first to generate similarity scores.")


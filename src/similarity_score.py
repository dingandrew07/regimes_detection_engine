# similarity_score.py | Section 3 – Similarity Score Calculation
# ------------------------------------------------------------------------------
# Implements distance-based similarity metric using Euclidean distance
# between months based on 7 state variables. Computes global scores for regime analysis.
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Union
import warnings

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
# 1  Similarity Score Calculation
# -----------------------------------------------------------------------------#
def calculate_global_scores(
    df_winsorized: pd.DataFrame,
    use_cache: bool = True,
    mask_horizon: int = 36,
    variable_weights: Optional[Dict[str, float]] = None,
    similarity_window: int = 1,
) -> pd.Series:
    """
    Compute weighted Euclidean distance similarity scores between months based on 7 state variables.
    
    Formula: d_Ti = sqrt(sum_{v=1 to V} w_v * (x_iv - x_Tv)^2)
    Where: d_Ti = distance between current month T and historical month i
           V = 7 state variables
           w_v = weight for variable v
           x_iv = value of variable v for month i (or rolling average if similarity_window > 1)
           x_Tv = value of variable v for month T (or rolling average if similarity_window > 1)

    Parameters
    ----------
    df_winsorized : DataFrame
        Winsorized z-scores of the 7 state variables (rows: dates, cols: variables)
    use_cache : bool, default True
        Whether to save the result to cache
    mask_horizon : int, default 36
        Number of months to mask before the current month (prevents look-ahead bias)
    variable_weights : dict or None, default None
        Dictionary mapping variable names to weights. If None, equal weights (1.0) are used.
        Example: {'Market': 1.5, 'Yield curve': 0.8, 'Oil': 1.2, ...}
    similarity_window : int, default 1
        Rolling window size for similarity calculation. 
        - 1: Use single-month snapshots (original behavior)
        - 3: Use 3-month rolling averages (quarterly rebalancing)
        - 6: Use 6-month rolling averages (semi-annual rebalancing)

    Returns
    -------
    similarity_scores : pd.Series
        Series of Series where:
        - Outer Series index = current month T
        - Inner Series index = historical month i  
        - Values = Weighted Euclidean distance d_Ti (smaller = more similar)
        - NaN for masked months (within mask_horizon of current month)
    """
    # Determine cache file based on mask_horizon and similarity_window
    if mask_horizon == 0:
        cache_file = CACHE_DIR / f"similarity_scores_unmasked_window{similarity_window}.pkl"
    else:
        cache_file = CACHE_DIR / f"similarity_scores_window{similarity_window}.pkl"

    print(f"Calculating similarity scores with {similarity_window}-month rolling window...")
    
    # Apply rolling window if similarity_window > 1
    if similarity_window > 1:
        print(f"Applying {similarity_window}-month rolling average to smooth state variables...")
        df_smoothed = df_winsorized.rolling(window=similarity_window, min_periods=similarity_window).mean()
        # Remove NaN values from the beginning (where we don't have enough data for rolling window)
        df_smoothed = df_smoothed.dropna()
        print(f"Smoothed data shape: {df_smoothed.shape} (removed {len(df_winsorized) - len(df_smoothed)} initial months)")
    else:
        df_smoothed = df_winsorized
    
    # Set up variable weights
    if variable_weights is None:
        # Equal weights for all variables
        weights = np.ones(len(df_smoothed.columns))
        print("Using equal weights for all variables")
    else:
        # Use provided weights, defaulting to 1.0 for missing variables
        weights = np.ones(len(df_smoothed.columns))
        for i, col in enumerate(df_smoothed.columns):
            if col in variable_weights:
                weights[i] = variable_weights[col]
        print(f"Using custom weights: {dict(zip(df_smoothed.columns, weights))}")
    
    # Convert to numpy array for efficient computation
    z = df_smoothed.values.astype(float)
    n_months, n_vars = z.shape
    dates = df_smoothed.index
    
    # Initialize output: Series of Series
    similarity_scores = pd.Series(dtype=object)
    
    # For each current month T, compute distances to all historical months i
    for T in range(n_months):
        current_month = dates[T]
        
        # Initialize distances for this month T
        distances = pd.Series(index=dates[:T+1], dtype=float)
        
        # Compute Euclidean distance to each historical month i (up to month T)
        for i in range(T + 1):
            historical_month = dates[i]
            
            # Only mask if mask_horizon > 0
            if mask_horizon > 0 and T - i <= mask_horizon:
                # Mask this month (set to NaN) to prevent look-ahead bias
                distances[historical_month] = np.nan
            else:
                # Weighted Euclidean distance formula: sqrt(sum_{v=1 to V} w_v * (x_iv - x_Tv)^2)
                diff = z[i, :] - z[T, :]  # x_iv - x_Tv for all variables v
                squared_diff = diff ** 2  # (x_iv - x_Tv)^2
                weighted_squared_diff = weights * squared_diff  # w_v * (x_iv - x_Tv)^2
                sum_weighted_squared = np.sum(weighted_squared_diff)  # sum_{v=1 to V} w_v * (x_iv - x_Tv)^2
                distance = np.sqrt(sum_weighted_squared)  # sqrt(sum_{v=1 to V} w_v * (x_iv - x_Tv)^2)
                
                distances[historical_month] = distance
        
        # Store the distance vector for current month T
        similarity_scores[current_month] = distances
    
    # Always overwrite the cache file
    if use_cache:
        joblib.dump(similarity_scores, cache_file, compress=3)
        print(f"Cached similarity scores to {cache_file}")
    
    # Display date range information
    print(f"Similarity scores date range: {similarity_scores.index.min()} to {similarity_scores.index.max()}")
    print(f"Total months with similarity scores: {len(similarity_scores)}")
    
    return similarity_scores


def get_similar_months(
    similarity_scores: pd.Series,
    target_date: Union[str, pd.Timestamp],
    n_similar: int = 10
) -> pd.Series:
    """
    Get the n most similar months to a target date.
    
    Parameters
    ----------
    similarity_scores : pd.Series
        Output from calculate_global_scores
    target_date : str or pd.Timestamp
        Date to find similar months for
    n_similar : int, default 10
        Number of most similar months to return
        
    Returns
    -------
    similar_months : pd.Series
        Series with dates as index and distances as values, sorted by similarity
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Get distances for target date
    if target_date in similarity_scores.index:
        distances = similarity_scores[target_date]
        if isinstance(distances, pd.Series):
            distances = distances.dropna()
            # Sort by distance (smaller = more similar) and exclude self
            filtered_distances = pd.Series(distances[distances.index != target_date])
            similar_months = filtered_distances.sort_values().head(n_similar)
            return similar_months
        else:
            raise ValueError(f"Unexpected data type for distances: {type(distances)}")
    else:
        raise ValueError(f"Target date {target_date} not found in similarity scores")


# -----------------------------------------------------------------------------#
# 2  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    parser = argparse.ArgumentParser(description="Calculate similarity scores with optional mask horizon and similarity window.")
    parser.add_argument('--mask_horizon', type=int, default=None, help='Number of months to mask (default: from config.yaml)')
    parser.add_argument('--similarity_window', type=int, default=None, help='Rolling window for similarity calculation (default: from config.yaml)')
    args = parser.parse_args()

    # Load state variables
    try:
        df_winsorized = joblib.load(CACHE_DIR / "df_winsorized.pkl")
        print(f"Loaded winsorized state variables: {df_winsorized.shape}")
        print(f"Variables: {list(df_winsorized.columns)}")
    except FileNotFoundError:
        print("Error: df_winsorized.pkl not found. Run state_variables.py first.")
        exit(1)
    
    # Load variable weights from config
    variable_weights = cfg['similarity_score']['variable_weights']
    
    # Determine mask_horizon
    if args.mask_horizon is not None:
        mask_horizon = args.mask_horizon
    else:
        mask_horizon = cfg['similarity_score']['mask_horizon']
    print(f"Using mask_horizon = {mask_horizon}")

    # Determine similarity_window
    if args.similarity_window is not None:
        similarity_window = args.similarity_window
    else:
        similarity_window = cfg['similarity_score']['similarity_window']
    print(f"Using similarity_window = {similarity_window}")

    # Calculate global similarity scores (cache file is handled inside the function)
    similarity_scores = calculate_global_scores(
        df_winsorized,
        use_cache=True,
        mask_horizon=mask_horizon,
        variable_weights=variable_weights,
        similarity_window=similarity_window
    )
    
    print(f"Computed similarity scores for {len(similarity_scores)} months")
    print(f"Date range: {similarity_scores.index.min()} to {similarity_scores.index.max()}")
    
    # Example: Find similar months to a recent date
    recent_date = similarity_scores.index[-12]  # 12 months ago
    similar_months = get_similar_months(similarity_scores, str(recent_date), n_similar=5)
    
    print(f"\nMost similar months to {recent_date}:")
    for date, distance in similar_months.items():
        print(f"  {date}: distance = {distance:.4f}")

# efficacy_score.py | Efficacy Score Extension
# ------------------------------------------------------------------------------
# Extension for confidence-based position scaling using efficacy scores.
# Computes correlation between predicted and realized factor returns to adjust exposure.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

# -----------------------------------------------------------------------------#
# 0  Extension Utilities
# -----------------------------------------------------------------------------#

def get_active_extensions(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get all active extensions and their parameters from config.
    
    Returns:
    -------
    dict: {extension_name: extension_config}
    """
    extensions_config = config.get('extensions', {})
    active = {}
    
    for name, value in extensions_config.items():
        if isinstance(value, dict) and value.get('enabled', False):
            # Extension is enabled with its own config dict
            active[name] = {k: v for k, v in value.items() if k != 'enabled'}
    
    return active


def get_extension_suffix(config: Dict[str, Any]) -> str:
    """
    Generate a suffix string from active extensions for folder naming.
    Example: "efficacy_score" or "" if no extensions
    """
    active = get_active_extensions(config)
    if not active:
        return ""
    return "_".join(sorted(active.keys()))


def get_reports_dir(base_dir: Path, config: Dict[str, Any], subfolder: str = "") -> Path:
    """
    Get reports directory with extension suffix appended.
    
    Parameters:
    -----------
    base_dir : Path
        Base reports directory from config
    config : dict
        Full config dict
    subfolder : str
        Subfolder name (e.g., "backtest", "state variables")
        
    Returns:
    --------
    Path to reports directory with extension suffix
    """
    active = get_active_extensions(config)
    if active:
        # For extensions, save to reports/extensions/{extension_name}
        extension_name = list(active.keys())[0]  # Use first active extension name
        if subfolder:
            base_path = base_dir / "extensions" / extension_name / subfolder
        else:
            base_path = base_dir / "extensions" / extension_name
    else:
        # No extensions, use default subfolder
        base_path = base_dir / subfolder if subfolder else base_dir
    
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

# -----------------------------------------------------------------------------#
# 1  Efficacy Score Extension
# -----------------------------------------------------------------------------#

def compute_efficacy_score(
    S_T: pd.Index,
    df_factors: pd.DataFrame,
    factor_names: list,
    bootstrap_iterations: int = 200,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute efficacy score for a given similar month pool S(T).
    
    Parameters:
    -----------
    S_T : pd.Index
        Fixed pool of similar months (Quintile 1 months) for month T
    df_factors : pd.DataFrame
        Factor returns DataFrame (already shifted, so row s contains returns from s to s+1)
    factor_names : list
        List of factor names (columns in df_factors)
    bootstrap_iterations : int, default 200
        Number of bootstrap iterations for stabilization
    random_seed : int or None, default None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple: (efficacy_score, efficacy_std)
        efficacy_score: Mean correlation across bootstrap iterations
        efficacy_std: Standard deviation of correlations across bootstrap iterations
    """
    if len(S_T) == 0:
        return np.nan, np.nan
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get predicted returns: average df_factors.loc[s] for all s in S(T)
    # Since df_factors is shifted, row s contains returns from s to s+1
    predicted_returns = df_factors.loc[S_T, factor_names].mean().values
    
    # Get realized returns: df_factors.loc[T] (returns from T to T+1)
    # Note: T is not passed here, but we need to get it from the calling context
    # For now, we'll compute the raw correlation, and the realized returns will be passed separately
    # Actually, wait - we need T to get realized returns. Let me reconsider the function signature.
    
    # Actually, let's compute efficacy differently - we need T passed in
    # Let me refactor this function
    
    # For bootstrap, we resample S(T) with replacement
    n_S_T = len(S_T)
    bootstrap_corrs = []
    
    for b in range(bootstrap_iterations):
        # Resample S(T) with replacement (same size)
        S_T_b = np.random.choice(S_T, size=n_S_T, replace=True)
        S_T_b = pd.Index(S_T_b)
        
        # Recompute predicted returns from bootstrap sample
        mu_hat_b = df_factors.loc[S_T_b, factor_names].mean().values
        
        # The realized returns will be passed from the calling context
        # For now, we'll just return the bootstrap samples and compute correlation externally
        bootstrap_corrs.append(mu_hat_b)
    
    # Return bootstrap samples - we'll compute correlation in the calling function
    return bootstrap_corrs


def compute_efficacy_with_realized(
    S_T: pd.Index,
    T: pd.Timestamp,
    df_factors: pd.DataFrame,
    factor_names: list,
    bootstrap_iterations: int = 200,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute efficacy score for month T using similar month pool S(T) and realized returns at T+1.
    
    Parameters:
    -----------
    S_T : pd.Index
        Fixed pool of similar months (Quintile 1 months) for month T
    T : pd.Timestamp
        Current month T
    df_factors : pd.DataFrame
        Factor returns DataFrame (already shifted, so row s contains returns from s to s+1)
    factor_names : list
        List of factor names (columns in df_factors)
    bootstrap_iterations : int, default 200
        Number of bootstrap iterations for stabilization
    random_seed : int or None, default None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple: (efficacy_score, efficacy_std)
        efficacy_score: Mean correlation across bootstrap iterations (range [-1, 1])
        efficacy_std: Standard deviation of correlations across bootstrap iterations
    """
    if len(S_T) == 0:
        return np.nan, np.nan
    
    # Check if T+1 exists in df_factors
    if T not in df_factors.index:
        return np.nan, np.nan
    
    # Get realized returns: df_factors.loc[T] (returns from T to T+1)
    r_real = df_factors.loc[T, factor_names].values
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # For bootstrap, we resample S(T) with replacement
    n_S_T = len(S_T)
    bootstrap_corrs = []
    
    for b in range(bootstrap_iterations):
        # Resample S(T) with replacement (same size)
        S_T_b = np.random.choice(S_T, size=n_S_T, replace=True)
        S_T_b = pd.Index(S_T_b)
        
        # Recompute predicted returns from bootstrap sample
        # Average df_factors.loc[s] for all s in S_T_b (returns at s+1)
        mu_hat_b = df_factors.loc[S_T_b, factor_names].mean().values
        
        # Compute cross-sectional correlation across the 6 factors
        # between mu_hat_b and r_real
        corr_b = np.corrcoef(mu_hat_b, r_real)[0, 1]
        
        # Handle NaN correlation (shouldn't happen, but just in case)
        if not np.isnan(corr_b):
            bootstrap_corrs.append(corr_b)
    
    if len(bootstrap_corrs) == 0:
        return np.nan, np.nan
    
    # Compute mean and std of bootstrap correlations
    efficacy_score = np.mean(bootstrap_corrs)
    efficacy_std = np.std(bootstrap_corrs)
    
    return efficacy_score, efficacy_std


def efficacy_to_multiplier(efficacy_score: float) -> float:
    """
    Map efficacy score (correlation, range [-1, 1]) to exposure multiplier (range [0, 1]).
    
    Parameters:
    -----------
    efficacy_score : float
        Efficacy score (correlation between predicted and realized factor returns)
        
    Returns:
    --------
    float
        Exposure multiplier in [0, 1]
    """
    if np.isnan(efficacy_score):
        return 0.5  # Default to neutral if efficacy is NaN
    
    # Map correlation [-1, 1] to multiplier [0, 1]
    # Formula: mult(T) = clip((efficacy(T) + 1) / 2, 0, 1)
    multiplier = np.clip((efficacy_score + 1) / 2, 0, 1)
    
    return multiplier

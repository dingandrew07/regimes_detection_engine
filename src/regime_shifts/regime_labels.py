# regime_labels.py | EWMA regime labeling (shared by alpha_by_regime and regime_gating)
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union

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


def load_config() -> dict:
    """Read parameters from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_regime_order(method: str, regime_labels: pd.Series) -> List[str]:
    """Return regime labels in display order, filtered to those present."""
    order = PHASE_REGIMES if method == 'phase' else LEGACY_REGIMES
    present = set(regime_labels.dropna().unique())
    return [r for r in order if r in present]


def load_ewma_regime_shifts(use_cache: bool = True, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load EWMA regime shifts from cache or calculate if not available.
    """
    if cache_dir is None:
        cache_dir = Path(load_config()["paths"]["cache_dir"])
    cache_file = cache_dir / "ewma_regime_shifts.pkl"

    if use_cache and cache_file.exists():
        print(f"Loading EWMA regime shifts from cache: {cache_file}")
        return joblib.load(cache_file)

    print("EWMA regime shifts not found in cache. Calculating...")
    try:
        from .regime_shift import run_regime_shift_analysis
    except ImportError:
        from regime_shift import run_regime_shift_analysis
    return run_regime_shift_analysis(use_cache=use_cache, create_visualization=False)


def get_phase_thresholds(
    mean_ewma: pd.Series,
    low_threshold_percentile: float,
    high_threshold_percentile: float,
) -> Dict[str, Union[float, pd.Series]]:
    """Return low/high threshold values and month-over-month EWMA delta for phase labeling."""
    if not (0 <= low_threshold_percentile <= 1):
        raise ValueError(
            f"low_threshold_percentile must be between 0 and 1 (got {low_threshold_percentile})"
        )
    if not (0 <= high_threshold_percentile <= 1):
        raise ValueError(
            f"high_threshold_percentile must be between 0 and 1 (got {high_threshold_percentile})"
        )
    if low_threshold_percentile >= high_threshold_percentile:
        raise ValueError("low_threshold_percentile must be less than high_threshold_percentile")

    return {
        "low": float(mean_ewma.quantile(low_threshold_percentile)),
        "high": float(mean_ewma.quantile(high_threshold_percentile)),
        "delta": mean_ewma.diff(),
    }


def label_regimes(
    ewma_df: pd.DataFrame,
    method: str = "percentile",
    threshold_percentile: Optional[float] = None,
    threshold_absolute: Optional[float] = None,
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None,
    verbose: bool = True,
) -> pd.Series:
    """
    Label each month by regime based on EWMA values and (for phase method) direction.
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

        thresholds = get_phase_thresholds(
            mean_ewma, low_threshold_percentile, high_threshold_percentile
        )
        low_threshold = thresholds["low"]
        high_threshold = thresholds["high"]
        delta = thresholds["delta"]

        labels = pd.Series('stable', index=mean_ewma.index, name='regime')
        labels[(mean_ewma > high_threshold) & (delta > 0)] = 'crisis_onset'
        labels[(mean_ewma > high_threshold) & (delta <= 0)] = 'resolution'
        labels[(mean_ewma > low_threshold) & (mean_ewma <= high_threshold)] = 'elevated'

        if verbose:
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
        if verbose:
            print(f"Regime labeling complete (method: {method}, percentile: {threshold_percentile:.3f}, threshold value: {threshold_value:.4f})")
        labels = pd.Series(
            index=mean_ewma.index,
            data=['transition' if val > threshold_value else 'stable' for val in mean_ewma.values],
            name='regime'
        )
        if verbose:
            print(f"  Transition months: {(labels == 'transition').sum()} ({(labels == 'transition').mean()*100:.1f}%)")
            print(f"  Stable months: {(labels == 'stable').sum()} ({(labels == 'stable').mean()*100:.1f}%)")

    elif method == "absolute":
        if threshold_absolute is None:
            raise ValueError("threshold_absolute parameter required when method='absolute'")
        threshold_value = threshold_absolute
        if verbose:
            print(f"Regime labeling complete (method: {method}, threshold: {threshold_value:.4f})")
        labels = pd.Series(
            index=mean_ewma.index,
            data=['transition' if val > threshold_value else 'stable' for val in mean_ewma.values],
            name='regime'
        )
        if verbose:
            print(f"  Transition months: {(labels == 'transition').sum()} ({(labels == 'transition').mean()*100:.1f}%)")
            print(f"  Stable months: {(labels == 'stable').sum()} ({(labels == 'stable').mean()*100:.1f}%)")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'phase', 'percentile', or 'absolute'")

    return labels


DEFAULT_AGE_BIN_EDGES = [0, 6, 12, 24]
DEFAULT_AGE_BIN_LABELS = ['0-6', '6-12', '12-24', '24+']


def compute_months_since_transition(labels: pd.Series) -> pd.Series:
    """
    Months since the regime label last changed (0 at the transition month).
    """
    if len(labels) == 0:
        return pd.Series(dtype=int, name='regime_age')

    ages = []
    current_age = 0
    prev_label = None

    for label in labels:
        if prev_label is None or label != prev_label:
            current_age = 0
        else:
            current_age += 1
        ages.append(current_age)
        prev_label = label

    return pd.Series(ages, index=labels.index, name='regime_age')


def bucket_regime_age(
    ages: pd.Series,
    bin_edges: Optional[List[int]] = None,
    bin_labels: Optional[List[str]] = None,
) -> pd.Series:
    """
    Map continuous regime age to ordered bucket labels.

    Default buckets: 0-6, 6-12, 12-24, 24+ months (left-closed, right-open).
    """
    if bin_edges is None:
        bin_edges = DEFAULT_AGE_BIN_EDGES
    if bin_labels is None:
        bin_labels = DEFAULT_AGE_BIN_LABELS

    if len(bin_labels) != len(bin_edges):
        raise ValueError("bin_labels must have the same length as bin_edges")

    extended_edges = list(bin_edges) + [np.inf]
    return pd.cut(
        ages,
        bins=extended_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )

# regime_gating.py | Regime-conditional exposure for backtest (Step 2)
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from .regime_labels import (
    REGIME_DISPLAY_NAMES,
    load_ewma_regime_shifts,
    label_regimes,
)

VALID_MODES = ("resolution_only", "exclude_crisis_onset")


def regime_exposure(regime: str, mode: str) -> float:
    """Return 1.0 (full exposure) or 0.0 (flat) for a regime label and gating mode."""
    if mode == "resolution_only":
        return 1.0 if regime == "resolution" else 0.0
    if mode == "exclude_crisis_onset":
        return 0.0 if regime == "crisis_onset" else 1.0
    raise ValueError(f"Unknown regime gating mode: {mode}. Use one of {VALID_MODES}")


def validate_regime_gating_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return regime_gating config subsection."""
    gating = config.get("regime_shifts", {}).get("regime_gating", {})
    if not gating.get("enabled", False):
        return gating

    mode = gating.get("mode", "resolution_only")
    if mode not in VALID_MODES:
        raise ValueError(f"regime_gating.mode must be one of {VALID_MODES} (got {mode!r})")

    alpha_config = config.get("regime_shifts", {}).get("alpha_by_regime", {})
    regime_method = alpha_config.get("regime_method", "phase")
    if regime_method != "phase":
        raise ValueError(
            "regime_gating requires regime_shifts.alpha_by_regime.regime_method: 'phase'"
        )
    return gating


def build_regime_exposure_series(
    config: Dict[str, Any],
    dates: pd.Index,
    use_cache: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build per-month exposure multipliers and regime labels aligned to backtest dates.

    Returns
    -------
    exposure : pd.Series
        0.0 or 1.0 per date
    regime_labels : pd.Series
        Phase regime label per date (NaN where unlabeled)
    """
    gating = validate_regime_gating_config(config)
    mode = gating.get("mode", "resolution_only")
    alpha_config = config.get("regime_shifts", {}).get("alpha_by_regime", {})

    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)
    labels = label_regimes(
        ewma_df,
        method="phase",
        low_threshold_percentile=alpha_config.get("low_threshold_percentile", 0.40),
        high_threshold_percentile=alpha_config.get("high_threshold_percentile", 0.75),
        verbose=True,
    )

    exposure = pd.Series(0.0, index=dates, name="regime_exposure")
    aligned_labels = labels.reindex(dates)

    for date in dates:
        regime = aligned_labels.get(date)
        if pd.isna(regime):
            exposure.loc[date] = 0.0
        else:
            exposure.loc[date] = regime_exposure(regime, mode)

    n_traded = int((exposure == 1.0).sum())
    n_flat = int((exposure == 0.0).sum())
    pct = n_traded / len(dates) * 100 if len(dates) else 0.0
    print(f"Regime gating enabled (mode: {mode})")
    print(f"  Full exposure months: {n_traded} ({pct:.1f}%)")
    print(f"  Flat (cash) months: {n_flat} ({100 - pct:.1f}%)")

    return exposure, aligned_labels


def get_gated_backtest_reports_dir(base_reports_dir: Path, mode: str) -> Path:
    """Reports directory for gated backtest exhibits."""
    path = base_reports_dir / "regime_shifts" / "gated_backtest" / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_backtest_reports_dir(config: Dict[str, Any], base_reports_dir: Path) -> Path:
    """
    Resolve backtest report output directory based on active extensions/gating.
    Priority: efficacy extension > regime gating > default backtest folder.
    """
    efficacy = config.get("extensions", {}).get("efficacy_score", {})
    if isinstance(efficacy, dict) and efficacy.get("enabled", False):
        try:
            from src.extensions.efficacy_score import get_reports_dir
        except ImportError:
            from extensions.efficacy_score import get_reports_dir
        return get_reports_dir(base_reports_dir, config, "backtest")

    gating = config.get("regime_shifts", {}).get("regime_gating", {})
    if isinstance(gating, dict) and gating.get("enabled", False):
        mode = gating.get("mode", "resolution_only")
        return get_gated_backtest_reports_dir(base_reports_dir, mode)

    path = base_reports_dir / "backtest"
    path.mkdir(parents=True, exist_ok=True)
    return path

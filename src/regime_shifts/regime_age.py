# regime_age.py | Alpha vs Regime Age
# ------------------------------------------------------------------------------
# Tests whether alpha increases as regimes age by:
# 1. Computing months since the last regime transition for each month
# 2. Grouping observations into age buckets (0-6, 6-12, 12-24, 24+ months)
# 3. Plotting mean return and Sharpe by regime age
# 4. Testing the hypothesis that alpha rises with regime age (Spearman + trend)

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, linregress, spearmanr

try:
    from .regime_labels import (
        DEFAULT_AGE_BIN_EDGES,
        DEFAULT_AGE_BIN_LABELS,
        bucket_regime_age,
        compute_months_since_transition,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )
except ImportError:
    from regime_labels import (
        DEFAULT_AGE_BIN_EDGES,
        DEFAULT_AGE_BIN_LABELS,
        bucket_regime_age,
        compute_months_since_transition,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )

cfg = load_config()
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_backtest_returns(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
) -> pd.DataFrame:
    """Load or calculate backtest returns."""
    try:
        from ..backtest.back_test import run_backtest
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backtest.back_test import run_backtest
    return run_backtest(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window,
        show_alignment_message=False,
    )


def _bucket_midpoints(bin_edges: List[int], bin_labels: List[str]) -> Dict[str, float]:
    """Representative age (months) for each bucket, used in trend tests."""
    midpoints = {}
    for i, label in enumerate(bin_labels):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i + 1 < len(bin_edges) else lo + 12
        midpoints[label] = (lo + hi) / 2
    return midpoints


def compute_age_bucket_metrics(
    returns: pd.Series,
    age_buckets: pd.Series,
    bucket_order: List[str],
    full_sample_vol: Optional[float] = None,
) -> pd.DataFrame:
    """Mean return, Sharpe, and sample size for each regime-age bucket."""
    if full_sample_vol is None:
        full_sample_vol = returns.std()

    rows = []
    for bucket in bucket_order:
        mask = age_buckets == bucket
        bucket_returns = returns[mask].dropna()
        n_months = len(bucket_returns)

        if n_months == 0:
            rows.append({
                'age_bucket': bucket,
                'mean_return': np.nan,
                'ann_sharpe': np.nan,
                'n_months': 0,
            })
            continue

        mean_ret = bucket_returns.mean()
        if full_sample_vol > 0:
            ann_sharpe = mean_ret / full_sample_vol * np.sqrt(12)
        else:
            ann_sharpe = np.nan

        rows.append({
            'age_bucket': bucket,
            'mean_return': mean_ret,
            'ann_sharpe': ann_sharpe,
            'n_months': n_months,
        })

    return pd.DataFrame(rows).set_index('age_bucket').reindex(bucket_order)


def test_alpha_increases_with_age(
    returns: pd.Series,
    regime_age: pd.Series,
    bucket_metrics: pd.DataFrame,
    bucket_order: List[str],
    bin_edges: List[int],
) -> Dict[str, float]:
    """
    Test whether alpha increases with regime age.

    Returns Spearman correlation (continuous age), bucket-trend slope,
    and Kruskal-Wallis p-value across buckets.
    """
    aligned = pd.DataFrame({
        'returns': returns,
        'age': regime_age,
    }).dropna()

    if len(aligned) < 3:
        return {
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'trend_slope': np.nan,
            'trend_p': np.nan,
            'trend_r2': np.nan,
            'kruskal_p': np.nan,
        }

    rho, spearman_p = spearmanr(aligned['age'], aligned['returns'])

    midpoints = _bucket_midpoints(bin_edges, bucket_order)
    x = np.array([midpoints[b] for b in bucket_order])
    y = bucket_metrics.loc[bucket_order, 'mean_return'].values

    valid = ~np.isnan(y)
    if valid.sum() >= 2:
        trend = linregress(x[valid], y[valid])
        trend_slope, trend_p, trend_r2 = trend.slope, trend.pvalue, trend.rvalue ** 2
    else:
        trend_slope, trend_p, trend_r2 = np.nan, np.nan, np.nan

    age_buckets = bucket_regime_age(aligned['age'], bin_edges=bin_edges, bin_labels=bucket_order)
    groups = [aligned.loc[age_buckets == b, 'returns'].values for b in bucket_order]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        kruskal_p = float(kruskal(*groups).pvalue)
    else:
        kruskal_p = np.nan

    return {
        'spearman_rho': float(rho),
        'spearman_p': float(spearman_p),
        'trend_slope': float(trend_slope),
        'trend_p': float(trend_p),
        'trend_r2': float(trend_r2),
        'kruskal_p': kruskal_p,
    }


def create_regime_age_exhibit(
    bucket_metrics: pd.DataFrame,
    bucket_order: List[str],
    hypothesis: Dict[str, float],
    strategy: str,
    save_path: Optional[Path] = None,
) -> None:
    """Two-panel exhibit: mean monthly return and Sharpe by regime-age bucket."""
    if save_path is None:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "regime_age_exhibit.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, (ax_return, ax_sharpe) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        gridspec_kw={'width_ratios': [1.2, 1]},
    )

    x = np.arange(len(bucket_order))
    colors = sns.color_palette("Blues_d", len(bucket_order))

    mean_rets = bucket_metrics.loc[bucket_order, 'mean_return']
    sharpes = bucket_metrics.loc[bucket_order, 'ann_sharpe']
    n_months = bucket_metrics.loc[bucket_order, 'n_months']

    ax_return.bar(
        x, mean_rets * 100,
        color=colors,
        edgecolor='white',
        linewidth=0.8,
    )
    ax_return.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_return.set_xticks(x)
    ax_return.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bucket_order, n_months)])
    ax_return.set_xlabel('Months Since Last Transition')
    ax_return.set_ylabel('Mean Monthly Return (%)')
    ax_return.set_title('Alpha by Regime Age', fontsize=12, fontweight='bold')

    ax_sharpe.bar(
        x, sharpes,
        color=colors,
        edgecolor='white',
        linewidth=0.8,
    )
    ax_sharpe.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_sharpe.set_xticks(x)
    ax_sharpe.set_xticklabels(bucket_order)
    ax_sharpe.set_xlabel('Months Since Last Transition')
    ax_sharpe.set_ylabel('Annualized Sharpe')
    ax_sharpe.set_title('Sharpe by Regime Age', fontsize=12, fontweight='bold')

    rho = hypothesis['spearman_rho']
    sp = hypothesis['spearman_p']
    tp = hypothesis['trend_p']
    kp = hypothesis['kruskal_p']

    def _fmt_p(p: float) -> str:
        if np.isnan(p):
            return 'n/a'
        if p < 0.001:
            return '<0.001'
        return f'{p:.3f}'

    supports = (
        not np.isnan(rho) and rho > 0
        and not np.isnan(sp) and sp < 0.05
    )
    verdict = 'Supported' if supports else 'Not supported'

    stats_text = (
        f"Hypothesis: alpha increases as regimes age ({verdict})\n"
        f"Spearman rho = {rho:.3f}, p = {_fmt_p(sp)}\n"
        f"Bucket trend slope = {hypothesis['trend_slope']*100:.4f}%/mo, p = {_fmt_p(tp)}\n"
        f"Kruskal-Wallis across buckets: p = {_fmt_p(kp)}"
    )

    fig.suptitle(
        f'Regime Age Analysis — {strategy}',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.text(
        0.5, -0.02, stats_text,
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc'),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exhibit saved to: {save_path}")
    plt.close()


def _print_interpretation(
    bucket_metrics: pd.DataFrame,
    bucket_order: List[str],
    hypothesis: Dict[str, float],
) -> None:
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    young = bucket_order[0]
    old = bucket_order[-1]
    young_ret = bucket_metrics.loc[young, 'mean_return']
    old_ret = bucket_metrics.loc[old, 'mean_return']

    if not np.isnan(young_ret) and not np.isnan(old_ret):
        if old_ret > young_ret:
            print(f"  -> Mean return higher in oldest bucket ({old}: {old_ret:.4%}) "
                  f"than youngest ({young}: {young_ret:.4%})")
        else:
            print(f"  -> Mean return not higher in oldest bucket ({old}: {old_ret:.4%}) "
                  f"vs youngest ({young}: {young_ret:.4%})")

    rho = hypothesis['spearman_rho']
    sp = hypothesis['spearman_p']
    if not np.isnan(rho) and not np.isnan(sp):
        direction = 'positive' if rho > 0 else 'negative'
        sig = 'significant' if sp < 0.05 else 'not significant'
        print(f"  -> Spearman correlation ({direction}, {sig}): rho={rho:.3f}, p={sp:.4f}")


def run_regime_age_analysis(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    regime_method: str = "phase",
    regime_threshold_percentile: Optional[float] = None,
    regime_threshold_absolute: Optional[float] = None,
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None,
    age_bin_edges: Optional[List[int]] = None,
    age_bin_labels: Optional[List[str]] = None,
    strategy: str = "Q1_minus_Q5",
    use_cache: bool = True,
    create_exhibit: bool = True,
) -> pd.DataFrame:
    """
    Run regime-age analysis end-to-end.

    Returns a DataFrame of performance metrics by age bucket.
    """
    if age_bin_edges is None:
        age_bin_edges = DEFAULT_AGE_BIN_EDGES
    if age_bin_labels is None:
        age_bin_labels = DEFAULT_AGE_BIN_LABELS

    bucket_order = age_bin_labels

    print("=" * 70)
    print("Regime Age Analysis")
    print("Hypothesis: Does alpha increase as regimes age?")
    print("=" * 70)

    print("\n1. Loading EWMA regime shifts...")
    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)

    print("\n2. Labeling regimes and computing regime age...")
    regime_labels = label_regimes(
        ewma_df,
        method=regime_method,
        threshold_percentile=regime_threshold_percentile,
        threshold_absolute=regime_threshold_absolute,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile,
    )
    regime_age = compute_months_since_transition(regime_labels)

    print("\n3. Loading backtest returns...")
    backtest_returns = load_backtest_returns(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window,
    )

    if strategy not in backtest_returns.columns:
        available = ', '.join(backtest_returns.columns)
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")

    print("\n4. Aligning data...")
    common_dates = backtest_returns.index.intersection(regime_age.index)
    returns = backtest_returns.loc[common_dates, strategy]
    regime_age_aligned = regime_age.loc[common_dates]
    age_buckets = bucket_regime_age(
        regime_age_aligned,
        bin_edges=age_bin_edges,
        bin_labels=age_bin_labels,
    )

    print(f"   Aligned {len(common_dates)} months")
    print(f"   Date range: {common_dates.min()} to {common_dates.max()}")

    print("\n5. Computing metrics by regime-age bucket...")
    bucket_metrics = compute_age_bucket_metrics(
        returns, age_buckets, bucket_order, full_sample_vol=returns.std()
    )

    print("\n6. Testing hypothesis...")
    hypothesis = test_alpha_increases_with_age(
        returns, regime_age_aligned, bucket_metrics, bucket_order, age_bin_edges
    )

    print("\n" + "=" * 70)
    print(f"SUMMARY: Alpha by Regime Age ({strategy})")
    print("=" * 70)
    display = bucket_metrics.copy()
    display['mean_return'] = (display['mean_return'] * 100).round(3)
    display['ann_sharpe'] = display['ann_sharpe'].round(3)
    print("\n" + display.to_string())
    print(f"\nSpearman rho: {hypothesis['spearman_rho']:.4f} (p={hypothesis['spearman_p']:.4f})")
    print(f"Bucket trend p: {hypothesis['trend_p']:.4f}")
    print(f"Kruskal-Wallis p: {hypothesis['kruskal_p']:.4f}")

    _print_interpretation(bucket_metrics, bucket_order, hypothesis)

    if create_exhibit:
        print("\n7. Creating exhibit...")
        create_regime_age_exhibit(
            bucket_metrics, bucket_order, hypothesis, strategy
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return bucket_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze alpha vs regime age.")
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--method', type=str, default=None,
                        choices=['phase', 'percentile', 'absolute'],
                        help='Regime labeling method (overrides config)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Backtest strategy column (default: Q1_minus_Q5)')
    parser.add_argument('--low-threshold-percentile', type=float, default=None)
    parser.add_argument('--high-threshold-percentile', type=float, default=None)
    parser.add_argument('--threshold-percentile', type=float, default=None)
    parser.add_argument('--threshold-absolute', type=float, default=None)
    args = parser.parse_args()

    alpha_config = cfg.get("regime_shifts", {}).get("alpha_by_regime", {})
    age_config = cfg.get("regime_shifts", {}).get("regime_age", {})

    params = dict(
        n_buckets=cfg["backtest"].get("n_buckets", 5),
        back_test_start_date=cfg["backtest"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["backtest"].get("forward_look_months", 1),
        similarity_window=cfg["state_variables"]["similarity_score"].get("similarity_window", 1),
        regime_method=args.method if args.method is not None else alpha_config.get("regime_method", "phase"),
        regime_threshold_percentile=(
            args.threshold_percentile if args.threshold_percentile is not None
            else alpha_config.get("regime_threshold_percentile")
        ),
        regime_threshold_absolute=(
            args.threshold_absolute if args.threshold_absolute is not None
            else alpha_config.get("regime_threshold_absolute")
        ),
        low_threshold_percentile=(
            args.low_threshold_percentile if args.low_threshold_percentile is not None
            else alpha_config.get("low_threshold_percentile")
        ),
        high_threshold_percentile=(
            args.high_threshold_percentile if args.high_threshold_percentile is not None
            else alpha_config.get("high_threshold_percentile")
        ),
        age_bin_edges=age_config.get("age_bin_edges", DEFAULT_AGE_BIN_EDGES),
        age_bin_labels=age_config.get("age_bin_labels", DEFAULT_AGE_BIN_LABELS),
        strategy=args.strategy if args.strategy is not None else age_config.get("strategy", "Q1_minus_Q5"),
        use_cache=not args.no_cache,
    )

    run_regime_age_analysis(**params)

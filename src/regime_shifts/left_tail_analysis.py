# left_tail_analysis.py | Left-Tail Event Analysis
# ------------------------------------------------------------------------------
# Investigates whether regime transitions following configured left-tail events
# generate disproportionate alpha compared to gradual transitions.
#
# 1. Load known left-tail events from config (regime_shifts.known_events)
# 2. Classify EWMA regime transitions as crisis_linked vs gradual
# 3. Test post-crisis transition alpha (including pre-stabilization windows)
# 4. Compare crisis-linked vs gradual episodes with statistical tests

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu, spearmanr

try:
    from .known_events import (
        find_anchor_event,
        known_events_to_dataframe,
        load_known_events,
        parse_event_date,
    )
    from .regime_labels import (
        DEFAULT_AGE_BIN_EDGES,
        DEFAULT_AGE_BIN_LABELS,
        REGIME_COLORS,
        bucket_regime_age,
        compute_months_since_transition,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )
except ImportError:
    from known_events import (
        find_anchor_event,
        known_events_to_dataframe,
        load_known_events,
        parse_event_date,
    )
    from regime_labels import (
        DEFAULT_AGE_BIN_EDGES,
        DEFAULT_AGE_BIN_LABELS,
        REGIME_COLORS,
        bucket_regime_age,
        compute_months_since_transition,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )

cfg = load_config()
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])
DATA_DIR = Path(cfg["paths"]["data_dir"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GRADUAL_REGIMES = {"stable", "elevated"}


def load_spx_monthly_levels() -> pd.Series:
    """Load SPX monthly price levels for exhibit chart context only."""
    market_file = cfg["state_variables"]["data_sources"]["market_file"]
    path = DATA_DIR / market_file
    df = pd.read_excel(path, index_col=0, parse_dates=True)
    series = df["SPX_monthly"].squeeze().resample("ME").last()
    end_date = cfg["state_variables"]["end_date"]
    return series.loc[:end_date].dropna()


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


def find_label_changes(labels: pd.Series) -> pd.DataFrame:
    """Return DataFrame of transition dates with from/to regime labels."""
    rows = []
    prev_label = None
    for date, label in labels.items():
        if prev_label is not None and label != prev_label:
            rows.append({
                "transition_date": date,
                "from_regime": prev_label,
                "to_regime": label,
            })
        prev_label = label
    return pd.DataFrame(rows)


def _episode_path(labels: pd.Series, start_idx: int, end_idx: int) -> List[str]:
    """Regime labels visited from start_idx through end_idx (inclusive)."""
    return list(labels.iloc[start_idx : end_idx + 1].unique())


def _has_crisis_regime_in_window(
    labels: pd.Series,
    start_date: pd.Timestamp,
    window_months: int,
) -> Tuple[bool, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Check if crisis_onset or resolution appears within window after start_date."""
    window_end = start_date + pd.DateOffset(months=window_months)
    window = labels.loc[(labels.index >= start_date) & (labels.index <= window_end)]

    first_onset = None
    first_resolution = None
    for date, label in window.items():
        if label == "crisis_onset" and first_onset is None:
            first_onset = date
        if label == "resolution" and first_resolution is None:
            first_resolution = date

    has_crisis = first_onset is not None or first_resolution is not None
    return has_crisis, first_onset, first_resolution


def build_transition_episodes(
    labels: pd.Series,
    events: pd.DataFrame,
    lt_config: dict,
) -> pd.DataFrame:
    """
    Classify each regime label change as crisis_linked or gradual.

    crisis_linked: left-tail event in lookback window AND episode enters
    crisis_onset/resolution within post_event_window_months.
    gradual: stable/elevated only path OR no left-tail event in lookback.
    """
    lookback = lt_config.get("transition_lookback_months", 6)
    post_window = lt_config.get("post_event_window_months", 12)

    transitions = find_label_changes(labels)
    if transitions.empty:
        return pd.DataFrame(columns=[
            "transition_date", "from_regime", "to_regime", "episode_type",
            "anchor_event_date", "first_crisis_onset", "first_resolution", "episode_end",
        ])

    label_index = labels.index
    rows = []
    for i, trans in transitions.iterrows():
        t_date = trans["transition_date"]
        t_loc = label_index.get_loc(t_date)

        if i + 1 < len(transitions):
            episode_end = transitions.iloc[i + 1]["transition_date"]
            end_loc = label_index.get_loc(episode_end) - 1
        else:
            episode_end = label_index[-1]
            end_loc = len(label_index) - 1

        end_loc = max(end_loc, t_loc)
        path = _episode_path(labels, t_loc, end_loc)
        path_set = set(path)

        anchor = find_anchor_event(t_date, events, lookback)
        has_crisis, first_onset, first_resolution = _has_crisis_regime_in_window(
            labels, t_date, post_window
        )

        only_gradual_path = path_set.issubset(GRADUAL_REGIMES)
        if anchor is not None and has_crisis:
            episode_type = "crisis_linked"
        elif only_gradual_path or anchor is None:
            episode_type = "gradual"
        else:
            episode_type = "gradual"

        rows.append({
            "transition_date": t_date,
            "from_regime": trans["from_regime"],
            "to_regime": trans["to_regime"],
            "episode_type": episode_type,
            "anchor_event_date": anchor,
            "first_crisis_onset": first_onset,
            "first_resolution": first_resolution,
            "episode_end": episode_end,
        })

    return pd.DataFrame(rows)


def _slice_episode_returns(
    returns: pd.Series,
    transition_date: pd.Timestamp,
    episode_end: pd.Timestamp,
    max_months: Optional[int] = None,
) -> pd.Series:
    """Returns from transition_date through episode_end (or max_months)."""
    mask = (returns.index >= transition_date) & (returns.index <= episode_end)
    sliced = returns.loc[mask]
    if max_months is not None and len(sliced) > max_months:
        sliced = sliced.iloc[:max_months]
    return sliced


def compute_window_metrics(
    returns: pd.Series,
    episodes: pd.DataFrame,
    windows: List[int],
    full_sample_vol: Optional[float] = None,
) -> pd.DataFrame:
    """Mean return, Sharpe, and n_months by episode_type and post-transition window."""
    if full_sample_vol is None:
        full_sample_vol = returns.std()

    rows = []
    for episode_type in ["crisis_linked", "gradual"]:
        subset = episodes[episodes["episode_type"] == episode_type]
        for window in windows:
            window_returns = []
            for _, ep in subset.iterrows():
                ep_rets = _slice_episode_returns(
                    returns,
                    ep["transition_date"],
                    ep["episode_end"],
                    max_months=window,
                )
                window_returns.extend(ep_rets.dropna().tolist())

            n_months = len(window_returns)
            if n_months == 0:
                rows.append({
                    "episode_type": episode_type,
                    "window_months": window,
                    "mean_return": np.nan,
                    "ann_sharpe": np.nan,
                    "n_months": 0,
                })
                continue

            arr = np.array(window_returns)
            mean_ret = arr.mean()
            ann_sharpe = mean_ret / full_sample_vol * np.sqrt(12) if full_sample_vol > 0 else np.nan
            rows.append({
                "episode_type": episode_type,
                "window_months": window,
                "mean_return": mean_ret,
                "ann_sharpe": ann_sharpe,
                "n_months": n_months,
            })

    return pd.DataFrame(rows)


def compute_stable_baseline(
    returns: pd.Series,
    labels: pd.Series,
    full_sample_vol: Optional[float] = None,
) -> Dict[str, float]:
    """Metrics for stable-regime months only."""
    if full_sample_vol is None:
        full_sample_vol = returns.std()

    common = returns.index.intersection(labels.index)
    stable_mask = labels.loc[common] == "stable"
    stable_rets = returns.loc[common][stable_mask].dropna()

    if len(stable_rets) == 0:
        return {"mean_return": np.nan, "ann_sharpe": np.nan, "n_months": 0}

    mean_ret = stable_rets.mean()
    ann_sharpe = mean_ret / full_sample_vol * np.sqrt(12) if full_sample_vol > 0 else np.nan
    return {
        "mean_return": mean_ret,
        "ann_sharpe": ann_sharpe,
        "n_months": len(stable_rets),
    }


def _collect_pooled_returns(
    returns: pd.Series,
    episodes: pd.DataFrame,
    episode_type: str,
    max_months: int,
) -> np.ndarray:
    """Pool post-transition returns across episodes of a given type."""
    subset = episodes[episodes["episode_type"] == episode_type]
    pooled = []
    for _, ep in subset.iterrows():
        ep_rets = _slice_episode_returns(
            returns,
            ep["transition_date"],
            ep["episode_end"],
            max_months=max_months,
        )
        pooled.extend(ep_rets.dropna().tolist())
    return np.array(pooled)


def compare_crisis_vs_gradual(
    returns: pd.Series,
    episodes: pd.DataFrame,
    window_months: int,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Mann-Whitney U test and bootstrap CI on mean-return difference."""
    crisis_rets = _collect_pooled_returns(returns, episodes, "crisis_linked", window_months)
    gradual_rets = _collect_pooled_returns(returns, episodes, "gradual", window_months)

    result = {
        "window_months": window_months,
        "crisis_n": len(crisis_rets),
        "gradual_n": len(gradual_rets),
        "crisis_mean": np.nan,
        "gradual_mean": np.nan,
        "mean_diff": np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
    }

    if len(crisis_rets) == 0 or len(gradual_rets) == 0:
        return result

    result["crisis_mean"] = float(crisis_rets.mean())
    result["gradual_mean"] = float(gradual_rets.mean())
    result["mean_diff"] = result["crisis_mean"] - result["gradual_mean"]

    try:
        u_stat, p_val = mannwhitneyu(crisis_rets, gradual_rets, alternative="two-sided")
        result["mannwhitney_u"] = float(u_stat)
        result["mannwhitney_p"] = float(p_val)
    except ValueError:
        pass

    rng = np.random.default_rng(random_seed)
    diffs = []
    for _ in range(n_bootstrap):
        c_sample = rng.choice(crisis_rets, size=len(crisis_rets), replace=True)
        g_sample = rng.choice(gradual_rets, size=len(gradual_rets), replace=True)
        diffs.append(c_sample.mean() - g_sample.mean())
    result["bootstrap_ci_low"] = float(np.percentile(diffs, 2.5))
    result["bootstrap_ci_high"] = float(np.percentile(diffs, 97.5))
    return result


def _bucket_midpoints(bin_edges: List[int], bin_labels: List[str]) -> Dict[str, float]:
    midpoints = {}
    for i, label in enumerate(bin_labels):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i + 1 < len(bin_edges) else lo + 12
        midpoints[label] = (lo + hi) / 2
    return midpoints


def compute_episode_type_age_metrics(
    returns: pd.Series,
    labels: pd.Series,
    episodes: pd.DataFrame,
    episode_type: str,
    age_bin_edges: List[int],
    age_bin_labels: List[str],
    full_sample_vol: Optional[float] = None,
) -> pd.DataFrame:
    """
    Regime-age bucket metrics restricted to months belonging to episodes
    of the given type (from transition_date through episode_end).
    """
    if full_sample_vol is None:
        full_sample_vol = returns.std()

    subset = episodes[episodes["episode_type"] == episode_type]
    episode_mask = pd.Series(False, index=returns.index)
    for _, ep in subset.iterrows():
        mask = (returns.index >= ep["transition_date"]) & (returns.index <= ep["episode_end"])
        episode_mask |= mask

    regime_age = compute_months_since_transition(labels)
    common = returns.index.intersection(regime_age.index)
    masked_returns = returns.loc[common][episode_mask.loc[common]]
    masked_age = regime_age.loc[common][episode_mask.loc[common]]

    age_buckets = bucket_regime_age(
        masked_age, bin_edges=age_bin_edges, bin_labels=age_bin_labels
    )

    rows = []
    for bucket in age_bin_labels:
        bucket_rets = masked_returns[age_buckets == bucket].dropna()
        n_months = len(bucket_rets)
        if n_months == 0:
            rows.append({
                "age_bucket": bucket,
                "mean_return": np.nan,
                "ann_sharpe": np.nan,
                "n_months": 0,
            })
            continue
        mean_ret = bucket_rets.mean()
        ann_sharpe = mean_ret / full_sample_vol * np.sqrt(12) if full_sample_vol > 0 else np.nan
        rows.append({
            "age_bucket": bucket,
            "mean_return": mean_ret,
            "ann_sharpe": ann_sharpe,
            "n_months": n_months,
        })

    return pd.DataFrame(rows).set_index("age_bucket").reindex(age_bin_labels)


def test_age_trend_within_type(
    returns: pd.Series,
    labels: pd.Series,
    episodes: pd.DataFrame,
    episode_type: str,
    age_bin_edges: List[int],
    age_bin_labels: List[str],
) -> Dict[str, float]:
    """Spearman and trend tests on regime age within an episode type."""
    subset = episodes[episodes["episode_type"] == episode_type]
    episode_mask = pd.Series(False, index=returns.index)
    for _, ep in subset.iterrows():
        mask = (returns.index >= ep["transition_date"]) & (returns.index <= ep["episode_end"])
        episode_mask |= mask

    regime_age = compute_months_since_transition(labels)
    common = returns.index.intersection(regime_age.index)
    aligned = pd.DataFrame({
        "returns": returns.loc[common][episode_mask.loc[common]],
        "age": regime_age.loc[common][episode_mask.loc[common]],
    }).dropna()

    if len(aligned) < 3:
        return {
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "trend_slope": np.nan,
            "trend_p": np.nan,
        }

    rho, spearman_p = spearmanr(aligned["age"], aligned["returns"])
    bucket_metrics = compute_episode_type_age_metrics(
        returns, labels, episodes, episode_type,
        age_bin_edges, age_bin_labels,
    )
    midpoints = _bucket_midpoints(age_bin_edges, age_bin_labels)
    x = np.array([midpoints[b] for b in age_bin_labels])
    y = bucket_metrics.loc[age_bin_labels, "mean_return"].values
    valid = ~np.isnan(y)
    if valid.sum() >= 2:
        trend = linregress(x[valid], y[valid])
        trend_slope, trend_p = trend.slope, trend.pvalue
    else:
        trend_slope, trend_p = np.nan, np.nan

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(spearman_p),
        "trend_slope": float(trend_slope),
        "trend_p": float(trend_p),
    }


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _hypothesis_verdict(
    crisis_age_metrics: pd.DataFrame,
    gradual_age_metrics: pd.DataFrame,
    stable_baseline: Dict[str, float],
    comparisons: List[Dict[str, float]],
    age_bin_labels: List[str],
) -> str:
    """Structured verdict on the left-tail alpha hypothesis."""
    young_bucket = age_bin_labels[0]
    crisis_young_sharpe = crisis_age_metrics.loc[young_bucket, "ann_sharpe"]
    gradual_young_sharpe = gradual_age_metrics.loc[young_bucket, "ann_sharpe"]
    stable_sharpe = stable_baseline.get("ann_sharpe", np.nan)

    six_mo_comp = next((c for c in comparisons if c["window_months"] == 6), {})

    crisis_beats_gradual = (
        not np.isnan(crisis_young_sharpe)
        and not np.isnan(gradual_young_sharpe)
        and crisis_young_sharpe > gradual_young_sharpe
    )
    crisis_beats_stable = (
        not np.isnan(crisis_young_sharpe)
        and not np.isnan(stable_sharpe)
        and crisis_young_sharpe > stable_sharpe
    )
    significant = (
        six_mo_comp.get("mannwhitney_p", np.nan) < 0.05
        if not np.isnan(six_mo_comp.get("mannwhitney_p", np.nan))
        else False
    )

    if crisis_beats_gradual and crisis_beats_stable and significant:
        return "Supported"
    if crisis_beats_gradual or crisis_beats_stable:
        return "Partially supported"
    return "Not supported"


def create_left_tail_exhibit(
    spx_levels: pd.Series,
    labels: pd.Series,
    known_events: List[dict],
    window_metrics: pd.DataFrame,
    crisis_age_metrics: pd.DataFrame,
    gradual_age_metrics: pd.DataFrame,
    stable_baseline: Dict[str, float],
    comparisons: List[Dict[str, float]],
    n_crisis_episodes: int,
    n_gradual_episodes: int,
    verdict: str,
    strategy: str,
    age_bin_labels: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """Multi-panel exhibit for left-tail event analysis."""
    if save_path is None:
        save_path = REPORTS_DIR / "regime_shifts" / "left_tail_exhibit.png"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # Panel 1: Timeline
    ax_timeline = fig.add_subplot(gs[0, :])
    cum_max = spx_levels.cummax()
    drawdown_pct = (spx_levels / cum_max - 1) * 100

    ax_timeline.fill_between(
        drawdown_pct.index, drawdown_pct.values, 0,
        color="#E8E8E8", alpha=0.8,
    )
    ax_timeline.plot(drawdown_pct.index, drawdown_pct.values, color="#666666", linewidth=0.8)

    for date, label in labels.items():
        color = REGIME_COLORS.get(label, "#CCCCCC")
        ax_timeline.axvspan(date, date + pd.DateOffset(days=28), color=color, alpha=0.15)

    for i, event in enumerate(known_events):
        shock_start = parse_event_date(event["shock_start"])
        shock_end = parse_event_date(event["shock_end"])
        ax_timeline.axvspan(
            shock_start, shock_end,
            color="black", alpha=0.12,
        )
        mid = shock_start + (shock_end - shock_start) / 2
        y_pos = ax_timeline.get_ylim()[0] * (0.92 if i % 2 == 0 else 0.75)
        ax_timeline.text(
            mid, y_pos,
            event["name"], ha="center", fontsize=7, fontweight="bold",
        )

    ax_timeline.set_ylabel("SPX Drawdown (%)")
    ax_timeline.set_title("Timeline: SPX Drawdown, Regime Phases, and Known Left-Tail Events", fontweight="bold")
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: Window metrics bar chart (mean return)
    ax_windows = fig.add_subplot(gs[1, 0])
    windows = sorted(window_metrics["window_months"].unique())
    x = np.arange(len(windows))
    width = 0.35

    crisis_means = []
    gradual_means = []
    for w in windows:
        crisis_row = window_metrics[
            (window_metrics["episode_type"] == "crisis_linked") & (window_metrics["window_months"] == w)
        ]
        gradual_row = window_metrics[
            (window_metrics["episode_type"] == "gradual") & (window_metrics["window_months"] == w)
        ]
        crisis_means.append(crisis_row["mean_return"].values[0] * 100 if len(crisis_row) else 0)
        gradual_means.append(gradual_row["mean_return"].values[0] * 100 if len(gradual_row) else 0)

    ax_windows.bar(x - width / 2, crisis_means, width, label="Crisis-linked", color="#C00000", alpha=0.85)
    ax_windows.bar(x + width / 2, gradual_means, width, label="Gradual", color="#5B9BD5", alpha=0.85)
    ax_windows.axhline(
        stable_baseline["mean_return"] * 100,
        color="gray", linestyle="--", linewidth=1, label="Stable baseline",
    )
    ax_windows.set_xticks(x)
    ax_windows.set_xticklabels([f"{w}mo" for w in windows])
    ax_windows.set_ylabel("Mean Monthly Return (%)")
    ax_windows.set_title("Post-Transition Alpha by Window", fontweight="bold")
    ax_windows.legend(fontsize=8)
    ax_windows.axhline(0, color="black", linewidth=0.5)

    # Panel 3: Window metrics bar chart (Sharpe)
    ax_sharpe = fig.add_subplot(gs[1, 1])
    crisis_sharpes = []
    gradual_sharpes = []
    for w in windows:
        crisis_row = window_metrics[
            (window_metrics["episode_type"] == "crisis_linked") & (window_metrics["window_months"] == w)
        ]
        gradual_row = window_metrics[
            (window_metrics["episode_type"] == "gradual") & (window_metrics["window_months"] == w)
        ]
        crisis_sharpes.append(crisis_row["ann_sharpe"].values[0] if len(crisis_row) else 0)
        gradual_sharpes.append(gradual_row["ann_sharpe"].values[0] if len(gradual_row) else 0)

    ax_sharpe.bar(x - width / 2, crisis_sharpes, width, label="Crisis-linked", color="#C00000", alpha=0.85)
    ax_sharpe.bar(x + width / 2, gradual_sharpes, width, label="Gradual", color="#5B9BD5", alpha=0.85)
    ax_sharpe.axhline(
        stable_baseline["ann_sharpe"],
        color="gray", linestyle="--", linewidth=1, label="Stable baseline",
    )
    ax_sharpe.set_xticks(x)
    ax_sharpe.set_xticklabels([f"{w}mo" for w in windows])
    ax_sharpe.set_ylabel("Annualized Sharpe")
    ax_sharpe.set_title("Post-Transition Sharpe by Window", fontweight="bold")
    ax_sharpe.legend(fontsize=8)
    ax_sharpe.axhline(0, color="black", linewidth=0.5)

    # Panel 4: Age curves
    ax_age = fig.add_subplot(gs[2, 0])
    x_age = np.arange(len(age_bin_labels))
    ax_age.plot(
        x_age, crisis_age_metrics.loc[age_bin_labels, "ann_sharpe"],
        marker="o", color="#C00000", label="Crisis-linked", linewidth=2,
    )
    ax_age.plot(
        x_age, gradual_age_metrics.loc[age_bin_labels, "ann_sharpe"],
        marker="s", color="#5B9BD5", label="Gradual", linewidth=2,
    )
    ax_age.axhline(stable_baseline["ann_sharpe"], color="gray", linestyle="--", label="Stable baseline")
    ax_age.set_xticks(x_age)
    ax_age.set_xticklabels(age_bin_labels)
    ax_age.set_xlabel("Months Since Last Transition")
    ax_age.set_ylabel("Annualized Sharpe")
    ax_age.set_title("Alpha by Regime Age (Episode-Filtered)", fontweight="bold")
    ax_age.legend(fontsize=8)
    ax_age.axhline(0, color="black", linewidth=0.5)

    # Panel 5: Summary table
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis("off")

    table_rows = [
        ["Known events", str(len(known_events))],
        ["Crisis-linked episodes", str(n_crisis_episodes)],
        ["Gradual episodes", str(n_gradual_episodes)],
        ["", ""],
        ["Mann-Whitney (6mo)", ""],
    ]
    six_mo = next((c for c in comparisons if c["window_months"] == 6), {})
    table_rows.append(["  p-value", _fmt_p(six_mo.get("mannwhitney_p", np.nan))])
    table_rows.append([
        "  Bootstrap CI",
        f"[{six_mo.get('bootstrap_ci_low', np.nan)*100:.3f}, {six_mo.get('bootstrap_ci_high', np.nan)*100:.3f}]%",
    ])
    table_rows.append(["", ""])
    table_rows.append(["Hypothesis verdict", verdict])

    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax_table.set_title("Summary Statistics", fontweight="bold", pad=20)

    fig.suptitle(
        f"Left-Tail Event Analysis — {strategy}\n"
        f"Hypothesis: alpha peaks near crisis-linked transitions before stabilization ({verdict})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Exhibit saved to: {save_path}")
    plt.close()
    return save_path


def _print_interpretation(
    crisis_age_metrics: pd.DataFrame,
    gradual_age_metrics: pd.DataFrame,
    stable_baseline: Dict[str, float],
    comparisons: List[Dict[str, float]],
    n_known_events: int,
    n_crisis_episodes: int,
    n_gradual_episodes: int,
    verdict: str,
    age_bin_labels: List[str],
) -> None:
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    print(f"  -> Known events: {n_known_events}; "
          f"crisis-linked episodes: {n_crisis_episodes}; gradual episodes: {n_gradual_episodes}")

    young = age_bin_labels[0]
    c_sharpe = crisis_age_metrics.loc[young, "ann_sharpe"]
    g_sharpe = gradual_age_metrics.loc[young, "ann_sharpe"]
    s_sharpe = stable_baseline.get("ann_sharpe", np.nan)

    if not np.isnan(c_sharpe) and not np.isnan(g_sharpe):
        if c_sharpe > g_sharpe:
            print(f"  -> Crisis-linked Sharpe higher than gradual in {young} bucket "
                  f"({c_sharpe:.3f} vs {g_sharpe:.3f})")
        else:
            print(f"  -> Crisis-linked Sharpe NOT higher than gradual in {young} bucket "
                  f"({c_sharpe:.3f} vs {g_sharpe:.3f})")

    if not np.isnan(c_sharpe) and not np.isnan(s_sharpe):
        if c_sharpe > s_sharpe:
            print(f"  -> Crisis-linked Sharpe exceeds stable baseline ({c_sharpe:.3f} vs {s_sharpe:.3f})")
        else:
            print(f"  -> Crisis-linked Sharpe does not exceed stable baseline ({c_sharpe:.3f} vs {s_sharpe:.3f})")

    for comp in comparisons:
        p = comp.get("mannwhitney_p", np.nan)
        w = comp["window_months"]
        if not np.isnan(p):
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  -> {w}-month window Mann-Whitney ({sig}): p={p:.4f}, "
                  f"diff={comp.get('mean_diff', np.nan)*100:.3f}%/mo")

    print(f"  -> Hypothesis verdict: {verdict}")


def run_left_tail_analysis(
    n_buckets: int = 5,
    back_test_start_date: str = "1985-01-31",
    forward_look_months: int = 1,
    similarity_window: int = 1,
    regime_method: str = "phase",
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None,
    lt_config: Optional[dict] = None,
    age_bin_edges: Optional[List[int]] = None,
    age_bin_labels: Optional[List[str]] = None,
    use_cache: bool = True,
    create_exhibit: bool = True,
) -> Dict[str, object]:
    """Run left-tail event analysis end-to-end."""
    if lt_config is None:
        lt_config = cfg.get("regime_shifts", {}).get("left_tail", {})
    if age_bin_edges is None:
        age_bin_edges = cfg.get("regime_shifts", {}).get("regime_age", {}).get(
            "age_bin_edges", DEFAULT_AGE_BIN_EDGES
        )
    if age_bin_labels is None:
        age_bin_labels = cfg.get("regime_shifts", {}).get("regime_age", {}).get(
            "age_bin_labels", DEFAULT_AGE_BIN_LABELS
        )

    strategy = lt_config.get("strategy", "Q1_minus_Q5")
    windows = lt_config.get("post_transition_windows", [3, 6, 12])
    event_anchor = lt_config.get("event_anchor", "shock_end")
    known_events = load_known_events(cfg)
    events = known_events_to_dataframe(known_events, anchor=event_anchor)

    print("=" * 70)
    print("Left-Tail Event Analysis")
    print("Hypothesis: alpha peaks near crisis-linked transitions before stabilization")
    print("=" * 70)

    print(f"\n1. Loaded {len(known_events)} known left-tail events from config...")

    print("\n2. Loading EWMA regime shifts and labeling...")
    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)
    regime_labels = label_regimes(
        ewma_df,
        method=regime_method,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile,
    )

    print("\n3. Building transition episodes...")
    episodes = build_transition_episodes(regime_labels, events, lt_config)
    n_crisis = int((episodes["episode_type"] == "crisis_linked").sum())
    n_gradual = int((episodes["episode_type"] == "gradual").sum())
    print(f"   Crisis-linked episodes: {n_crisis}")
    print(f"   Gradual episodes: {n_gradual}")

    print("\n4. Loading backtest returns...")
    backtest_returns = load_backtest_returns(
        n_buckets=n_buckets,
        back_test_start_date=back_test_start_date,
        forward_look_months=forward_look_months,
        similarity_window=similarity_window,
    )
    if strategy not in backtest_returns.columns:
        raise ValueError(f"Strategy '{strategy}' not found in backtest returns")

    print("\n5. Aligning data and computing metrics...")
    common_dates = backtest_returns.index.intersection(regime_labels.index)
    returns = backtest_returns.loc[common_dates, strategy]
    labels_aligned = regime_labels.loc[common_dates]
    episodes_aligned = episodes[
        episodes["transition_date"].isin(common_dates)
    ].copy()
    full_vol = returns.std()

    window_metrics = compute_window_metrics(returns, episodes_aligned, windows, full_vol)
    stable_baseline = compute_stable_baseline(returns, labels_aligned, full_vol)

    crisis_age_metrics = compute_episode_type_age_metrics(
        returns, labels_aligned, episodes_aligned, "crisis_linked",
        age_bin_edges, age_bin_labels, full_vol,
    )
    gradual_age_metrics = compute_episode_type_age_metrics(
        returns, labels_aligned, episodes_aligned, "gradual",
        age_bin_edges, age_bin_labels, full_vol,
    )

    print("\n6. Running statistical comparisons...")
    comparisons = [
        compare_crisis_vs_gradual(returns, episodes_aligned, w)
        for w in windows
    ]
    crisis_age_trend = test_age_trend_within_type(
        returns, labels_aligned, episodes_aligned, "crisis_linked",
        age_bin_edges, age_bin_labels,
    )

    verdict = _hypothesis_verdict(
        crisis_age_metrics, gradual_age_metrics, stable_baseline,
        comparisons, age_bin_labels,
    )

    print("\n" + "=" * 70)
    print(f"SUMMARY: Left-Tail Analysis ({strategy})")
    print("=" * 70)
    print("\nPost-transition window metrics:")
    display = window_metrics.copy()
    display["mean_return"] = (display["mean_return"] * 100).round(3)
    display["ann_sharpe"] = display["ann_sharpe"].round(3)
    print(display.to_string(index=False))

    print(f"\nStable baseline: mean={stable_baseline['mean_return']*100:.3f}%, "
          f"Sharpe={stable_baseline['ann_sharpe']:.3f}, n={stable_baseline['n_months']}")

    print("\nCrisis-linked age buckets:")
    print(crisis_age_metrics.round(4).to_string())
    print(f"Crisis-linked age trend Spearman rho={crisis_age_trend['spearman_rho']:.4f} "
          f"(p={crisis_age_trend['spearman_p']:.4f})")

    _print_interpretation(
        crisis_age_metrics, gradual_age_metrics,
        stable_baseline, comparisons,
        len(known_events), n_crisis, n_gradual, verdict, age_bin_labels,
    )

    if create_exhibit:
        print("\n7. Creating exhibit...")
        spx_levels = load_spx_monthly_levels()
        create_left_tail_exhibit(
            spx_levels, labels_aligned, known_events,
            window_metrics, crisis_age_metrics, gradual_age_metrics,
            stable_baseline, comparisons, n_crisis, n_gradual, verdict,
            strategy, age_bin_labels,
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return {
        "events": events,
        "known_events": known_events,
        "episodes": episodes_aligned,
        "window_metrics": window_metrics,
        "crisis_age_metrics": crisis_age_metrics,
        "gradual_age_metrics": gradual_age_metrics,
        "stable_baseline": stable_baseline,
        "comparisons": comparisons,
        "verdict": verdict,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Left-tail event analysis.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--method", type=str, default=None,
                        choices=["phase", "percentile", "absolute"])
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--low-threshold-percentile", type=float, default=None)
    parser.add_argument("--high-threshold-percentile", type=float, default=None)
    args = parser.parse_args()

    alpha_config = cfg.get("regime_shifts", {}).get("alpha_by_regime", {})
    lt_config = cfg.get("regime_shifts", {}).get("left_tail", {}).copy()
    if args.strategy is not None:
        lt_config["strategy"] = args.strategy

    params = dict(
        n_buckets=cfg["backtest"].get("n_buckets", 5),
        back_test_start_date=cfg["backtest"].get("back_test_start_date", "1985-01-31"),
        forward_look_months=cfg["backtest"].get("forward_look_months", 1),
        similarity_window=cfg["state_variables"]["similarity_score"].get("similarity_window", 1),
        regime_method=args.method if args.method is not None else alpha_config.get("regime_method", "phase"),
        low_threshold_percentile=(
            args.low_threshold_percentile if args.low_threshold_percentile is not None
            else alpha_config.get("low_threshold_percentile")
        ),
        high_threshold_percentile=(
            args.high_threshold_percentile if args.high_threshold_percentile is not None
            else alpha_config.get("high_threshold_percentile")
        ),
        lt_config=lt_config,
        use_cache=not args.no_cache,
    )

    run_left_tail_analysis(**params)

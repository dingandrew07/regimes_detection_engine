# detection_quality.py | Regime detection quality evaluation
# ------------------------------------------------------------------------------
# Measures whether phase regime labels reliably identify shifts (timing and
# stability), separate from economic validation in alpha_by_regime.py.

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    from .regime_labels import (
        REGIME_COLORS,
        REGIME_DISPLAY_NAMES,
        get_phase_thresholds,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )
except ImportError:
    from regime_labels import (
        REGIME_COLORS,
        REGIME_DISPLAY_NAMES,
        get_phase_thresholds,
        label_regimes,
        load_config,
        load_ewma_regime_shifts,
    )

cfg = load_config()
CACHE_DIR = Path(cfg["paths"]["cache_dir"])
REPORTS_DIR = Path(cfg["paths"]["reports_dir"])

HIGH_STRESS_REGIMES = {"crisis_onset", "resolution"}
CRISIS_RESOLUTION_LABELS = ["crisis_onset", "resolution"]


def _parse_event_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _months_between(start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex) -> int:
    """Count index steps from start (inclusive) to end (inclusive); NaN if either missing."""
    if start not in index or end not in index:
        return np.nan
    return int(index.get_loc(end) - index.get_loc(start))


def build_stable_mask(
    labels: pd.Series,
    known_events: List[dict],
    stable_buffer_months: int,
) -> pd.Series:
    """True for months considered ground-truth stable (outside buffered event windows)."""
    stable = pd.Series(True, index=labels.index)
    for event in known_events:
        start = _parse_event_date(event["shock_start"]) - pd.DateOffset(months=stable_buffer_months)
        end = _parse_event_date(event["shock_end"]) + pd.DateOffset(months=stable_buffer_months)
        stable.loc[(stable.index >= start) & (stable.index <= end)] = False
    return stable


def compute_shift_detection(
    labels: pd.Series,
    known_events: List[dict],
    detection_window_months: int,
) -> Tuple[pd.DataFrame, float]:
    """
    Per event: did crisis_onset appear within the detection window after shock_start,
    followed by at least one resolution before the window ends?
    """
    rows = []
    for event in known_events:
        shock_start = _parse_event_date(event["shock_start"])
        window_end = shock_start + pd.DateOffset(months=detection_window_months)

        window_labels = labels.loc[
            (labels.index >= shock_start) & (labels.index <= window_end)
        ]

        onset_dates = window_labels.index[window_labels == "crisis_onset"]
        resolution_dates = window_labels.index[window_labels == "resolution"]

        detected = False
        first_onset = pd.NaT
        first_resolution = pd.NaT

        if len(onset_dates) > 0:
            first_onset = onset_dates[0]
            resolutions_after_onset = resolution_dates[resolution_dates > first_onset]
            if len(resolutions_after_onset) > 0:
                first_resolution = resolutions_after_onset[0]
                detected = True

        rows.append({
            "metric_type": "shift_detection",
            "event": event["name"],
            "detected": detected,
            "first_onset": first_onset,
            "first_resolution": first_resolution,
            "detection_window_months": detection_window_months,
        })

    result = pd.DataFrame(rows)
    recall = result["detected"].mean() if len(result) > 0 else np.nan
    return result, float(recall)


def compute_false_positive_rate(
    labels: pd.Series,
    stable_mask: pd.Series,
) -> Tuple[float, int, int]:
    """Fraction of stable months mislabeled as crisis_onset or resolution."""
    stable_labels = labels[stable_mask]
    if len(stable_labels) == 0:
        return np.nan, 0, 0

    false_positives = stable_labels.isin(CRISIS_RESOLUTION_LABELS).sum()
    total_stable = len(stable_labels)
    fpr = false_positives / total_stable
    return float(fpr), int(false_positives), int(total_stable)


def compute_elevated_prediction(labels: pd.Series) -> Dict:
    """
    Test whether elevated months predict crisis_onset in the next month.

    Returns hit rate, baseline rate, lift, contingency table counts, and Fisher exact p-value.
    """
    next_labels = labels.shift(-1)
    valid = labels.notna() & next_labels.notna()

    elevated = valid & (labels == "elevated")
    crisis_next = valid & (next_labels == "crisis_onset")
    not_elevated = valid & ~elevated

    n_elevated = int(elevated.sum())
    n_hits = int((elevated & crisis_next).sum())
    n_crisis_next = int(crisis_next.sum())
    n_valid = int(valid.sum())

    hit_rate = n_hits / n_elevated if n_elevated > 0 else np.nan
    baseline_rate = n_crisis_next / n_valid if n_valid > 0 else np.nan
    lift = hit_rate / baseline_rate if baseline_rate and baseline_rate > 0 else np.nan

    not_elevated_crisis = int((not_elevated & crisis_next).sum())
    not_elevated_no_crisis = int((not_elevated & ~crisis_next).sum())
    elevated_no_crisis = n_elevated - n_hits

    table = np.array([
        [n_hits, elevated_no_crisis],
        [not_elevated_crisis, not_elevated_no_crisis],
    ])
    if table.sum() > 0:
        _, p_value = fisher_exact(table)
        p_value = float(p_value)
    else:
        p_value = np.nan

    return {
        "hit_rate": float(hit_rate) if not np.isnan(hit_rate) else np.nan,
        "baseline_rate": float(baseline_rate) if not np.isnan(baseline_rate) else np.nan,
        "lift": float(lift) if not np.isnan(lift) else np.nan,
        "n_elevated": n_elevated,
        "n_hits": n_hits,
        "n_crisis_next": n_crisis_next,
        "n_valid": n_valid,
        "fisher_p": p_value,
        "table_elevated_crisis": n_hits,
        "table_elevated_no_crisis": elevated_no_crisis,
        "table_not_elevated_crisis": not_elevated_crisis,
        "table_not_elevated_no_crisis": not_elevated_no_crisis,
    }


def compute_resolution_lag(
    labels: pd.Series,
    known_events: List[dict],
    lag_anchor: str,
) -> pd.DataFrame:
    """Months from lag anchor to first resolution label per event."""
    if lag_anchor not in ("shock_end", "shock_start"):
        raise ValueError(f"lag_anchor must be 'shock_end' or 'shock_start' (got {lag_anchor})")

    rows = []
    for event in known_events:
        anchor = _parse_event_date(event[lag_anchor])
        future_labels = labels.loc[labels.index >= anchor]
        resolution_dates = future_labels.index[future_labels == "resolution"]

        lag_months = np.nan
        first_resolution = pd.NaT
        if len(resolution_dates) > 0:
            first_resolution = resolution_dates[0]
            lag_months = _months_between(anchor, first_resolution, labels.index)

        rows.append({
            "metric_type": "resolution_lag",
            "event": event["name"],
            "lag_anchor": lag_anchor,
            "anchor_date": anchor,
            "first_resolution": first_resolution,
            "lag_months": lag_months,
        })

    return pd.DataFrame(rows)


def compute_label_persistence(labels: pd.Series, high_threshold: float, mean_ewma: pd.Series) -> Dict:
    """
    Flip rate among high-stress months and median episode length for
    consecutive crisis_onset / resolution runs.
    """
    high_stress_mask = mean_ewma > high_threshold
    stress_labels = labels[high_stress_mask & labels.isin(HIGH_STRESS_REGIMES)]

    flip_count = 0
    if len(stress_labels) > 1:
        for prev, curr in zip(stress_labels.iloc[:-1], stress_labels.iloc[1:]):
            if prev in HIGH_STRESS_REGIMES and curr in HIGH_STRESS_REGIMES and prev != curr:
                flip_count += 1

    high_stress_months = int(len(stress_labels))
    flip_rate = flip_count / high_stress_months if high_stress_months > 0 else np.nan

    episode_lengths = []
    current_regime = None
    current_length = 0
    for regime in labels:
        if regime in HIGH_STRESS_REGIMES:
            if regime == current_regime:
                current_length += 1
            else:
                if current_regime is not None and current_length > 0:
                    episode_lengths.append(current_length)
                current_regime = regime
                current_length = 1
        else:
            if current_regime is not None and current_length > 0:
                episode_lengths.append(current_length)
            current_regime = None
            current_length = 0
    if current_regime is not None and current_length > 0:
        episode_lengths.append(current_length)

    median_episode_length = float(np.median(episode_lengths)) if episode_lengths else np.nan

    return {
        "flip_count": flip_count,
        "high_stress_months": high_stress_months,
        "flip_rate": float(flip_rate) if not np.isnan(flip_rate) else np.nan,
        "median_episode_length": median_episode_length,
        "n_episodes": len(episode_lengths),
    }


def _add_regime_spans(ax, labels: pd.Series, y_lo: float, y_hi: float, alpha: float = 0.25) -> None:
    """Draw axvspan bands for consecutive months sharing the same regime."""
    if len(labels) == 0:
        return

    current = labels.iloc[0]
    span_start = labels.index[0]

    for date, regime in zip(labels.index[1:], labels.iloc[1:]):
        if regime != current:
            if current in REGIME_COLORS:
                ax.axvspan(
                    span_start, date - pd.Timedelta(days=1),
                    ymin=y_lo, ymax=y_hi,
                    alpha=alpha, facecolor=REGIME_COLORS[current], edgecolor="none",
                )
            span_start = date
            current = regime

    if current in REGIME_COLORS:
        ax.axvspan(
            span_start, labels.index[-1],
            ymin=y_lo, ymax=y_hi,
            alpha=alpha, facecolor=REGIME_COLORS[current], edgecolor="none",
        )


def create_detection_timeline_exhibit(
    ewma_df: pd.DataFrame,
    labels: pd.Series,
    thresholds: Dict,
    known_events: List[dict],
    save_path: Optional[Path] = None,
) -> None:
    """Two-panel exhibit: EWMA + regime bands (top), regime swimlane (bottom)."""
    if save_path is None:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "detection_quality_timeline.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    mean_ewma = ewma_df["mean"].dropna()
    plot_labels = labels.loc[mean_ewma.index]
    plot_start = mean_ewma.index[0]
    plot_end = mean_ewma.index[-1]

    fig, (ax_ewma, ax_lane) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    _add_regime_spans(ax_ewma, plot_labels, 0.0, 1.0, alpha=0.2)
    ax_ewma.plot(mean_ewma.index, mean_ewma.values, color="black", linewidth=1.5, label="mean EWMA", zorder=3)
    ax_ewma.axhline(thresholds["low"], color="gray", linestyle="--", linewidth=1, label="low threshold", zorder=2)
    ax_ewma.axhline(thresholds["high"], color="gray", linestyle=":", linewidth=1, label="high threshold", zorder=2)

    for event in known_events:
        shock_start = _parse_event_date(event["shock_start"])
        shock_end = _parse_event_date(event["shock_end"])
        ax_ewma.axvspan(
            shock_start, shock_end,
            alpha=0.15, facecolor="none", edgecolor="black", linewidth=1.5, hatch="///", zorder=4,
        )
        mid = shock_start + (shock_end - shock_start) / 2
        ax_ewma.text(
            mid, ax_ewma.get_ylim()[1] * 0.97, event["name"],
            ha="center", va="top", fontsize=9, fontweight="bold", zorder=5,
        )

    ax_ewma.set_ylabel("Mean EWMA of global score", fontsize=11)
    ax_ewma.set_title("Detection Quality Timeline", fontsize=14, fontweight="bold")
    ax_ewma.grid(True, axis="y", alpha=0.3)
    ax_ewma.legend(loc="upper right", fontsize=9)

    regime_to_y = {regime: i for i, regime in enumerate(["stable", "elevated", "crisis_onset", "resolution"])}
    y_vals = plot_labels.map(regime_to_y).astype(float)
    for regime, y in regime_to_y.items():
        mask = plot_labels == regime
        if mask.any():
            ax_lane.scatter(
                plot_labels.index[mask], [y] * mask.sum(),
                c=REGIME_COLORS[regime], s=18, label=REGIME_DISPLAY_NAMES[regime], zorder=3,
            )

    ax_lane.set_yticks(list(regime_to_y.values()))
    ax_lane.set_yticklabels([REGIME_DISPLAY_NAMES[r] for r in regime_to_y])
    ax_lane.set_ylabel("Regime", fontsize=11)
    ax_lane.set_xlabel("Date", fontsize=11)
    ax_lane.set_xlim(plot_start, plot_end)
    ax_lane.grid(True, axis="x", alpha=0.2)

    ax_lane.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_lane.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for event in known_events:
        shock_start = _parse_event_date(event["shock_start"])
        shock_end = _parse_event_date(event["shock_end"])
        ax_lane.axvline(shock_start, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_lane.axvline(shock_end, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    handles, legend_labels = ax_lane.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ax_lane.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, ncol=2)

    fig.subplots_adjust(hspace=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved detection quality timeline to {save_path}")


def _print_interpretation(
    shift_recall: float,
    fpr: float,
    persistence: Dict,
    lag_df: pd.DataFrame,
    elevated_pred: Dict,
) -> None:
    print("\nInterpretation:")
    if not np.isnan(shift_recall):
        if shift_recall >= 1.0:
            print("  - Shift detection: all known events show onset -> resolution within the window.")
        else:
            print(f"  - Shift detection: only {shift_recall:.0%} of known events fully detected.")
    if not np.isnan(fpr):
        if fpr > 0.05:
            print(f"  - High false positive rate ({fpr:.1%}): stable months often mislabeled as crisis/resolution.")
        else:
            print(f"  - Low false positive rate ({fpr:.1%}): few stable months mislabeled.")
    if not np.isnan(persistence.get("flip_rate", np.nan)):
        if persistence["flip_rate"] > 0.3:
            print(f"  - High flip rate ({persistence['flip_rate']:.1%}): labels oscillate between onset and resolution.")
        else:
            print(f"  - Moderate flip rate ({persistence['flip_rate']:.1%}): phase labels are relatively persistent.")
    for _, row in lag_df.iterrows():
        if pd.notna(row["lag_months"]):
            print(f"  - {row['event']}: first resolution {int(row['lag_months'])} months after {row['lag_anchor']}.")
        else:
            print(f"  - {row['event']}: no resolution label found after {row['lag_anchor']}.")

    lift = elevated_pred.get("lift", np.nan)
    fisher_p = elevated_pred.get("fisher_p", np.nan)
    if not np.isnan(lift) and not np.isnan(fisher_p):
        if lift > 1.0 and fisher_p < 0.05:
            print(
                "  - Elevated prediction: elevated months modestly predict crisis onset, "
                "but crisis onset is rare so interpret with caution."
            )
        else:
            print(
                "  - Elevated prediction: elevated does not reliably predict next-month crisis onset "
                "(exploratory / near coin-flip)."
            )


def _fmt_pct(value: float, digits: int = 1) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_pvalue(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_date(value) -> str:
    if pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _style_table(table, header_color: str = "#E6E6E6", stripe_color: str = "#F8F8F8") -> None:
    """Apply consistent header and zebra-striping to a matplotlib table."""
    cells = table.get_celld()
    n_cols = max(col for _, col in cells) + 1
    for (row, col), cell in cells.items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", size=10)
        elif row % 2 == 0:
            cell.set_facecolor(stripe_color)
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if col == 0 and row > 0:
            cell.set_text_props(ha="left", size=10)
        else:
            cell.set_text_props(size=10)
    table.auto_set_font_size(False)
    table.scale(1.0, 1.6)


def _add_table_to_axis(ax, col_labels: List[str], rows: List[List[str]], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="gray")
        return
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.92],
    )
    _style_table(table)


def create_detection_quality_summary_exhibit(
    shift_recall: float,
    fpr: float,
    false_positives: int,
    total_stable: int,
    persistence: Dict,
    shift_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    elevated_pred: Dict,
    save_path: Optional[Path] = None,
) -> Path:
    """Render detection quality metrics as a multi-section summary exhibit."""
    if save_path is None:
        reports_dir = REPORTS_DIR / "regime_shifts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "detection_quality_summary.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    n_detected = int(shift_df["detected"].sum()) if len(shift_df) > 0 else 0
    n_events = len(shift_df)
    median_ep = persistence["median_episode_length"]
    median_ep_str = f"{median_ep:.1f}" if not np.isnan(median_ep) else "n/a"

    aggregate_rows = [
        ["Shift detection recall", f"{_fmt_pct(shift_recall)} ({n_detected}/{n_events} events)"],
        ["False positive rate", f"{_fmt_pct(fpr, 2)} ({false_positives}/{total_stable} stable months)"],
        [
            "Onset/resolution flips",
            f"{persistence['flip_count']} ({_fmt_pct(persistence['flip_rate'])} of high-stress months)",
        ],
        [
            "Median episode length",
            f"{median_ep_str} months ({persistence['n_episodes']} episodes)",
        ],
    ]

    shift_rows = [
        [
            row["event"],
            "Detected" if row["detected"] else "Missed",
            _fmt_date(row["first_onset"]),
            _fmt_date(row["first_resolution"]),
        ]
        for _, row in shift_df.iterrows()
    ]

    lag_rows = [
        [
            row["event"],
            row["lag_anchor"],
            "—" if pd.isna(row["lag_months"]) else str(int(row["lag_months"])),
            _fmt_date(row["first_resolution"]),
        ]
        for _, row in lag_df.iterrows()
    ]

    lift = elevated_pred["lift"]
    lift_str = f"{lift:.2f}x" if not np.isnan(lift) else "n/a"
    elevated_rows = [
        [
            "Hit rate (elevated → crisis next)",
            f"{_fmt_pct(elevated_pred['hit_rate'])} ({elevated_pred['n_hits']} / {elevated_pred['n_elevated']})",
        ],
        ["Baseline (crisis next, all months)", _fmt_pct(elevated_pred["baseline_rate"])],
        ["Lift", lift_str],
        ["Fisher exact p-value", _fmt_pvalue(elevated_pred["fisher_p"])],
    ]
    contingency_rows = [
        [
            "Elevated",
            str(elevated_pred["table_elevated_crisis"]),
            str(elevated_pred["table_elevated_no_crisis"]),
        ],
        [
            "Not elevated",
            str(elevated_pred["table_not_elevated_crisis"]),
            str(elevated_pred["table_not_elevated_no_crisis"]),
        ],
    ]

    sections = [
        ("Aggregate Metrics", ["Metric", "Value"], aggregate_rows),
        ("Shift Detection by Event", ["Event", "Status", "First Onset", "First Resolution"], shift_rows),
        ("Resolution Lag", ["Event", "Lag Anchor", "Lag (months)", "First Resolution"], lag_rows),
        ("Elevated → Crisis Onset (next month)", ["Metric", "Value"], elevated_rows),
        ("Contingency Table", ["", "Crisis Next", "No Crisis Next"], contingency_rows),
    ]

    row_counts = [len(rows) + 1 for _, _, rows in sections]
    fig_height = max(12, sum(row_counts) * 0.45 + 2.5)
    fig, axes = plt.subplots(
        len(sections), 1,
        figsize=(11, fig_height),
        gridspec_kw={"hspace": 0.55},
    )
    if len(sections) == 1:
        axes = [axes]

    fig.suptitle("Detection Quality Summary", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.985,
        "Timing and stability of phase regime labels (not economic outcomes)",
        ha="center", fontsize=9, style="italic", color="gray",
    )

    for ax, (title, col_labels, rows) in zip(axes, sections):
        _add_table_to_axis(ax, col_labels, rows, title)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved detection quality summary exhibit to {save_path}")
    return save_path


def run_detection_quality_analysis(
    use_cache: bool = True,
    create_exhibit: bool = True,
    low_threshold_percentile: Optional[float] = None,
    high_threshold_percentile: Optional[float] = None,
) -> pd.DataFrame:
    """Run detection quality evaluation end-to-end."""
    print("=" * 70)
    print("Regime Detection Quality Analysis")
    print("Focus: timing and stability of phase labels (not economic outcomes)")
    print("=" * 70)

    alpha_config = cfg.get("regime_shifts", {}).get("alpha_by_regime", {})
    dq_config = cfg.get("regime_shifts", {}).get("detection_quality", {})

    regime_method = alpha_config.get("regime_method", "phase")
    if regime_method != "phase":
        raise ValueError(
            "Detection quality requires regime_method='phase'. "
            f"Got '{regime_method}' in config."
        )

    low_pct = low_threshold_percentile if low_threshold_percentile is not None else alpha_config.get(
        "low_threshold_percentile", 0.40
    )
    high_pct = high_threshold_percentile if high_threshold_percentile is not None else alpha_config.get(
        "high_threshold_percentile", 0.75
    )

    known_events = dq_config.get("known_events", [])
    stable_buffer_months = dq_config.get("stable_buffer_months", 6)
    lag_anchor = dq_config.get("lag_anchor", "shock_end")
    detection_window_months = dq_config.get("detection_window_months", 24)

    print("\n1. Loading EWMA regime shifts...")
    ewma_df = load_ewma_regime_shifts(use_cache=use_cache)

    print("\n2. Labeling regimes (phase method)...")
    regime_labels = label_regimes(
        ewma_df,
        method="phase",
        low_threshold_percentile=low_pct,
        high_threshold_percentile=high_pct,
    )

    mean_ewma = ewma_df["mean"].dropna()
    thresholds = get_phase_thresholds(mean_ewma, low_pct, high_pct)

    print("\n3. Computing detection quality metrics...")
    stable_mask = build_stable_mask(regime_labels, known_events, stable_buffer_months)
    shift_df, shift_recall = compute_shift_detection(
        regime_labels, known_events, detection_window_months
    )
    fpr, false_positives, total_stable = compute_false_positive_rate(regime_labels, stable_mask)
    lag_df = compute_resolution_lag(regime_labels, known_events, lag_anchor)
    persistence = compute_label_persistence(regime_labels, thresholds["high"], mean_ewma)
    elevated_pred = compute_elevated_prediction(regime_labels)

    print("\n--- Detection Quality Summary ---")
    print(f"  Shift detection recall:     {shift_recall:.1%} ({shift_df['detected'].sum()}/{len(shift_df)} events)")
    print(f"  False positive rate:        {fpr:.2%} ({false_positives}/{total_stable} stable months)")
    print(f"  Onset/resolution flips:     {persistence['flip_count']} (rate {persistence['flip_rate']:.1%} among high-stress months)")
    print(f"  Median episode length:      {persistence['median_episode_length']:.1f} months ({persistence['n_episodes']} episodes)")

    print("\n--- Per-Event Details ---")
    for _, row in shift_df.iterrows():
        status = "DETECTED" if row["detected"] else "MISSED"
        print(f"  {row['event']}: {status}")
        if pd.notna(row["first_onset"]):
            print(f"    first onset:      {row['first_onset'].date()}")
        if pd.notna(row["first_resolution"]):
            print(f"    first resolution: {row['first_resolution'].date()}")

    print("\n--- Resolution Lag ---")
    print(lag_df[["event", "lag_anchor", "lag_months", "first_resolution"]].to_string(index=False))

    print("\n--- Elevated -> Crisis Onset (next month) ---")
    print(
        f"  Hit rate (elevated -> crisis next):  {elevated_pred['hit_rate']:.1%}  "
        f"({elevated_pred['n_hits']} / {elevated_pred['n_elevated']})"
    )
    print(f"  Baseline (crisis next, all months): {elevated_pred['baseline_rate']:.1%}")
    print(f"  Lift:                               {elevated_pred['lift']:.2f}x")
    fisher_p = elevated_pred["fisher_p"]
    fisher_str = f"{fisher_p:.3f}" if not np.isnan(fisher_p) else "n/a"
    print(f"  Fisher exact p-value:               {fisher_str}")

    _print_interpretation(shift_recall, fpr, persistence, lag_df, elevated_pred)

    if create_exhibit:
        print("\n4. Creating summary exhibit...")
        create_detection_quality_summary_exhibit(
            shift_recall, fpr, false_positives, total_stable,
            persistence, shift_df, lag_df, elevated_pred,
        )
        print("\n5. Creating timeline exhibit...")
        create_detection_timeline_exhibit(
            ewma_df, regime_labels, thresholds, known_events,
        )

    print("\n" + "=" * 70)
    print("Detection quality analysis complete!")
    print("=" * 70)

    return shift_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate regime detection quality.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-plot", action="store_true", help="Skip summary and timeline exhibits")
    parser.add_argument("--low-threshold-percentile", type=float, default=None,
                        help="Lower percentile for phase method (overrides config)")
    parser.add_argument("--high-threshold-percentile", type=float, default=None,
                        help="Upper percentile for phase method (overrides config)")
    args = parser.parse_args()

    run_detection_quality_analysis(
        use_cache=not args.no_cache,
        create_exhibit=not args.no_plot,
        low_threshold_percentile=args.low_threshold_percentile,
        high_threshold_percentile=args.high_threshold_percentile,
    )

# known_events.py | Shared known left-tail event catalog
# ------------------------------------------------------------------------------
# Single source of truth for configured market crisis episodes used by
# detection_quality.py and left_tail_analysis.py.

from typing import Dict, List, Optional, Union

import pandas as pd


def parse_event_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def load_known_events(cfg: dict) -> List[dict]:
    """
    Load known_events from regime_shifts.known_events.

    Falls back to legacy regime_shifts.detection_quality.known_events if present.
    """
    regime_cfg = cfg.get("regime_shifts", {})
    events = regime_cfg.get("known_events")
    if events:
        return events
    return regime_cfg.get("detection_quality", {}).get("known_events", [])


def _anchor_date(event: dict, anchor: str) -> pd.Timestamp:
    if anchor not in ("shock_end", "shock_start"):
        raise ValueError(f"anchor must be 'shock_end' or 'shock_start' (got {anchor})")
    return parse_event_date(event[anchor])


def known_events_to_dataframe(
    events: List[dict],
    anchor: str = "shock_end",
) -> pd.DataFrame:
    """Convert known_events list to DataFrame with anchor date column."""
    if not events:
        return pd.DataFrame(columns=["name", "shock_start", "shock_end", "event_date"])

    rows = []
    for event in events:
        rows.append({
            "name": event["name"],
            "shock_start": parse_event_date(event["shock_start"]),
            "shock_end": parse_event_date(event["shock_end"]),
            "event_date": _anchor_date(event, anchor),
        })
    return pd.DataFrame(rows)


def event_in_lookback(
    transition_date: pd.Timestamp,
    event: Union[dict, pd.Series],
    lookback_months: int,
    anchor: str = "shock_end",
) -> bool:
    """True if event anchor date falls in [transition_date - lookback, transition_date]."""
    if isinstance(event, pd.Series):
        anchor_date = event["event_date"]
    else:
        anchor_date = _anchor_date(event, anchor)
    lookback_start = transition_date - pd.DateOffset(months=lookback_months)
    return lookback_start <= anchor_date <= transition_date


def find_anchor_event(
    transition_date: pd.Timestamp,
    events_df: pd.DataFrame,
    lookback_months: int,
) -> Optional[pd.Timestamp]:
    """Most recent event anchor in [t - lookback, t], or None."""
    if events_df.empty:
        return None

    lookback_start = transition_date - pd.DateOffset(months=lookback_months)
    mask = (events_df["event_date"] >= lookback_start) & (events_df["event_date"] <= transition_date)
    hits = events_df.loc[mask]
    if hits.empty:
        return None
    return hits["event_date"].max()

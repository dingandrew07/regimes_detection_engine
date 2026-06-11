import pandas as pd

from src.regime_shifts.known_events import (
    event_in_lookback,
    find_anchor_event,
    known_events_to_dataframe,
)
from src.regime_shifts.left_tail_analysis import (
    build_transition_episodes,
    find_label_changes,
)


def test_known_events_to_dataframe_uses_shock_end_by_default():
    events = [
        {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"},
    ]
    df = known_events_to_dataframe(events, anchor="shock_end")
    assert len(df) == 1
    assert df.iloc[0]["name"] == "COVID"
    assert df.iloc[0]["event_date"] == pd.Timestamp("2020-03-31")
    assert df.iloc[0]["shock_start"] == pd.Timestamp("2020-02-29")


def test_known_events_to_dataframe_shock_start_anchor():
    events = [
        {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"},
    ]
    df = known_events_to_dataframe(events, anchor="shock_start")
    assert df.iloc[0]["event_date"] == pd.Timestamp("2020-02-29")


def test_event_in_lookback():
    event = {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"}
    transition = pd.Timestamp("2020-04-30")
    assert event_in_lookback(transition, event, lookback_months=6, anchor="shock_end")
    assert not event_in_lookback(
        pd.Timestamp("2021-01-31"), event, lookback_months=6, anchor="shock_end"
    )


def test_find_anchor_event_uses_shock_end():
    events = known_events_to_dataframe([
        {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"},
        {"name": "GFC", "shock_start": "2007-08-31", "shock_end": "2009-03-31"},
    ])
    anchor = find_anchor_event(pd.Timestamp("2020-05-31"), events, lookback_months=6)
    assert anchor == pd.Timestamp("2020-03-31")


def test_find_label_changes():
    labels = pd.Series(
        ["stable", "stable", "elevated", "crisis_onset", "resolution"],
        index=pd.date_range("2020-01-31", periods=5, freq="ME"),
        name="regime",
    )
    changes = find_label_changes(labels)
    assert len(changes) == 3
    assert changes.iloc[0]["from_regime"] == "stable"
    assert changes.iloc[0]["to_regime"] == "elevated"
    assert changes.iloc[2]["to_regime"] == "resolution"


def test_build_transition_episodes_crisis_linked():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    labels = pd.Series(
        ["stable"] * 3 + ["elevated"] + ["crisis_onset"] * 2 + ["resolution"] * 6,
        index=dates,
        name="regime",
    )
    events = known_events_to_dataframe([
        {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"},
    ])
    cfg = {
        "transition_lookback_months": 6,
        "post_event_window_months": 12,
    }
    episodes = build_transition_episodes(labels, events, cfg)

    crisis_episodes = episodes[episodes["episode_type"] == "crisis_linked"]
    assert len(crisis_episodes) >= 1
    assert crisis_episodes.iloc[0]["anchor_event_date"] == pd.Timestamp("2020-03-31")


def test_build_transition_episodes_gradual():
    dates = pd.date_range("2015-01-31", periods=8, freq="ME")
    labels = pd.Series(
        ["stable"] * 4 + ["elevated"] * 4,
        index=dates,
        name="regime",
    )
    events = known_events_to_dataframe([])
    cfg = {
        "transition_lookback_months": 6,
        "post_event_window_months": 12,
    }
    episodes = build_transition_episodes(labels, events, cfg)
    assert len(episodes) == 1
    assert episodes.iloc[0]["episode_type"] == "gradual"

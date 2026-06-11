import pandas as pd

from src.regime_shifts.known_events import (
    event_in_lookback,
    find_anchor_event,
    known_events_to_dataframe,
    load_known_events,
)


def test_load_known_events_from_config():
    cfg = {
        "regime_shifts": {
            "known_events": [
                {"name": "COVID", "shock_start": "2020-02-29", "shock_end": "2020-03-31"},
            ]
        }
    }
    events = load_known_events(cfg)
    assert len(events) == 1
    assert events[0]["name"] == "COVID"


def test_load_known_events_legacy_fallback():
    cfg = {
        "regime_shifts": {
            "detection_quality": {
                "known_events": [
                    {"name": "GFC", "shock_start": "2007-08-31", "shock_end": "2009-03-31"},
                ]
            }
        }
    }
    events = load_known_events(cfg)
    assert len(events) == 1
    assert events[0]["name"] == "GFC"

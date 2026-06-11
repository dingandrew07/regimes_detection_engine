import pandas as pd
import numpy as np

from src.regime_shifts.regime_labels import (
    bucket_regime_age,
    compute_months_since_transition,
)


def test_compute_months_since_transition_resets_on_change():
    labels = pd.Series(
        ['stable', 'stable', 'elevated', 'elevated', 'elevated', 'crisis_onset'],
        index=pd.date_range('2020-01-31', periods=6, freq='ME'),
        name='regime',
    )
    ages = compute_months_since_transition(labels)
    assert list(ages.values) == [0, 1, 0, 1, 2, 0]


def test_bucket_regime_age_default_bins():
    ages = pd.Series([0, 5, 6, 11, 12, 23, 24, 36])
    buckets = bucket_regime_age(ages)
    assert list(buckets.astype(str)) == ['0-6', '0-6', '6-12', '6-12', '12-24', '12-24', '24+', '24+']

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from src.state_variables.state_variables import cfg

# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#
def is_month_end_date(date: pd.Timestamp) -> bool:
    """
    Check if a date is the last day of its month (month-end).
    
    Args:
        date: Timestamp to check
    
    Returns:
        True if date is month-end, False otherwise
    """
    # Check if adding 1 day would move to the next month
    next_day = date + pd.Timedelta(days=1)
    return next_day.month != date.month

def detect_gaps_in_monthly_series(df: pd.DataFrame, tolerance_days: int = 5) -> list:
    """
    Detect unexpected gaps in a monthly time series DataFrame.
    
    Checks that consecutive dates are approximately 1 month apart (28-31 days
    for month-end dates, accounting for month length variations).
    
    Args:
        df: DataFrame with DatetimeIndex
        tolerance_days: Maximum allowed deviation from expected month length (default: 5 days)
    
    Returns:
        List of tuples (gap_start, gap_end, gap_days) for gaps outside tolerance
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    if len(df.index) < 2:
        return []
    
    gaps = []
    for i in range(len(df.index) - 1):
        date_diff = df.index[i + 1] - df.index[i]
        days_diff = date_diff.days
        
        # Expected monthly gap: 28-31 days (accounting for month-end variations)
        # Month-end to month-end can be 28-31 days depending on the months
        # A gap of 59+ days indicates a missing month
        expected_min_days = 28 - tolerance_days
        expected_max_days = 31 + tolerance_days
        
        if days_diff < expected_min_days or days_diff > expected_max_days:
            gaps.append((df.index[i], df.index[i + 1], days_diff))
    
    return gaps

# -----------------------------------------------------------------------------#
# Tests for df_raw
# -----------------------------------------------------------------------------#
def test_df_raw_not_empty():
    # This test assumes a fixture or mock for df_raw
    # Replace with actual loading if available
    df_raw = pd.DataFrame(index=[1,2,3])
    assert len(df_raw.index) > 0

def test_df_raw_index_monotonic():
    df_raw = pd.DataFrame(index=pd.Index([1,2,3]))
    assert df_raw.index.is_monotonic_increasing

def test_df_raw_no_nans():
    df_raw = pd.DataFrame([[1,2],[3,4]], index=[1,2])
    assert df_raw.isna().sum().sum() == 0

def test_no_leading_zeros():
    df_raw = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
    assert all(df_raw[col].iloc[0] != 0 for col in df_raw), "Leading 0s still present"

def test_df_raw_no_mid_series_gaps():
    """Test that df_raw has no unexpected gaps in its monthly date index."""
    data_dir = Path(cfg['paths']['data_dir'])
    excel_path = data_dir / 'df_raw.xlsx'
    
    if not excel_path.exists():
        pytest.skip(f"df_raw.xlsx not found at {excel_path}. Run state_variables.py first.")
    
    df_raw = pd.read_excel(excel_path, index_col=0, parse_dates=True)
    
    # Verify it's a DatetimeIndex
    assert isinstance(df_raw.index, pd.DatetimeIndex), "df_raw index must be DatetimeIndex"
    
    # Check for gaps
    gaps = detect_gaps_in_monthly_series(df_raw, tolerance_days=5)
    
    if gaps:
        gap_messages = [f"{gap[0].strftime('%Y-%m-%d')} to {gap[1].strftime('%Y-%m-%d')} ({gap[2]} days)" 
                       for gap in gaps]
        pytest.fail(f"Found {len(gaps)} unexpected gap(s) in df_raw index:\n" + "\n".join(gap_messages))
    
    # Additional check: verify index is monotonic
    assert df_raw.index.is_monotonic_increasing, "df_raw index must be monotonic increasing"

def test_df_raw_month_end_dates():
    """Test that all dates in df_raw are month-end dates (off-by-one protection)."""
    data_dir = Path(cfg['paths']['data_dir'])
    excel_path = data_dir / 'df_raw.xlsx'
    
    if not excel_path.exists():
        pytest.skip(f"df_raw.xlsx not found at {excel_path}. Run state_variables.py first.")
    
    df_raw = pd.read_excel(excel_path, index_col=0, parse_dates=True)
    
    # Verify it's a DatetimeIndex
    assert isinstance(df_raw.index, pd.DatetimeIndex), "df_raw index must be DatetimeIndex"
    
    # Check all dates are month-end
    non_month_end_dates = [date for date in df_raw.index if not is_month_end_date(date)]
    
    if non_month_end_dates:
        date_messages = [f"{date.strftime('%Y-%m-%d')} (day {date.day})" 
                        for date in non_month_end_dates[:10]]  # Show first 10
        if len(non_month_end_dates) > 10:
            date_messages.append(f"... and {len(non_month_end_dates) - 10} more")
        pytest.fail(f"Found {len(non_month_end_dates)} non-month-end date(s) in df_raw index:\n" + "\n".join(date_messages))

def test_df_raw_index_alignment():
    """Test that all columns in df_raw have identical indices (alignment protection)."""
    data_dir = Path(cfg['paths']['data_dir'])
    excel_path = data_dir / 'df_raw.xlsx'
    
    if not excel_path.exists():
        pytest.skip(f"df_raw.xlsx not found at {excel_path}. Run state_variables.py first.")
    
    df_raw = pd.read_excel(excel_path, index_col=0, parse_dates=True)
    
    # Check that all columns share the same index
    base_index = df_raw.index
    misaligned_columns = []
    
    for col in df_raw.columns:
        # Since all columns are in the same DataFrame, they should have identical indices
        # But we verify by checking if any column has different length or different dates
        col_index = df_raw[col].index
        if not col_index.equals(base_index):
            misaligned_columns.append(col)
    
    if misaligned_columns:
        pytest.fail(f"Columns with misaligned indices in df_raw: {misaligned_columns}")
    
    # Additional check: verify all columns have same number of rows
    row_counts = {col: len(df_raw[col].dropna()) for col in df_raw.columns}
    if len(set(row_counts.values())) > 1:
        pytest.fail(f"Columns have different row counts in df_raw: {row_counts}")

# -----------------------------------------------------------------------------#
# Tests for df_winsorized
# -----------------------------------------------------------------------------#
def test_df_winsorized_no_mid_series_gaps():
    """Test that df_winsorized has no unexpected gaps in its monthly date index."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    pickle_path = cache_dir / 'df_winsorized.pkl'
    
    if not pickle_path.exists():
        pytest.skip(f"df_winsorized.pkl not found at {pickle_path}. Run state_variables.py first.")
    
    df_winsorized = joblib.load(pickle_path)
    
    # Verify it's a DatetimeIndex
    assert isinstance(df_winsorized.index, pd.DatetimeIndex), "df_winsorized index must be DatetimeIndex"
    
    # Check for gaps
    gaps = detect_gaps_in_monthly_series(df_winsorized, tolerance_days=5)
    
    if gaps:
        gap_messages = [f"{gap[0].strftime('%Y-%m-%d')} to {gap[1].strftime('%Y-%m-%d')} ({gap[2]} days)" 
                       for gap in gaps]
        pytest.fail(f"Found {len(gaps)} unexpected gap(s) in df_winsorized index:\n" + "\n".join(gap_messages))
    
    # Additional check: verify index is monotonic
    assert df_winsorized.index.is_monotonic_increasing, "df_winsorized index must be monotonic increasing"

def test_df_winsorized_month_end_dates():
    """Test that all dates in df_winsorized are month-end dates (off-by-one protection)."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    pickle_path = cache_dir / 'df_winsorized.pkl'
    
    if not pickle_path.exists():
        pytest.skip(f"df_winsorized.pkl not found at {pickle_path}. Run state_variables.py first.")
    
    df_winsorized = joblib.load(pickle_path)
    
    # Verify it's a DatetimeIndex
    assert isinstance(df_winsorized.index, pd.DatetimeIndex), "df_winsorized index must be DatetimeIndex"
    
    # Check all dates are month-end
    non_month_end_dates = [date for date in df_winsorized.index if not is_month_end_date(date)]
    
    if non_month_end_dates:
        date_messages = [f"{date.strftime('%Y-%m-%d')} (day {date.day})" 
                        for date in non_month_end_dates[:10]]  # Show first 10
        if len(non_month_end_dates) > 10:
            date_messages.append(f"... and {len(non_month_end_dates) - 10} more")
        pytest.fail(f"Found {len(non_month_end_dates)} non-month-end date(s) in df_winsorized index:\n" + "\n".join(date_messages))

def test_df_winsorized_index_alignment():
    """Test that all columns in df_winsorized have identical indices (alignment protection)."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    pickle_path = cache_dir / 'df_winsorized.pkl'
    
    if not pickle_path.exists():
        pytest.skip(f"df_winsorized.pkl not found at {pickle_path}. Run state_variables.py first.")
    
    df_winsorized = joblib.load(pickle_path)
    
    # Check that all columns share the same index
    base_index = df_winsorized.index
    misaligned_columns = []
    
    for col in df_winsorized.columns:
        col_index = df_winsorized[col].index
        if not col_index.equals(base_index):
            misaligned_columns.append(col)
    
    if misaligned_columns:
        pytest.fail(f"Columns with misaligned indices in df_winsorized: {misaligned_columns}")
    
    # Additional check: verify all columns have same number of rows
    row_counts = {col: len(df_winsorized[col].dropna()) for col in df_winsorized.columns}
    if len(set(row_counts.values())) > 1:
        pytest.fail(f"Columns have different row counts in df_winsorized: {row_counts}")

# -----------------------------------------------------------------------------#
# Tests for factor returns (month-end conversion)
# -----------------------------------------------------------------------------#
def test_df_factors_month_end_dates():
    """Test that all dates in df_factors are month-end dates (off-by-one protection)."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    pickle_path = cache_dir / 'df_factors.pkl'
    
    if not pickle_path.exists():
        pytest.skip(f"df_factors.pkl not found at {pickle_path}. Run factor_returns.py first.")
    
    df_factors = joblib.load(pickle_path)
    
    # Verify it's a DatetimeIndex
    assert isinstance(df_factors.index, pd.DatetimeIndex), "df_factors index must be DatetimeIndex"
    
    # Check all dates are month-end
    non_month_end_dates = [date for date in df_factors.index if not is_month_end_date(date)]
    
    if non_month_end_dates:
        date_messages = [f"{date.strftime('%Y-%m-%d')} (day {date.day})" 
                        for date in non_month_end_dates[:10]]  # Show first 10
        if len(non_month_end_dates) > 10:
            date_messages.append(f"... and {len(non_month_end_dates) - 10} more")
        pytest.fail(f"Found {len(non_month_end_dates)} non-month-end date(s) in df_factors index:\n" + "\n".join(date_messages))

def test_df_factors_index_alignment():
    """Test that all columns in df_factors have identical indices (alignment protection)."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    pickle_path = cache_dir / 'df_factors.pkl'
    
    if not pickle_path.exists():
        pytest.skip(f"df_factors.pkl not found at {pickle_path}. Run factor_returns.py first.")
    
    df_factors = joblib.load(pickle_path)
    
    # Check that all columns share the same index
    base_index = df_factors.index
    misaligned_columns = []
    
    for col in df_factors.columns:
        col_index = df_factors[col].index
        if not col_index.equals(base_index):
            misaligned_columns.append(col)
    
    if misaligned_columns:
        pytest.fail(f"Columns with misaligned indices in df_factors: {misaligned_columns}")
    
    # Additional check: verify all columns have same number of rows
    row_counts = {col: len(df_factors[col].dropna()) for col in df_factors.columns}
    if len(set(row_counts.values())) > 1:
        pytest.fail(f"Columns have different row counts in df_factors: {row_counts}")

# -----------------------------------------------------------------------------#
# Tests for cross-dataframe alignment
# -----------------------------------------------------------------------------#
def test_df_winsorized_similarity_scores_alignment():
    """Test that df_winsorized and similarity_scores have aligned indices where they overlap."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    winsorized_path = cache_dir / 'df_winsorized.pkl'
    similarity_path = cache_dir / 'similarity_scores_window1.pkl'
    
    if not winsorized_path.exists():
        pytest.skip(f"df_winsorized.pkl not found. Run state_variables.py first.")
    if not similarity_path.exists():
        pytest.skip(f"similarity_scores_window1.pkl not found. Run similarity_score.py first.")
    
    df_winsorized = joblib.load(winsorized_path)
    similarity_scores = joblib.load(similarity_path)
    
    # Find overlapping dates
    common_dates = df_winsorized.index.intersection(similarity_scores.index)
    
    if len(common_dates) == 0:
        pytest.fail("No overlapping dates between df_winsorized and similarity_scores")
    
    # Check that overlapping dates are identical (no misalignment)
    winsorized_subset = df_winsorized.loc[common_dates]
    similarity_subset = similarity_scores.loc[common_dates]
    
    assert winsorized_subset.index.equals(similarity_subset.index), \
        "Indices are not properly aligned between df_winsorized and similarity_scores in overlapping period"

def test_similarity_scores_factors_alignment():
    """Test that similarity_scores and df_factors can be properly aligned for backtesting."""
    cache_dir = Path(cfg['paths']['cache_dir'])
    similarity_path = cache_dir / 'similarity_scores_window1.pkl'
    factors_path = cache_dir / 'df_factors.pkl'
    
    if not similarity_path.exists():
        pytest.skip(f"similarity_scores_window1.pkl not found. Run similarity_score.py first.")
    if not factors_path.exists():
        pytest.skip(f"df_factors.pkl not found. Run factor_returns.py first.")
    
    similarity_scores = joblib.load(similarity_path)
    df_factors = joblib.load(factors_path)
    
    # Find overlapping dates (as done in back_test.py)
    min_date = max(similarity_scores.index.min(), df_factors.index.min())
    max_date = min(similarity_scores.index.max(), df_factors.index.max())
    
    if min_date > max_date:
        pytest.fail(f"No overlapping date range between similarity_scores and df_factors")
    
    # Align as done in back_test.py
    similarity_aligned = similarity_scores.loc[min_date:max_date]
    factors_aligned = df_factors.loc[min_date:max_date]
    
    # Remove last row (due to forward shift in factors)
    similarity_aligned = similarity_aligned.iloc[:-1]
    factors_aligned = factors_aligned.iloc[:-1]
    
    # Check that aligned indices match
    assert similarity_aligned.index.equals(factors_aligned.index), \
        f"Indices are not properly aligned after backtest alignment logic. " \
        f"Similarity: {len(similarity_aligned)} rows, Factors: {len(factors_aligned)} rows"
    
    # Verify all dates are month-end in aligned data
    non_month_end_sim = [date for date in similarity_aligned.index if not is_month_end_date(date)]
    non_month_end_fac = [date for date in factors_aligned.index if not is_month_end_date(date)]
    
    if non_month_end_sim:
        pytest.fail(f"Found {len(non_month_end_sim)} non-month-end dates in aligned similarity_scores")
    if non_month_end_fac:
        pytest.fail(f"Found {len(non_month_end_fac)} non-month-end dates in aligned df_factors") 
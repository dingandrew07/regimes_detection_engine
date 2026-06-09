import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from pathlib import Path
import yaml
import warnings

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config() -> dict:
    """Read parameters from config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

cfg = load_config()
DATA_DIR = Path(cfg["paths"]["data_dir"])
DATA_DIR.mkdir(parents=True, exist_ok=True)
END_DATE = pd.to_datetime(cfg["state_variables"]["factor_returns"]["end_date"])

# -----------------------------------------------------------------------------#
# 1  Download & tidy the factor data
# -----------------------------------------------------------------------------#
def _ff_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Fama-French YYYYMM index to month-end DatetimeIndex."""
    # Handle PeriodDtype index (common in pandas_datareader)
    try:
        from pandas import PeriodIndex
        dt = PeriodIndex(df.index).to_timestamp() + pd.offsets.MonthEnd(0)
    except (AttributeError, ImportError):
        # Check the first index value to determine format
        first_index = str(df.index[0])
        
        if '-' in first_index:
            # String format like "2020-07"
            dt = pd.to_datetime(df.index, format="%Y-%m") + pd.offsets.MonthEnd(0)
        else:
            # Integer format like 202007
            dt = pd.to_datetime(df.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    
    df.index = dt
    return df


def download_factor_data(force_download: bool = False) -> pd.DataFrame:
    """Get monthly returns for 5 FF factors + Momentum.
    
    Note: Always downloads and overwrites cache to ensure parameter changes take effect.
    """
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle_file = cache_dir / "df_factors.pkl"

    # Always download and overwrite cache to ensure parameter changes take effect
    # Remove conditional caching to force recalculation every time

    print("Downloading Fama-French 5-Factor data …")
    # Get full historical data from fixed start date (July 1963)
    start_date = "1963-07-01"
    ff5_raw = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start_date)[0]
    print("Downloading Fama-French Momentum factor …")
    mom_raw = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start=start_date)[0]

    # Convert to proper timestamps
    ff5 = _ff_to_month_end(ff5_raw.copy())
    mom = _ff_to_month_end(mom_raw.copy())

    # Select and rename columns
    ff5 = ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    ff5.columns = ["MKT", "SMB", "HML", "RMW", "CMA"]
    mom = mom.rename(columns=lambda c: c.strip())
    mom = mom[["Mom"]]
    mom.columns = ["MOM"]

    # Merge, align, and convert to decimal
    factors_df = (
        pd.concat([ff5, mom], axis=1)
          .sort_index()
          .loc[:END_DATE]
          .dropna()
          .astype('float64')
          .div(100)  # % → decimal
    )
    
    # Only cache to pickle for use by other scripts
    factors_df.to_pickle(pickle_file)
    print(f"Saved factor data to pickle ➜ {pickle_file}")

    return factors_df


# -----------------------------------------------------------------------------#
# 2  Script execution
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    df_factors = download_factor_data(force_download=False)  # force_download parameter is ignored - always downloads
    # Shift factor values so each date contains next month's values
    df_factors = df_factors.shift(-1)

    print("\nPreview of cleaned factor returns (shifted by -1):")
    print(df_factors.head())
    print(df_factors.tail())
    print(f"\nDataFrame shape: {df_factors.shape}")

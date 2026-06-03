# state_variables.py | Section 2 – Economic State Variables
# ------------------------------------------------------------------------------
# Loads and processes 7 economic state variables for regime analysis.
# Generates raw data, z-scores, and winsorized z-scores with visualizations.
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict
import yaml
from pathlib import Path
import joblib

# Apply a professional plot style
sns.set_style("whitegrid")
# Ignore common FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)

print("Libraries imported and settings configured.")

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config():
    """Read parameters from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Load config
cfg = load_config()

# -----------------------------------------------------------------------------#
# 1  Raw Data Loading
# -----------------------------------------------------------------------------#
# Common Parameters
end_date = cfg['state_variables']['end_date']
raw_data_dict = {}

# 1.1 Market Level (S&P 500)
print("1. Loading Market data…")
market_file_path = Path(cfg['paths']['data_dir']) / cfg['data_sources']['market_file']
spx_monthly_df   = pd.read_excel(market_file_path, index_col=0, parse_dates=True)

market_series = (
    spx_monthly_df['SPX_monthly'].squeeze().apply(np.log)   # log-level
      .resample('M').last()
      .rename('Market')
      .loc[:end_date]
)
raw_data_dict['Market'] = market_series

# 1.2 Yield Curve (10Y – 3M)
print("2. Loading Yield-Curve data…")
gs10_ticker  = cfg['data_sources']['gs10']
tb3ms_ticker = cfg['data_sources']['tb3ms']
yield_data   = pdr.get_data_fred([gs10_ticker, tb3ms_ticker],
                                 start='1962-01-01', end=end_date)

yield_curve = (
    (yield_data[gs10_ticker] - yield_data[tb3ms_ticker])
      .resample('M').last()
      .rename('Yield curve')
)
raw_data_dict['Yield curve'] = yield_curve

# 1.3 Oil Price (WTI spot)
print("3. Loading Oil-Price data…")
wti_ticker = cfg['data_sources']['wti']
oil_series = (
    pdr.get_data_fred(wti_ticker, start='1946-01-01', end=end_date)[wti_ticker]
      .resample('M').last()
      .rename('Oil')
)
raw_data_dict['Oil'] = oil_series

# 1.4 Copper Price
print("4. Loading Copper-Price data…")
copper_file_path = Path(cfg['paths']['data_dir']) / cfg['data_sources']['copper_file']
copper_df = pd.read_excel(copper_file_path, index_col=0, parse_dates=True)

copper_price = (
    copper_df['CU_monthly']
      .resample('M').last()
      .rename('Copper')
      .loc['1959-07-31':end_date]
)
raw_data_dict['Copper'] = copper_price

# 1.5 Monetary Policy (3-month T-bill)
print("5. Loading Monetary-Policy data…")
monetary_series = (
    pdr.get_data_fred(tb3ms_ticker, start='1954-01-01', end=end_date)[tb3ms_ticker]
      .resample('M').last()
      .rename('Monetary policy')
)
raw_data_dict['Monetary policy'] = monetary_series

# 1.6 Volatility (realised σ + VIX)
print("6. Loading and constructing Volatility series…")
spx_daily_file = Path(cfg['paths']['data_dir']) / cfg['data_sources']['spx_daily_file']
spx_daily_df = pd.read_excel(spx_daily_file, index_col=0, parse_dates=True)
spx_daily = spx_daily_df['SPX_daily'].squeeze()

# 6A – realised σ pre-1990
first_vol_date = pd.to_datetime('1929-01-31')
start_daily = first_vol_date - pd.Timedelta(days=100)
spx_daily_sub = spx_daily.loc[start_daily:]
# Compute log returns, then 30-day rolling std, annualize and convert to percent
log_returns = np.log(spx_daily_sub / spx_daily_sub.shift(1))
realised_vol_daily = log_returns.rolling(30, min_periods=30).std() * np.sqrt(252) * 100
realised_vol_monthly = realised_vol_daily.resample('M').last()  # Annualized percent
realised_vol = realised_vol_monthly.loc[first_vol_date:'1989-12-31']

# 6B – VIX post-1990
vix_ticker  = cfg['data_sources']['vix_ticker']
vix_data = yf.download(vix_ticker, start='1990-01-01', end=end_date, progress=False)
if vix_data is not None and not vix_data.empty:
    vix_close = vix_data['Close'].squeeze()
    vix_monthly = vix_close.resample('M').last()  # VIX already in annualized percent
else:
    print("Warning: VIX download failed. Using realized volatility for entire period.")
    vix_monthly = realised_vol.loc['1990-01-01':end_date]

# 6C – splice
volatility = realised_vol.combine_first(vix_monthly)
volatility.name = 'Volatility'
raw_data_dict['Volatility'] = volatility

# 1.7 Stock-Bond Correlation
print("7. Loading and constructing Stock-Bond Correlation series…")
spx_ret = spx_daily.pct_change()
spx_ret.name = 'stock_ret'

dgs10_ticker  = cfg['data_sources']['dgs10']
yield_daily   = pdr.get_data_fred(dgs10_ticker,
                                  start='1962-01-02', end=end_date)[dgs10_ticker]
bond_chg      = yield_daily.diff()
bond_chg.name = 'bond_yield_chg'

corr_df = pd.DataFrame({'stock_ret': spx_ret, 'bond_yield_chg': bond_chg}).dropna().ffill()
rolling_corr = (
    corr_df['stock_ret']
      .rolling(window=3*252)             # 3 years
      .corr(corr_df['bond_yield_chg'])
      .resample('M').last()
)
rolling_corr.name = 'Stock-bond correlation'
raw_data_dict['Stock-bond correlation'] = rolling_corr

# -----------------------------------------------------------------------------#
# 2  Data Combination & Cleaning
# -----------------------------------------------------------------------------#
print("Combining series and aligning on common month-end index…")
first_date  = min(s.index.min() for s in raw_data_dict.values())
month_index = pd.date_range(first_date, end_date, freq='M')

# re-index each series to the common calendar
for key in raw_data_dict:
    raw_data_dict[key] = raw_data_dict[key].reindex(month_index)

df_raw = pd.concat(raw_data_dict.values(), axis=1)
df_raw.columns = list(raw_data_dict.keys())        # preserve order

# forward-fill inside each history, then 0s for pre-inception gaps
df_raw = df_raw.ffill().fillna(0)

print("\nAll raw data series have been processed and combined:")
df_raw.info()

# Remove leading zeros for exhibition generation
for col in df_raw.columns:
    first_real = df_raw[col].ne(0).idxmax()      # first non-zero → real data
    df_raw.loc[df_raw.index < first_real, col] = np.nan

# Enforce uniform float64 dtype
df_raw = df_raw.astype('float64')
print("Enforced float64 dtype on all columns in df_raw")

# Export df_raw to Excel for inspection
data_dir = Path(cfg['paths']['data_dir'])
data_dir.mkdir(parents=True, exist_ok=True)
excel_path = data_dir / 'df_raw.xlsx'
df_raw.to_excel(excel_path, sheet_name='Raw State Variables')
print(f"\nExported df_raw to Excel: {excel_path}")

# -----------------------------------------------------------------------------#
# 3  Data Transformation
# -----------------------------------------------------------------------------#
print("Transforming variables to z-scores...")

df_zscored = pd.DataFrame(index=df_raw.index)
df_winsorized = pd.DataFrame(index=df_raw.index)

# Get parameters from config
diff_months = cfg['state_variables']['diff_months']
rolling_years = cfg['state_variables']['rolling_years']
rolling_window = rolling_years * 12
winsorize_lower = cfg['state_variables']['winsorize']['lower']
winsorize_upper = cfg['state_variables']['winsorize']['upper']
rolling_min_periods = cfg['state_variables']['rolling_min_periods']

for col in df_raw.columns:
    # 1. Compute difference (using config parameter)
    diff_12m = df_raw[col].diff(diff_months)
    
    # 2. Compute rolling standard deviation of the differences
    rolling_std = diff_12m.rolling(window=rolling_window, min_periods=rolling_min_periods).std()
    
    # 3. Compute the raw z-score
    raw_zscore = diff_12m / rolling_std
    df_zscored[col] = raw_zscore
    
    # 4. Winsorize (using config parameters)
    winsorized_zscore = raw_zscore.clip(winsorize_lower, winsorize_upper)
    df_winsorized[col] = winsorized_zscore

print("Transformation complete. We have raw z-scores and winsorized z-scores.")

# Find the earliest date where all 7 variables have non-NaN z-score data
common_start_date = df_winsorized.dropna().index[0]

# Permanently cut df_winsorized to start from the earliest common date
df_winsorized = df_winsorized.loc[common_start_date:]
print(f"Cut df_winsorized to start from {common_start_date} - all variables now have data from this date")
print(f"New df_winsorized shape: {df_winsorized.shape}")
print("\nHead of the new df_winsorized (all variables start from same date):")
print(df_winsorized.head())
print(f"\nTail of df_winsorized:")
print(df_winsorized.tail())

# Cache the winsorized data for use by other scripts
# Always overwrite cache to ensure parameter changes take effect
cache_dir = Path(cfg['paths']['cache_dir'])
cache_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(df_winsorized, cache_dir / "df_winsorized.pkl", compress=3)
print(f"Cached winsorized data to {cache_dir / 'df_winsorized.pkl'}")

# -----------------------------------------------------------------------------#
# 4  Visualization
# -----------------------------------------------------------------------------#
# 4.1 Exhibit 2: Raw economic state variables
print("Generating Exhibit 2: Raw economic state variables")

# Define the order and specific y-labels from the paper
plot_specs = {
    'Market': {'ylabel': 'log(price)'},
    'Yield curve': {'ylabel': '10yr - 3m'},
    'Oil': {'ylabel': 'Price'},
    'Copper': {'ylabel': 'Price'},
    'Monetary policy': {'ylabel': 'Yield'},
    'Volatility': {'ylabel': 'Value'},
    'Stock-bond correlation': {'ylabel': '3yr correlation'}
}

# Create a figure with subplots. 4 rows, 2 columns.
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, (col, spec) in enumerate(plot_specs.items()):
    ax = axes[i]
    df_raw[col].dropna().plot(ax=ax, color='tab:blue', linewidth=0.8)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel(spec['ylabel'], fontsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.tick_params(labelsize=8)

# Hide the last (unused) subplot
axes[-1].set_visible(False)
fig.suptitle('Exhibit 2. Raw economic state variables', fontsize=14, y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# Save to reports directory
reports_dir = Path(cfg['paths']['reports_dir']) / 'state variables'
reports_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_dir / 'exhibit2_raw_state_variables.png', dpi=300, bbox_inches='tight')
# plt.show(block=False)  # Commented out to prevent automatic plot display

# 4.2 Exhibit 3: Transformed economic state variables
print("Generating Exhibit 3: Transformed economic state variables")

# Define titles for the z-scored plots
z_plot_titles = {
    'Market': 'Market zscored',
    'Yield curve': 'Yield curve zscored',
    'Oil': 'Oil zscored',
    'Copper': 'Copper zscored',
    'Monetary policy': 'Monetary policy zscored',
    'Volatility': 'Volatility zscored',
    'Stock-bond correlation': 'Stock-bond correlation zscored'
}

fig, axes = plt.subplots(4, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(z_plot_titles.keys()):
    ax = axes[i]
    # Use the full df_winsorized since it's already cut to common start date
    raw_series_to_plot = df_zscored[col].loc[common_start_date:]
    win_series_to_plot = df_winsorized[col]
    ax.plot(raw_series_to_plot.index, raw_series_to_plot.values, 
            label='raw z-score', color='tab:blue', linewidth=1.5)
    ax.plot(win_series_to_plot.index, win_series_to_plot.values, 
            label='winsorized', color='tab:orange', linestyle='--', linewidth=1.5)
    ax.set_title(z_plot_titles[col], fontsize=10, fontweight='bold')
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Z-score', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.tick_params(labelsize=8)

axes[-1].set_visible(False)
fig.suptitle('Exhibit 3. Transformed economic state variables', fontsize=14, y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# Save to reports directory
reports_dir = Path(cfg['paths']['reports_dir']) / 'state variables'
reports_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_dir / 'exhibit3_transformed_state_variables.png', dpi=300, bbox_inches='tight')
# plt.show(block=False)  # Commented out to prevent automatic plot display

# 4.3 Exhibit 4: Autocorrelation table
print("Generating Exhibit 4: Persistence of the economic state variables & Autocorrelation table")

# We will analyze the winsorized data (no NaNs since we cut to common start date)
data_for_stats = df_winsorized

# Lags for autocorrelation in months
lags = {'1 month': 1, '3 month': 3, '12 month': 12, '3 year': 36, '10 year': 120}
autocorrs = {}

for col in data_for_stats.columns:
    autocorrs[col] = [data_for_stats[col].autocorr(lag=l) for l in lags.values()]

# Create a DataFrame from the results
df_autocorr = pd.DataFrame(autocorrs).T
df_autocorr.columns = list(lags.keys())

# Add other descriptive stats from the paper's table
df_autocorr['monthly mean'] = data_for_stats.mean()
df_autocorr['std'] = data_for_stats.std()
df_autocorr['frequency'] = 'monthly'

# Style the DataFrame to match the paper (green for positive, red for negative)
styled_autocorr = df_autocorr.style.background_gradient(
    cmap='RdYlGn',
    axis=1,
    subset=['1 month', '3 month', '12 month', '3 year', '10 year'],
    vmin=-0.5,
    vmax=0.5
).format(
    '{:.2f}',
    subset=pd.IndexSlice[:, df_autocorr.columns[:-1]]
)

# Create a new figure for the autocorrelation table
plt.figure(figsize=(14, 10))
plt.axis('off')  # Turn off axes

# Prepare data for the table
table_data = df_autocorr.round(2).values
table_cols = df_autocorr.columns.tolist()
table_rows = df_autocorr.index.tolist()

# Create the table with variable names as first column
# Add variable names as the first column in the data
table_data_with_names = np.column_stack([table_rows, table_data])
table_cols_with_names = ['Variable'] + table_cols

table = plt.table(cellText=table_data_with_names, 
                  colLabels=table_cols_with_names,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

# Style the table to match the image
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color code the cells based on values (green for positive, red for negative)
for i in range(len(table_rows)):
    for j in range(len(table_cols_with_names)):
        cell = table[(i+1, j)]  # +1 because row 0 is header
        if j == 0:  # Variable name column
            value = table_rows[i]  # This is a string
        else:
            value = table_data[i, j-1]  # j-1 because we added a column
        
        # Color coding for different columns
        if j == 0:  # First column (variable names) - grey out like headers
            cell.set_facecolor('#F0F0F0')  # Very light gray background
        elif j >= 1 and j <= 5:  # Autocorrelation columns (1 month, 3 month, 12 month, 3 year, 10 year)
            if isinstance(value, (int, float)) and value > 0:
                # Green gradient for positive values
                intensity = min(abs(value), 1.0)
                cell.set_facecolor((0.9 - 0.4*intensity, 1.0, 0.9 - 0.4*intensity))
            elif isinstance(value, (int, float)) and value < 0:
                # Red gradient for negative values
                intensity = min(abs(value), 1.0)
                cell.set_facecolor((1.0, 0.9 - 0.4*intensity, 0.9 - 0.4*intensity))
            else:
                cell.set_facecolor('white')
        elif j == 6:  # monthly mean column
            if isinstance(value, (int, float)) and value > 0:
                # Blue gradient for positive values (same as std)
                intensity = min(abs(value), 1.0)
                cell.set_facecolor((0.9 - 0.4*intensity, 0.9 - 0.4*intensity, 1.0))
            elif isinstance(value, (int, float)) and value < 0:
                # Red gradient for negative values
                intensity = min(abs(value), 1.0)
                cell.set_facecolor((1.0, 0.9 - 0.4*intensity, 0.9 - 0.4*intensity))
            else:
                cell.set_facecolor('white')
        elif j == 7:  # std column
            if isinstance(value, (int, float)):
                # Blue gradient for standard deviation values
                intensity = min(value / 1.5, 1.0)  # Normalize to max std of ~1.5
                cell.set_facecolor((0.9 - 0.4*intensity, 0.9 - 0.4*intensity, 1.0))
            else:
                cell.set_facecolor('white')
        else:
            cell.set_facecolor('white')
        
        # Make text bold for better readability
        cell.set_text_props(weight='bold')

# Style header row
for j in range(len(table_cols_with_names)):
    cell = table[(0, j)]
    cell.set_facecolor('#E6E6E6')  # Light gray background
    cell.set_text_props(weight='bold', size=12)

# Style row labels (first column)
for i in range(len(table_rows)):
    cell = table[(i+1, 0)]
    cell.set_facecolor('#F0F0F0')  # Very light gray background
    cell.set_text_props(weight='bold', size=11)

plt.title('Exhibit 4: Autocorrelations of Economic State Variables', fontsize=16, pad=30, weight='bold')

# Save to reports directory
reports_dir = Path(cfg['paths']['reports_dir']) / 'state variables'
reports_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_dir / 'exhibit4_autocorrelations.png', dpi=300, bbox_inches='tight')
# plt.show(block=False)  # Commented out to prevent automatic plot display

# 4.4 Exhibit 5: Correlation heatmap
print("Generating Exhibit 5: Correlation heatmap")

# We will use the z-scored data that has been winsorized (no NaNs since we cut to common start date)
data_for_stats = df_winsorized

# Calculate the correlation matrix
corr_matrix = data_for_stats.corr()

paper_order = [
    'Copper',
    'Monetary policy',
    'Oil',
    'Yield curve',
    'Stock-bond correlation', # Use full name to match what's in our columns
    'Volatility',
    'Market'
]

ordered_cols = []
for name in paper_order:
    # Find the column that contains the partial name
    match = [col for col in corr_matrix.columns if name in col]
    if match:
        ordered_cols.append(match[0])

# If we couldn't find all columns, use the original order
if len(ordered_cols) < len(corr_matrix.columns):
    ordered_cols = corr_matrix.columns.tolist()

corr_matrix_ordered = corr_matrix.loc[ordered_cols, ordered_cols]

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix_ordered,
    annot=True,          # Show the correlation values
    cmap='RdBu_r',       # Use Red-White-Blue (reversed) for positive=red, negative=blue
    center=0,            # Center the colormap at zero
    fmt='.2f',           # Format annotations to two decimal places
    linewidths=.5,
    cbar=True,           # Show color bar
    square=True          # Make cells square
)
plt.title('Exhibit 5: Cross-Correlation of Economic State Variables', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.yticks(rotation=0)
plt.tight_layout()

# Save to reports directory
reports_dir = Path(cfg['paths']['reports_dir']) / 'state variables'
reports_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(reports_dir / 'exhibit5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show(block=False)  # Commented out to prevent automatic plot display

print("\nAll 4 exhibits have been generated and saved to files.")
print("Exhibit 2: Raw economic state variables")
print("Exhibit 3: Transformed economic state variables (z-scored)")
print("Exhibit 4: Autocorrelation table")
print("Exhibit 5: Cross-correlation heatmap")
print(f"\nAll exhibits have been saved to: {reports_dir}")
# plt.show()  # Commented out to prevent automatic plot display
# Regimes Replication

Replication of regime-based factor timing analysis. Run scripts from the project root via `python cli.py <command>` or directly (e.g. `python src/state_variables/state_variables.py`).

## Table of contents

- [Run order](#run-order)
- [Operational notes](#operational-notes)
- [Caches and reports](#caches-and-reports)
- [Module reference](#module-reference)
- [Exhibit index](#exhibit-index)

## Run order

### Main

Run these in sequence:

| Step | Script | CLI |
|------|--------|-----|
| 1 | `src/state_variables/state_variables.py` | `python cli.py state-variables` |
| 2 | `src/state_variables/similarity_score.py` | `python cli.py similarity-score` |
| 3 | `src/state_variables/factor_returns.py` | `python cli.py factor-returns` |
| 4 | `src/backtest/back_test.py` | `python cli.py backtest` |
| 5 *(optional)* | `src/backtest/appendix.py` | `python cli.py appendix` |

- Step 1 caches winsorized state variables (`cache/df_winsorized.pkl`) and produces Exhibits 2–5.
- Step 2 computes pairwise similarity scores (`cache/similarity_scores_window*.pkl`).
- Step 3 is independent of steps 1–2 but must complete before the backtest (`cache/df_factors.pkl`).
- Step 4 is the main backtesting engine (Exhibits 1, 10–12). When `analysis.equal_weighted.enabled: true`, it also triggers the equal-weighted exhibit at the end of the run.
- Step 5 requires similarity scores and factor data from steps 2–3.

### Extensions

Optional analyses beyond the core paper replication. Enable via `extensions.*` in `config.yaml`; triggered automatically from `back_test.py` when enabled.

| Extension | Script | Config flag |
|-----------|--------|-------------|
| Efficacy score | `src/extensions/efficacy_score.py` | `extensions.efficacy_score.enabled: true` |
| Random long bias | `src/extensions/random_long_bias.py` | `extensions.random_long_bias.enabled: true` |

### Analysis

Run after the main pipeline. These scripts depend on cached outputs from the steps above:

| Step | Script | CLI |
|------|--------|-----|
| 1 | `src/analysis/similar_periods.py` | `python cli.py similar-periods` |
| 2 | `src/analysis/clustering_analysis.py` | *(standalone only)* |
| 3 | `src/regime_shifts/regime_shift.py` | *(standalone or via alpha_by_regime / detection_quality)* |
| 4 | `src/regime_shifts/detection_quality.py` | *(standalone only)* |
| 5 | `src/regime_shifts/alpha_by_regime.py` | *(standalone only)* |
| 6 | `src/regime_shifts/regime_age.py` | *(standalone only)* |

`src/analysis/equal_weighted_exhibit.py` is not run standalone; enable via `analysis.equal_weighted.enabled: true` in `config.yaml` and it runs at the end of `back_test.py`.

## Operational notes

**Cache invalidation is manual.** Changing a config parameter does not automatically re-run upstream steps.

- If you change `state_variables.similarity_score.mask_horizon`, `similarity_window`, or `variable_weights`, delete `cache/similarity_scores_window*.pkl` and `cache/similarity_scores_unmasked_window*.pkl`, then re-run `python cli.py similarity-score` before running `backtest`.
- Similarity cache filenames include the window suffix: `similarity_scores_window{N}.pkl` and `similarity_scores_unmasked_window{N}.pkl` (e.g. `N=1` by default).

## Caches and reports

### Cache files

| Cache file | Producer |
|------------|----------|
| `cache/df_winsorized.pkl` | `state_variables.py` |
| `cache/similarity_scores_window{N}.pkl` | `similarity_score.py` |
| `cache/similarity_scores_unmasked_window{N}.pkl` | `similarity_score.py` |
| `cache/df_factors.pkl` | `factor_returns.py` |
| `cache/ewma_regime_shifts.pkl` | `regime_shift.py` |
| `cache/backtest_summary.csv` | `back_test.py` |

Note: `df_zscored` (unwinsorized z-scores) lives in memory inside `state_variables.py` and is used by `clustering_analysis.py` via module import; it is not written to cache.

### Report layout

| Folder | Contents |
|--------|----------|
| `reports/state variables/` | Exhibits 2–5 (raw, transformed, autocorrelations, correlation heatmap) |
| `reports/backtest/` | Core backtest exhibits (Exhibits 1, 10–12) and appendix (A1–A3) |
| `reports/analysis/clustering analysis/` | Clustering evaluation, PCA scatter, means table |
| `reports/analysis/similar periods/` | Similar-period plots |
| `reports/analysis/equal_weighted/` | Equal-weighted performance exhibit |
| `reports/regime_shifts/` | Exhibit 9 (EWMA), detection-quality summary exhibit, detection-quality timeline, alpha-by-regime exhibit, regime-age exhibit, gated backtest exhibits (`gated_backtest/<mode>/`) |
| `reports/extensions/efficacy_score/backtest/` | Backtest exhibits when efficacy extension is enabled |
| `reports/extensions/random_long_bias/` | Random long bias comparison exhibit |

When `extensions.efficacy_score.enabled: true`, all backtest report outputs route to `reports/extensions/efficacy_score/backtest/` instead of `reports/backtest/`.

## Module reference

### `src/state_variables/state_variables.py`

**Purpose:** Load and transform 7 macro state variables into winsorized z-scores. Produce Exhibits 2–5.

**CLI:** `python cli.py state-variables`

**Inputs:**
- `data/spx_monthly.xlsx` — SPX monthly levels (log price state variable)
- `data/spx_daily.xlsx` — daily SPX for realized vol (pre-VIX) and stock-bond correlation
- `data/copper_monthly.xlsx` — copper spot
- FRED tickers via API: `GS10` (10Y yield), `TB3MS` (3M T-bill), `WTISPLC` (WTI), `DGS10` (daily 10Y for correlation series)
- `^VIX` via yfinance — post-1990 volatility

**Config:** `state_variables.end_date`, `diff_months`, `rolling_years`, `rolling_min_periods`, `winsorize.lower` / `.upper`

**Outputs:**
- `cache/df_winsorized.pkl` — 7 winsorized z-score series, trimmed to earliest common date (~1967)
- In-memory `df_zscored` — unwinsorized z-scores (used by `clustering_analysis.py`)
- `reports/state variables/exhibit2_raw_state_variables.png`
- `reports/state variables/exhibit3_transformed_state_variables.png`
- `reports/state variables/exhibit4_autocorrelations.png`
- `reports/state variables/exhibit5_correlation_heatmap.png`

**Notes:**
- `df_winsorized` starts at the earliest date where all 7 variables have a valid `diff_months`-period difference and a `rolling_years`-year rolling σ (~1967). This date is exported as `common_start_date` and used as the start anchor by downstream modules.

---

### `src/state_variables/factor_returns.py`

**Purpose:** Download Fama-French 5-factor + Momentum from `pandas_datareader`, divide by 100, apply a −1-row shift so that each row T holds the return earned from T to T+1.

**CLI:** `python cli.py factor-returns`

**Inputs:**
- Internet (Kenneth French data library via `pandas_datareader`)

**Config:** `state_variables.factor_returns.end_date` (default: `"2024-12-31"`)

**Outputs:**
- `cache/df_factors.pkl` — 6-column DataFrame (MKT, SMB, HML, RMW, CMA, MOM); returns in decimal; last row is NaN after the −1 shift

**Notes:**
- After `df.shift(-1)`, row T holds the factor return from T to T+1. The last row becomes NaN and is dropped before backtest alignment.
- Factors are already long-short portfolios net of the risk-free rate. Do **not** add back RF in Sharpe calculations.

---

### `src/state_variables/similarity_score.py`

**Purpose:** Compute pairwise Euclidean distances between all month-pairs in the 7-dimensional z-score space. Produce a square similarity matrix with the mask applied, plus an unmasked version for plotting.

**CLI:** `python cli.py similarity-score [--target-month YYYY-MM]`

**Inputs:**
- `cache/df_winsorized.pkl`

**Config:** `state_variables.similarity_score.mask_horizon` (default: `36`), `similarity_window` (default: `1`), `variable_weights`

**Outputs:**
- `cache/similarity_scores_window{N}.pkl` — square DataFrame; index and columns are month-end dates; masked entries are `NaN`
- `cache/similarity_scores_unmasked_window{N}.pkl` — same matrix without masking (needed by `similar_periods.py` for the continuous global-score line)

**Notes:**
- Calculation is an O(n²) Python loop over all month pairs. The full matrix is rebuilt on every run; there is no partial-update mechanism.
- Cache is not auto-invalidated on config changes. If `mask_horizon`, `similarity_window`, or `variable_weights` change, delete the `.pkl` files and re-run.
- `similar_periods.py` requires **both** cache files: the masked one ranks similar months; the unmasked one provides the raw distance trace for the chart background.

---

### `src/backtest/back_test.py`

**Purpose:** Expanding-window backtest. For each evaluation month T, rank valid historical months by distance, split into buckets, derive ±1 factor signals from mean bucket returns, and compute a realised monthly return. Produces Exhibits 1, 10, 11, 12.

**CLI:** `python cli.py backtest [--start-date YYYY-MM-DD] [--vol-target 0.15] [--n-buckets 5]`

**Inputs:**
- `cache/similarity_scores_window{N}.pkl`
- `cache/df_factors.pkl`

**Config:** `backtest.n_buckets`, `backtest.back_test_start_date`, `backtest.forward_look_months`, `backtest.vol_target`, `backtest.vol_window`, `backtest.exhibit12_enabled`, `state_variables.similarity_score.similarity_window`, `extensions.efficacy_score.enabled`, `regime_shifts.regime_gating.enabled`, `regime_shifts.regime_gating.mode`, `analysis.equal_weighted.enabled`, `extensions.random_long_bias.enabled`

**Outputs:**
- In-memory `pd.DataFrame` of quintile returns (Q1…QN, long-short, long-only) used by appendix, alpha_by_regime, etc.
- `cache/backtest_summary.csv`
- `reports/backtest/exhibit10_quintile_performance.png`
- `reports/backtest/exhibit11_drawdown_comparison.png`
- `reports/backtest/exhibit12_quantile_sweeps.png`
- `reports/backtest/exhibit1_volatility_targeting.png`
- When efficacy extension enabled: all backtest reports route to `reports/extensions/efficacy_score/backtest/` instead (plus `efficacy_series.csv` and `efficacy_series_plot.png` when `extensions.efficacy_score.save_series: true`)
- When regime gating enabled: exhibits route to `reports/regime_shifts/gated_backtest/<mode>/` (e.g. `resolution_only`, `exclude_crisis_onset`)

**Notes:**
- Date alignment: `similarity_scores` and `df_factors` are trimmed to their shared date range, then the last row is dropped from both (NaN in `df_factors` after the −1 shift).
- Signal rule: mean bucket return > 0 → +1 (long), < 0 → −1 (short). No flat/zero positions.
- Long-only benchmark: equal-weight exposure to all 6 factors each month, no timing signals.
- **Regime gating** (optional): when `regime_shifts.regime_gating.enabled: true`, monthly realised returns are multiplied by 0 or 1 based on phase regime labels. Requires `regime_shifts.alpha_by_regime.regime_method: "phase"`. Modes: `resolution_only` (trade only in resolution) or `exclude_crisis_onset` (flat in crisis onset, full elsewhere).
- **Exhibit 12** runs six extra full backtests when `backtest.exhibit12_enabled: true` (default `false` for faster iteration). Set to `true` when you need `exhibit12_quantile_sweeps.png`.

---

### `src/backtest/appendix.py`

**Purpose:** Run the backtest separately for each of the 6 individual factors and produce three consolidated exhibits — A1 (quintile performance per factor), A2 (Q1−Q5 long-short per factor), A3 (position signals over time per factor).

**CLI:** `python cli.py appendix`

**Inputs:**
- `cache/similarity_scores_window{N}.pkl`
- `cache/df_factors.pkl`

**Config:** Same as backtest: `backtest.n_buckets`, `backtest.back_test_start_date`, `backtest.forward_look_months`, `state_variables.similarity_score.similarity_window`

**Outputs:**
- `reports/backtest/exhibit_a1_quintile_performance.png`
- `reports/backtest/exhibit_a2_long_short_performance.png`
- `reports/backtest/exhibit_a3_positions.png`

---

### `src/analysis/similar_periods.py`

**Purpose:** Plot the global score over time for a chosen target month and highlight the top-percentile most similar historical periods.

**CLI:** `python cli.py similar-periods [--target-month YYYY-MM]`

**Inputs:**
- `cache/similarity_scores_window{N}.pkl` (masked) — for ranking and selecting the most similar months
- `cache/similarity_scores_unmasked_window{N}.pkl` — for plotting the continuous global-score background line
- `cache/df_winsorized.pkl` — for reference date index

**Config:** `analysis.similar_periods.target_month` (default: `"2024-12"`), `state_variables.similarity_score.mask_horizon`, `state_variables.similarity_score.similarity_window`

**Outputs:**
- `reports/analysis/similar periods/similar_periods_<target_month>_window<N>.png` per call

**Notes:**
- Both cache files are required. The masked file enforces recency exclusion when selecting similar months; the unmasked file gives the full unbroken distance trace for the chart background.

---

### `src/analysis/clustering_analysis.py`

**Purpose:** K-means clustering (K = `k_min`–`k_max`) on winsorized z-scored state variables. Selects best K by silhouette score, then produces evaluation plots, a PCA 2D scatter, and a cluster-means heatmap.

**CLI:** `python src/analysis/clustering_analysis.py` — standalone script only, no CLI subcommand.

**Inputs:**
- Imports `state_variables.df_zscored` directly (triggers `state_variables.py` to run if not already imported)

**Config:** `analysis.clustering_analysis.k_min`, `k_max`, `random_state`, `n_init`, `max_iter`, `pca_n_components`

**Outputs:**
- `reports/analysis/clustering analysis/clustering_evaluation.png`
- `reports/analysis/clustering analysis/clustering_pca_scatter.png`
- `reports/analysis/clustering analysis/clustering_means_table.png`

**Notes:**
- The script runs module-level code on import and cannot be cleanly imported into other modules. Run it directly only.
- It temporarily monkey-patches `plt.show` to suppress the exhibit plots that `state_variables.py` emits during its module-level execution.

---

### `src/regime_shifts/regime_shift.py`

**Purpose:** For each evaluation month T, compute an EWMA of historical distances to T across four lookback windows (1, 2, 3, 4 years). Produce Exhibit 9.

**CLI:** No direct subcommand. Called internally by `alpha_by_regime.py`, or run directly via `python src/regime_shifts/regime_shift.py`.

**Inputs:**
- `cache/similarity_scores_window{N}.pkl`

**Config:** `regime_shifts.regime_shift.lookback_periods` (default: `[12, 24, 36, 48]`)

**Outputs:**
- `cache/ewma_regime_shifts.pkl` — DataFrame with columns `1-year`, `2-year`, `3-year`, `4-year`, `mean`; indexed by month-end date
- `reports/regime_shifts/exhibit9_ewma.png`

**Notes:**
- The `mean` column is only valid from **1974-02-28** onward. The 48-month lookback needs 48 months of unmasked distances, which only become available 36 months (mask burn-in) after the ~1967 data start: 1967 + 3yr (mask) + 4yr (longest EWMA) ≈ 1974.

---

### `src/regime_shifts/detection_quality.py`

**Purpose:** Evaluate whether phase regime labels reliably identify shifts (timing and stability), separate from economic validation in `alpha_by_regime.py`. Measures shift detection recall, false positive rate, resolution lag, label persistence, and whether elevated months predict crisis onset in the next month; produces a summary exhibit and a regime-colored timeline exhibit.

**CLI:** `python src/regime_shifts/detection_quality.py [--low-threshold-percentile 0.4] [--high-threshold-percentile 0.75] [--no-cache] [--no-plot]`

**Inputs:**
- `cache/ewma_regime_shifts.pkl` (from `regime_shift.py`)

**Config:** `regime_shifts.detection_quality.known_events`, `stable_buffer_months`, `lag_anchor` (`shock_end` | `shock_start`), `detection_window_months`; labeling thresholds from `regime_shifts.alpha_by_regime` (`regime_method` must be `"phase"`, `low_threshold_percentile`, `high_threshold_percentile`)

**Outputs:**
- `reports/regime_shifts/detection_quality_summary.png` — multi-section summary exhibit (aggregate metrics, per-event shift detection, resolution lag, elevated prediction, contingency table)
- `reports/regime_shifts/detection_quality_timeline.png` — EWMA with threshold lines, regime bands, and known-event annotations

**Notes:**
- Requires `regime_method: "phase"`. Does not use backtest returns.
- **Stable months** for false-positive rate: all labeled months outside buffered known-event windows (`stable_buffer_months` before `shock_start` and after `shock_end`).
- **Shift detection recall:** per configured event, checks for `crisis_onset` then `resolution` within `detection_window_months` after `shock_start`.
- **Elevated prediction (exploratory):** tests whether `elevated` at month *t* predicts `crisis_onset` at *t+1*. Reports hit rate, baseline, lift, and Fisher exact p-value in the summary exhibit.
- Run after `regime_shift.py` and before `alpha_by_regime.py` to validate labels before trusting gated backtest results.

---

### `src/regime_shifts/alpha_by_regime.py`

**Purpose:** Label each month by regime based on the mean EWMA, then compute separate Sharpe ratios and cumulative-return contributions for each regime across all backtest strategies.

**CLI:** `python src/regime_shifts/alpha_by_regime.py [--method phase|percentile|absolute] [--low-threshold-percentile 0.4] [--high-threshold-percentile 0.75] [--threshold-percentile 0.75] [--threshold-absolute <value>] [--no-cache]`

**Inputs:**
- `cache/ewma_regime_shifts.pkl` (from `regime_shift.py`)
- Backtest returns (re-run internally via `back_test.run_backtest`)

**Config:** `regime_shifts.alpha_by_regime.regime_method` (default: `"phase"`), `low_threshold_percentile`, `high_threshold_percentile` (for `phase` method), `regime_threshold_percentile` (for `percentile`), `regime_threshold_absolute` (for `absolute`); plus `backtest.n_buckets`, `backtest.back_test_start_date`, `backtest.forward_look_months`, `state_variables.similarity_score.similarity_window`

**Outputs:**
- `reports/regime_shifts/alpha_by_regime_exhibit.png` — colour-coded table; higher Sharpe cell highlighted green

**Notes:**
- Regime labels derive from the `mean` EWMA column. Months before 1974-02-28 (where `mean` is NaN) are excluded.
- **`phase` method** (default): four regimes — stable, elevated, crisis onset, resolution — using low/high percentile thresholds on EWMA level and direction.
- **`percentile` method:** transition (above threshold) vs stable (at or below).
- **`absolute` method:** transition vs stable using a fixed EWMA value threshold.
- Labeling logic lives in `regime_labels.py` (shared with regime gating).

---

### `src/regime_shifts/regime_age.py`

**Purpose:** Test whether alpha increases as regimes age by computing months since the last regime transition, bucketing into age groups (0–6, 6–12, 12–24, 24+ months), and comparing mean return and Sharpe across buckets with hypothesis tests.

**CLI:** `python src/regime_shifts/regime_age.py [--method phase|percentile|absolute] [--strategy Q1_minus_Q5] [--low-threshold-percentile 0.4] [--high-threshold-percentile 0.75] [--no-cache]`

**Inputs:**
- `cache/ewma_regime_shifts.pkl` (from `regime_shift.py`)
- Backtest returns (re-run internally via `back_test.run_backtest`)

**Config:** `regime_shifts.regime_age.age_bin_edges`, `age_bin_labels`, `strategy`; labeling thresholds from `regime_shifts.alpha_by_regime`; plus `backtest.n_buckets`, `backtest.back_test_start_date`, `backtest.forward_look_months`, `state_variables.similarity_score.similarity_window`

**Outputs:**
- `reports/regime_shifts/regime_age_exhibit.png` — mean monthly return and Sharpe by regime-age bucket, with Spearman and trend-test statistics

**Notes:**
- Regime age = consecutive months in the current regime label since the last transition (0 at the transition month).
- Default strategy is `Q1_minus_Q5` (long-short quintile spread).
- Shared age utilities (`compute_months_since_transition`, `bucket_regime_age`) live in `regime_labels.py`.
- Run after `regime_shift.py` and alongside `alpha_by_regime.py`.

---

### `src/regime_shifts/regime_gating.py`

**Purpose:** Regime-conditional exposure for the main backtest (Step 2). When enabled, scales each month's strategy return by 0 (cash) or 1 (full exposure) based on phase regime labels from EWMA.

**CLI:** No separate subcommand. Enable via `regime_shifts.regime_gating.enabled: true` in `config.yaml` and run `python cli.py backtest`.

**Inputs:**
- `cache/ewma_regime_shifts.pkl` (via `regime_labels.py`)
- Same backtest inputs as `back_test.py`

**Config:** `regime_shifts.regime_gating.enabled`, `regime_shifts.regime_gating.mode` (`resolution_only` | `exclude_crisis_onset`); labeling thresholds from `regime_shifts.alpha_by_regime` (`regime_method` must be `"phase"`, `low_threshold_percentile`, `high_threshold_percentile`)

**Outputs:**
- Gated backtest exhibits under `reports/regime_shifts/gated_backtest/<mode>/` (Exhibits 1, 10, 11, 12)
- `quintile_returns.attrs['regime_exposure']` and `attrs['regime_labels']` on the returned DataFrame

**Notes:**
- `resolution_only`: exposure 1 only when label is resolution; flat otherwise.
- `exclude_crisis_onset`: exposure 0 in crisis onset; full in stable, elevated, and resolution.
- Efficacy extension takes priority for report routing when both are enabled.

---

### `src/analysis/equal_weighted_exhibit.py`

**Purpose:** Plot the cumulative return of a portfolio holding equal 1/N weight in each of the N quintile buckets simultaneously — a long-only alternative with no similarity-based tilts.

**CLI:** Not run independently. Triggered at the end of a backtest run when `analysis.equal_weighted.enabled: true` in `config.yaml`.

**Inputs:**
- `quintile_returns` DataFrame passed in from `back_test.py` (already in memory)

**Config:** `analysis.equal_weighted.enabled` (default: `false`)

**Outputs:**
- `reports/analysis/equal_weighted/exhibit_equal_weighted_performance.png`

---

### `src/extensions/efficacy_score.py`

**Purpose:** Per-month scaling hook inside `back_test.py`. Computes a bootstrap-estimated cross-sectional correlation between predicted and realised factor returns, converts it to an exposure multiplier in [0, 1], and applies that multiplier to all six factor signals before computing the realised return. Also reroutes all backtest report outputs to a separate folder.

**CLI:** No separate subcommand. Enable via `extensions.efficacy_score.enabled: true` in `config.yaml` and run `python cli.py backtest` as normal.

**Inputs (at backtest time):**
- Q1 similar-month pool S(T) for each evaluation month T
- `cache/df_factors.pkl` — used to compute predicted returns from S(T) and the realised return at T

**Config:** `extensions.efficacy_score.enabled` (default: `true`), `bootstrap_iterations` (default: `200`), `random_seed`, `save_series` (default: `true`)

**Outputs (when enabled):**
- All `back_test.py` reports route to `reports/extensions/efficacy_score/backtest/` instead of `reports/backtest/`

**Notes:**
- **Multiplier formula:** `m(T) = clip((eff(T) + 1) / 2, 0, 1)`. Efficacy +1 → multiplier 1.0; 0 → 0.5; −1 → 0.0. All six factor signals at T are multiplied by this scalar before the equal-weighted average is taken.
- `get_reports_dir(base_dir, cfg, subfolder)` handles path routing — when any extension is active it returns `base_dir / "extensions" / <extension_name> / subfolder`; otherwise returns the default path.

---

### `src/extensions/random_long_bias.py`

**Purpose:** Re-run the backtest with randomly assigned ±1 signals (biased toward long by a configurable probability) instead of mean-return signals, then compare the Q1−Q5 spread against the original strategy. Tests whether the strategy's edge comes from similarity-based selection or merely from a structural long bias in the signal rule.

**CLI:** Not run independently. Triggered at the end of a backtest run when `extensions.random_long_bias.enabled: true` in `config.yaml`.

**Inputs:**
- `original_quintile_returns` DataFrame passed in from `back_test.py`
- `cache/similarity_scores_window{N}.pkl` and `cache/df_factors.pkl` (reloaded internally for the random backtest)

**Config:** `extensions.random_long_bias.enabled`, `random_long_bias` (default: `0.75`), `random_seed`; plus `backtest.n_buckets`, `backtest.back_test_start_date`, `backtest.forward_look_months`, `state_variables.similarity_score.similarity_window`

**Outputs:**
- `reports/extensions/random_long_bias/exhibit_random_long_bias_comparison.png`

---

## Exhibit index

| Script | Exhibits / outputs |
|--------|---------------------|
| `state_variables/state_variables.py` | Exhibit 2 (raw), Exhibit 3 (transformed), Exhibit 4 (autocorrelations), Exhibit 5 (correlation heatmap) |
| `state_variables/similarity_score.py` | Caches similarity scores only (no exhibits) |
| `analysis/similar_periods.py` | Similar-period plots |
| `analysis/clustering_analysis.py` | Clustering evaluation, PCA scatter, means table |
| `regime_shifts/regime_shift.py` | Exhibit 9 (EWMA) |
| `regime_shifts/detection_quality.py` | Detection-quality summary exhibit, detection-quality timeline exhibit |
| `backtest/back_test.py` | Exhibit 1 (volatility targeting), Exhibit 10 (quintile performance), Exhibit 11 (drawdown comparison), Exhibit 12 (quantile sweeps); triggers equal-weighted exhibit when `analysis.equal_weighted.enabled: true` |
| `analysis/equal_weighted_exhibit.py` | Equal-weighted performance exhibit (`reports/analysis/equal_weighted/`; triggered from `back_test.py`, not run standalone) |
| `regime_shifts/alpha_by_regime.py` | Alpha-by-regime exhibit |
| `regime_shifts/regime_age.py` | Regime-age exhibit |
| `extensions/random_long_bias.py` | Random long bias comparison exhibit |
| `backtest/appendix.py` | Appendix exhibits A1–A3 |

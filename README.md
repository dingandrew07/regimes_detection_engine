# Regimes Replication

Replication of regime-based factor timing analysis. Scripts are run from the project root (e.g. `python src/state_variables.py` or via `python cli.py <command>`).

## Run Order

### Main

Run these in sequence:

1. `src/state_variables.py` — downloads/processes macro data and caches winsorized state variables (`cache/df_winsorized.pkl`)
2. `src/similarity_score.py` — computes pairwise similarity scores from state variables (`cache/similarity_scores_window*.pkl`)
3. `src/factor_returns.py` — downloads factor return data (`cache/df_factors.pkl`). Independent of steps 1–2, but must complete before the backtest.
4. `src/back_test.py` — main backtesting engine and core performance exhibits
5. *(Optional)* `src/appendix.py` — appendix exhibits (requires similarity scores and factor data)

### Extensions

Optional analyses beyond the core paper replication. Enable via `extensions.*` in `config.yaml`; some are triggered automatically from `back_test.py` when enabled.

- **Efficacy score** (`src/extensions/efficacy_score.py`) — modifies backtest behavior when `extensions.efficacy_score.enabled: true`
- **Equal-weighted** (`src/extensions/equal_weighted_exhibit.py`) — when `extensions.equal_weighted.enabled: true`
- **Random long bias** (`src/extensions/random_long_bias.py`) — when `extensions.random_long_bias.enabled: true`

### Analysis

Run after the main pipeline. These scripts depend on cached outputs from the steps above:

1. `src/analysis/similar_periods.py` — similar-period visualizations
2. `src/analysis/clustering_analysis.py` — k-means clustering exhibits
3. `src/regime_shifts/regime_shift.py` — EWMA regime-shift series and Exhibit 9
4. `src/regime_shifts/alpha_by_regime.py` — alpha performance by regime (requires backtest + EWMA regime shifts from step 3)

## Report Layout

| Folder | Contents |
|--------|----------|
| `reports/analysis/` | Clustering exhibits, similar-period plots (`similar periods/`) |
| `reports/regime_shifts/` | Exhibit 9 (EWMA), alpha-by-regime exhibit and summary |
| `reports/backtest/` | Core backtest exhibits (Exhibits 1, 10–12) |
| `reports/extensions/` | Extension outputs (`efficacy_score/`, `equal_weighted/`, `random_long_bias/`) |

## Exhibit Generation Dependencies

| Script | Exhibits / outputs |
|--------|----------------------|
| `state_variables.py` | Exhibit 2 (raw), Exhibit 3 (transformed), Exhibit 4 (autocorrelations), Exhibit 5 (correlation heatmap) |
| `clustering_analysis.py` | Clustering evaluation, PCA scatter, means table |
| `similarity_score.py` | Caches similarity scores only (no exhibits) |
| `similar_periods.py` | Similar-period plots |
| `regime_shifts/regime_shift.py` | Exhibit 9 (EWMA) |
| `back_test.py` | Exhibit 1 (volatility targeting), Exhibit 10 (quintile performance), Exhibit 11 (drawdown comparison), Exhibit 12 (quantile sweeps) |
| `regime_shifts/alpha_by_regime.py` | Alpha-by-regime exhibit and summary |
| `extensions/equal_weighted_exhibit.py` | Equal-weighted performance exhibit |
| `extensions/random_long_bias.py` | Random long bias comparison exhibit |
| `appendix.py` | Appendix exhibits A1–A3 |

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

- **Efficacy score** (`src/extensions/efficacy_score.py`) — not a separate run step. Enable via `extensions.efficacy_score.enabled` in `config.yaml`; `back_test.py` integrates it automatically when enabled.

### Analysis

Run after the main pipeline. These scripts depend on cached outputs from the steps above:

1. `src/analysis/similar_periods.py` — similar-period visualizations
2. `src/analysis/clustering_analysis.py` — k-means clustering exhibits
3. `src/analysis/regime_shift.py` — EWMA regime-shift series and Exhibit 9
4. `src/analysis/alpha_by_regime.py` — alpha performance by regime (requires backtest + EWMA regime shifts from step 3)
5. `src/analysis/equal_weighted_exhibit.py` — optional; also runnable from `back_test.py` when `analysis.generate_equal_weighted_exhibit: true`
6. `src/analysis/random_long_bias.py` — optional; also runnable from `back_test.py` when `analysis.generate_random_long_bias.enabled: true`

## Exhibit Generation Dependencies

| Script | Exhibits / outputs |
|--------|----------------------|
| `state_variables.py` | Exhibit 2 (raw), Exhibit 3 (transformed), Exhibit 4 (autocorrelations), Exhibit 5 (correlation heatmap) |
| `clustering_analysis.py` | Clustering evaluation, PCA scatter, means table |
| `similarity_score.py` | Caches similarity scores only (no exhibits) |
| `similar_periods.py` | Similar-period plots |
| `regime_shift.py` | Exhibit 9 (EWMA) |
| `back_test.py` | Exhibit 1 (volatility targeting), Exhibit 10 (quintile performance), Exhibit 11 (drawdown comparison), Exhibit 12 (quantile sweeps) |
| `alpha_by_regime.py` | Alpha-by-regime exhibit and summary |
| `appendix.py` | Appendix exhibits A1–A3 |

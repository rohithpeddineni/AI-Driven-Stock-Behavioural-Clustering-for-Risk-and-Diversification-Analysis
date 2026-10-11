# Final-validation data and leakage audit

## Data

- Source file: `data/processed/cleaned_daily_data.csv`.
- Persisted universe: `75` stocks, taken from the canonical
  `stock_fingerprints.csv` symbol index; no symbols were added or removed.
- Prices use the persisted `Adj Close` column. Returns are `pct_change()` on
  the aligned price panel and the first missing return is dropped.
- The cleaned panel contains `85,500` rows and `0` duplicate
  `(Date, Symbol)` records. The aligned price matrix contains `0` missing
  cells across the persisted universe.
- Every strategy uses six holdings in the primary comparison. The fixed
  baseline uses alphabetically first six stocks. The diversified baseline uses
  six deterministic quantiles of the alphabetically sorted universe; neither
  rule uses clustering or test returns.

## Feature and model audit

The final runner reconstructs the exact canonical 14-feature set:
annualised return, volatility, drawdown, rolling-volatility dispersion, beta,
3-month and 12-month momentum, trend strength, lag-1 autocorrelation,
skewness, excess kurtosis, volume volatility, return-volume correlation, and
extreme-move frequency. The prior refinement's extra downside-volatility
feature is not used here.

For each rolling window, fingerprints are built from the training slice only.
Scaling, PCA, and K-Means/GMM are fitted only on training fingerprints.
Ward is fitted only on training fingerprints. Representatives are nearest to
training centroids; no test-period value chooses a stock or model.

## Portfolio assumptions

Portfolios are equal-weighted and daily rebalanced across the fixed test window.
A 10 basis-point initial transaction cost is charged on the first test day;
turnover is recorded as 1.0 because there is no prior portfolio supplied.
No risk-free return is assumed, so Sharpe is excess return over zero. No
transaction cost optimization, future-return optimization, or rebalancing
optimization is performed.

## Remaining caveats

The benchmark index is not present in the cleaned stock panel, so the training
window beta uses the equal-weight stock-market proxy, matching the isolated
experiment convention rather than the canonical benchmark beta. Three windows
are available, with the last window shorter because the fixed dataset ends in
August 2026. This supports comparison, not statistical significance.

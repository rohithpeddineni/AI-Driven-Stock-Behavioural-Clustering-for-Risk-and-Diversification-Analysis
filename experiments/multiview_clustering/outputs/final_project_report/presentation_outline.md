# Presentation Outline

## Slide 1 — Title

**AI-Based Stock Behavioural Clustering for Risk Analysis**

NSE stock universe, historical behavioural fingerprints, clustering, and
diversification analysis.

## Slide 2 — Motivation

- Sector labels describe business activity, not necessarily price behaviour.
- Stocks from different sectors can move similarly.
- Behavioural clustering provides a history-based way to inspect return, risk,
  trend, tail, and volume characteristics.
- The project does not forecast prices or provide investment advice.

## Slide 3 — Research questions

1. Can meaningful behavioural groups be discovered?
2. Which stocks share similar historical behaviour?
3. Do clusters differ in risk and return characteristics?
4. Does selecting across clusters reduce co-movement?

## Slide 4 — Dataset

- 75 NSE stocks.
- January 2022 to August 2026.
- 1,140 aligned trading days in the canonical panel.
- Adjusted close used for returns.
- Volume used only through scale-free volume-behaviour features.
- `^NSEI` is the canonical benchmark; final isolated walk-forward beta uses
  the equal-weight stock proxy because the cleaned panel excludes the index.

## Slide 5 — Behavioural fingerprint

Show the 14 features:

annualised return, annualised volatility, maximum drawdown, rolling-volatility
dispersion, beta, 3-month momentum, 12-month momentum, trend strength, lag-1
autocorrelation, skewness, excess kurtosis, volume volatility, return-volume
correlation, and extreme-move frequency.

## Slide 6 — Canonical model

- Standardize the fingerprint.
- PCA retaining approximately 85% variance.
- K-Means model selection over candidate k values.
- Canonical selected model: PCA + K-Means, k=2.
- Cluster sizes: 69 and 6.
- Silhouette: 0.459.

## Slide 7 — Why test six clusters?

- The project requires investigation of a richer behavioural structure.
- k=2 separates a large majority cluster from a small high-risk group.
- A six-cluster candidate may provide more interpretable behavioural profiles,
  but cluster count alone is not evidence of superiority.

## Slide 8 — Refined six-cluster model

- PCA + K-Means, k=6.
- Four components in the final exact-feature validation.
- Cluster sizes: 13, 6, 10, 15, 14, 17.
- Silhouette: 0.242.
- Davies-Bouldin Index: 1.147.
- Mean seed ARI: 0.878.

Use [cluster_balance.png](../final_validation/plots/cluster_balance.png).

## Slide 9 — Behavioural profiles

Discuss:

- Cluster 1: higher return, volatility, beta, positive momentum, and tail
  measures.
- Cluster 2: very small high-risk group with high volatility, drawdown, beta,
  negative skewness, and high kurtosis.
- Cluster 3: lower return and lower beta defensive profile.
- Cluster 4: higher beta and market-sensitive industrial/energy profile.
- Cluster 5: lower-momentum technology/consumer profile.
- Cluster 6: moderate-beta diversified defensive/growth profile.

Use [cluster_profiles.png](../six_cluster_refinement/plots/cluster_profiles.png)
and verify detailed values in `final_validation/cluster_profiles.csv`.

## Slide 10 — Methods compared

- Canonical PCA k=2.
- Refined PCA k=6.
- Standard fingerprint K-Means k=6.
- RobustScaler K-Means.
- Ward hierarchical clustering.
- Diagonal-covariance GMM.
- Equal-weight and clustering-independent diversified baselines.

## Slide 11 — Leakage-free evaluation

- Three chronological windows.
- 504 training days followed by 252 test days, with a shorter final test
  segment because the dataset ends in August 2026.
- Fingerprints rebuilt from training data.
- Scaling, PCA, and clustering fit on training data only.
- Representatives selected from training geometry.
- Daily equal-weight rebalancing.
- 10 bps initial transaction cost.
- Zero risk-free rate for Sharpe.

## Slide 12 — Portfolio results

Present the final validation mean metrics:

- Canonical k=2: 9.95% return, 16.54% volatility, 11.18% drawdown.
- Refined PCA k=6: 12.37% return, 18.73% volatility, 15.56% drawdown.
- Standard fingerprint k=6: 19.22% return, 17.96% volatility, 12.93% drawdown.
- Equal-weight baseline: 19.35% return, 23.39% volatility, 15.40% drawdown.

Use [oos_comparison.png](../final_validation/plots/oos_comparison.png).

## Slide 13 — Diversification results

- Refined PCA k=6 correlation: 0.273.
- Refined PCA k=6 diversification ratio: 1.591.
- Equal-weight baseline correlation: 0.417.
- Equal-weight baseline diversification ratio: 1.364.
- Diversified clustering-independent baseline: correlation 0.281 and
  diversification ratio 1.562.

Use [oos_diversification.png](../final_validation/plots/oos_diversification.png).

## Slide 14 — Discussion

- Six clusters improve interpretability and reduce extreme imbalance.
- Standard fingerprint K-Means had the highest reported mean return among
  six-cluster alternatives, but this is not conclusive model superiority.
- Diversification gains are not unique to clustering because the
  clustering-independent diversified baseline is competitive.
- Lower risk and correlation can coexist with lower return.

## Slide 15 — Limitations

- Only three evaluation windows.
- Last evaluation window is shorter.
- Historical performance is uncertain and not a forecast.
- No statistical significance claim.
- Benchmark index was unavailable in the cleaned panel used by the isolated
  walk-forward implementation.
- Turnover is recorded as an initial one-time turnover because no prior
  portfolio is supplied.

## Slide 16 — Conclusion

- Academic objective: refined PCA k=6 is defensible.
- Portfolio objective: clustering improves some risk/diversification measures
  but does not establish universal superiority.
- Overall decision: retain canonical k=2 as reference; retain refined k=6 as
  the academic experimental candidate; investigate standard fingerprint k=6
  in future work.


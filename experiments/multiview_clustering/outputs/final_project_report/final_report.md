# AI-Based Stock Behavioural Clustering for Risk Analysis

## Abstract

This project studies whether unsupervised machine learning can group NSE stocks
according to historical market behaviour rather than conventional sector
classification. Each stock is represented by a scale-free behavioural
fingerprint containing return, risk, trend, distribution, market-sensitivity,
and volume-related statistics. The canonical pipeline uses standardized
features, PCA, and K-Means, with a PCA + K-Means k=2 model as the reference.
An isolated refinement investigates a six-cluster alternative and compares it
with standardized K-Means, RobustScaler K-Means, Ward clustering, Gaussian
mixtures, and clustering-independent portfolios.

The final validation uses three chronological walk-forward windows. In every
window, fingerprints are rebuilt from training observations, preprocessing and
clustering are fit only on training data, and representatives are selected
before the subsequent test period. The refined PCA k=6 model produces a more
balanced six-cluster structure than the earlier six-cluster result, with final
cluster sizes `13|6|10|15|14|17`, silhouette `0.242`, Davies-Bouldin Index
`1.147`, and mean seed ARI `0.878`. Its mean out-of-sample return is `12.37%`,
volatility `18.73%`, drawdown `15.56%`, average constituent correlation
`0.273`, and diversification ratio `1.591`.

The evidence supports the six-cluster model as an academic behavioural
framework, not as a proven superior investment strategy. Standard fingerprint
K-Means k=6 has the highest reported mean return among the tested six-cluster
alternatives (`19.22%`) and a favourable risk profile, but three evaluation
windows are insufficient to establish model superiority. The canonical k=2
model remains the reference model, while refined PCA k=6 remains the academic
experimental candidate.

## Executive summary

### Problem

Sector labels describe what a company does, but they do not necessarily describe
how its stock moves. A financial stock, technology stock, and industrial stock
may show similar volatility, momentum, drawdown, or tail behaviour. This project
therefore asks whether stocks can be organized by observed historical behaviour.

### Approach

1. Assemble a fixed universe of 75 NSE stocks.
2. Align daily adjusted prices and volumes.
3. Construct a 14-feature behavioural fingerprint for every stock.
4. Standardize features and compare PCA, K-Means, Ward, robust scaling, and GMM
   alternatives.
5. Profile clusters using the original behavioural features.
6. Construct equal-weight six-stock portfolios using training-only
   representatives.
7. Evaluate return, volatility, drawdown, correlation, diversification, and
   zero-risk-free-rate Sharpe ratios across rolling test windows.

### Main findings

- The canonical PCA k=2 model has the strongest separation score but is very
  imbalanced: 69 stocks in one cluster and 6 in the other.
- The refined PCA k=6 model is substantially more balanced, with no singleton
  clusters and a minimum cluster size of 6.
- Standard fingerprint K-Means k=6 has the highest reported mean return among
  the tested six-cluster clustering alternatives, but this result is descriptive
  and not statistically significant.
- Cluster-based portfolios generally reduce constituent correlation relative to
  the simple alphabetical six-stock baseline.
- A clustering-independent diversified baseline performs competitively, so the
  diversification improvement cannot be attributed solely to clustering.
- The evidence is insufficient to replace the canonical k=2 reference model for
  portfolio use.

## 1. Problem statement and research objectives

Traditional equity grouping often begins with sector classification. Sector
labels are useful metadata, but they are not a direct measurement of market
behaviour. Stocks in the same sector may have different volatility, trend,
drawdown, tail-risk, or volume-response patterns. Conversely, stocks from
different sectors may respond similarly to broad market conditions.

The project investigates four questions:

1. Can unsupervised learning discover meaningful behavioural groups?
2. Which stocks share similar historical fingerprints?
3. Do the discovered groups differ in historical risk and return?
4. Does selecting across behavioural groups reduce co-movement relative to
   selecting within one group?

The project is descriptive. It does not forecast prices, claim that clusters
predict future returns, or provide investment advice.

## 2. Dataset

The canonical universe contains 75 NSE-listed stocks spanning multiple sectors.
The fixed observation period is January 2022 through August 2026. The cleaned
panel contains 85,500 rows: 75 stocks across 1,140 aligned trading days.
Adjusted close is used for return calculations because it incorporates relevant
corporate-action adjustments. The project configuration identifies `^NSEI` as
the benchmark for the canonical beta feature.

The cleaned panel contains zero duplicate `(Date, Symbol)` records and the
aligned final-validation price matrix contains zero missing cells for the
persisted 75-stock universe.

### Behavioural fingerprint

The final canonical fingerprint contains 14 features:

| Feature | Behaviour represented |
|---|---|
| Annualised return | Typical historical performance level |
| Annualised volatility | Overall return magnitude and risk |
| Maximum drawdown | Worst peak-to-trough path loss |
| Rolling-volatility dispersion | Stability of the stock's risk level |
| Beta | Sensitivity to broad-market movement |
| 3-month momentum | Recent directional behaviour |
| 12-month momentum | Longer-horizon directional behaviour |
| Trend strength | Consistency of the log-price trend |
| Lag-1 autocorrelation | Persistence versus short-term reversal |
| Skewness | Asymmetry of the return distribution |
| Excess kurtosis | Fat-tail behaviour |
| Volume volatility | Instability of participation |
| Return-volume correlation | Whether large moves coincide with volume changes |
| Extreme-move frequency | Frequency of stock-relative tail moves |

Mean volume is not used because it would primarily capture company size.
Redundant features were reviewed during the canonical feature-quality process.
The final validation deliberately uses the persisted canonical 14-feature
definition rather than the earlier refinement's temporary 15-feature variant.

## 3. Methodology

### 3.1 Preprocessing

Adjusted prices are aligned to a common trading calendar. Daily returns are
calculated with percentage change and the first undefined observation is
removed. Volume enters only through scale-free changes and correlations.

### 3.2 Scaling and PCA

StandardScaler is used for the canonical and refined PCA representations. PCA is
fit after scaling. The canonical model retains approximately 85% requested
variance; the refined academic candidate uses approximately 70% requested
retention. In the final exact-feature validation, the refined k=6 representation
uses four components and explains approximately 74.8% of variance.

RobustScaler is tested as a sensitivity alternative. It is not selected merely
because it changes cluster sizes.

### 3.3 Clustering methods

The final comparison includes:

- Canonical PCA + K-Means k=2.
- Refined PCA + K-Means k=6.
- Standardized fingerprint K-Means k=6.
- RobustScaler fingerprint K-Means k=6.
- Ward hierarchical clustering k=6.
- Diagonal-covariance Gaussian Mixture Model k=6.

K-Means and GMM use repeated seeds for stability. Ward is deterministic for the
fixed representation. Cluster quality is evaluated independently of portfolio
returns using silhouette, Davies-Bouldin, Calinski-Harabasz, cluster sizes, and
seed stability.

## 4. Experimental design

The canonical reference is PCA + K-Means k=2. It is retained because it is the
existing project model and has strong internal separation, even though its
69|6 partition is too coarse and imbalanced for the six-cluster academic
objective.

The refined PCA k=6 model is evaluated as the academic candidate. Standard
fingerprint K-Means tests whether PCA is discarding useful information. Robust
scaling tests sensitivity to extreme observations. Ward and GMM test
non-K-Means alternatives without imposing equal cluster sizes.

Portfolio comparisons use six holdings. Cluster representatives are chosen by
distance to training-period cluster centroids, with deterministic alphabetical
tie-breaking. The canonical k=2 model is allowed to use both clusters but fills
the six-stock portfolio with additional nearest members, so it is not compared
with fewer holdings.

Two clustering-independent baselines are included:

1. The alphabetically first six stocks.
2. Six deterministic quantiles of the alphabetically sorted universe.

These are transparent comparators, not optimized portfolios.

## 5. Leakage-free evaluation

The final validation uses three chronological windows. Each has a 504-day
training period followed by a 252-day test period where possible; the last test
period is shorter because the fixed data ends in August 2026.

For each window:

1. The 14 fingerprints are recomputed from training observations only.
2. Scaling is fit on training fingerprints.
3. PCA is fit on training fingerprints.
4. The clustering algorithm is fit on training representations.
5. Representatives are selected using training geometry only.
6. The portfolio is evaluated on subsequent unseen returns.

All primary strategies use six holdings, equal weights, and daily rebalancing.
A 10-basis-point initial transaction cost is charged. Turnover is recorded as
1.0 because no pre-existing portfolio is supplied. Sharpe ratio uses a zero
risk-free-rate assumption. These assumptions are identical across directly
compared strategies.

The final validation cannot use the canonical `^NSEI` series for rolling beta
because the cleaned stock panel does not include that benchmark. The isolated
walk-forward implementation therefore uses an equal-weight stock return proxy
for training-window beta. This is documented rather than hidden.

## 6. Clustering results

| Model | k | Cluster sizes | Silhouette | Davies-Bouldin | Calinski-Harabasz | Mean ARI |
|---|---:|---|---:|---:|---:|---:|
| Canonical PCA K-Means | 2 | 69, 6 | 0.459 | 1.029 | 24.152 | 1.000 |
| Refined PCA K-Means | 6 | 13, 6, 10, 15, 14, 17 | 0.242 | 1.147 | 32.954 | 0.878 |
| Standard fingerprint K-Means | 6 | 12, 17, 17, 4, 2, 23 | 0.165 | 1.535 | 16.806 | 0.843 |
| Robust fingerprint K-Means | 6 | 21, 2, 3, 29, 19, 1 | 0.179 | 1.174 | 25.912 | 0.751 |
| Ward | 6 | 19, 5, 2, 19, 19, 11 | 0.142 | 1.498 | 15.436 | 1.000 |
| GMM | 6 | 15, 21, 1, 21, 5, 12 | 0.112 | 1.615 | 14.137 | 0.677 |

The refined PCA k=6 model is the strongest structure for the academic
six-cluster requirement. It avoids singleton clusters and has the narrowest
cluster-size range among the six-cluster methods except for the standard
fingerprint result's smaller range being driven by two tiny groups. Its
silhouette is lower than k=2, which is expected when a coarse two-group
partition is replaced by a richer six-group partition.

### Behavioural profiles

The final cluster profiles are in
[cluster_profiles.csv](../final_validation/cluster_profiles.csv). Selected
observations for refined PCA k=6 include:

- Cluster 1, 13 stocks: high average return (`0.281`), high volatility
  (`0.294`), positive 12-month momentum (`0.433`), beta `1.083`, and elevated
  tail statistics.
- Cluster 2, 6 stocks: high volatility (`0.396`), high drawdown (`0.622`),
  beta `1.494`, negative skewness (`-1.288`), and high excess kurtosis
  (`20.502`). This is a small but behaviourally distinct high-risk group.
- Cluster 3, 10 stocks: low average return (`0.050`), lower volatility
  (`0.224`), beta `0.679`, and negative 12-month momentum (`-0.169`).
- Cluster 4, 15 stocks: relatively high beta (`1.262`), volatility (`0.301`),
  and positive 12-month momentum (`0.164`).
- Cluster 5, 14 stocks: low average return (`0.029`), moderate volatility
  (`0.257`), beta `0.893`, and near-zero long-term momentum.
- Cluster 6, 17 stocks: moderate return (`0.164`), volatility (`0.229`), beta
  `0.808`, and positive 12-month momentum (`0.092`).

These profiles support behavioural interpretation, but they remain historical
descriptions rather than forecasts.

## 7. Portfolio results

The following values are arithmetic means across the three rolling test windows.
They are not a single concatenated-period performance calculation.

| Strategy | Return | Volatility | Drawdown | Sharpe | Correlation | Diversification ratio |
|---|---:|---:|---:|---:|---:|---:|
| Canonical PCA k=2 | 9.95% | 16.54% | 11.18% | 0.613 | 0.240 | 1.637 |
| Refined PCA k=6 | 12.37% | 18.73% | 15.56% | 0.595 | 0.273 | 1.591 |
| Standard fingerprint K-Means k=6 | 19.22% | 17.96% | 12.93% | 0.980 | 0.269 | 1.569 |
| RobustScaler K-Means k=6 | 4.98% | 20.27% | 16.94% | 0.339 | 0.344 | 1.454 |
| Ward k=6 | 5.31% | 20.14% | 17.95% | 0.314 | 0.342 | 1.453 |
| GMM k=6 | 17.17% | 20.01% | 15.58% | 0.863 | 0.295 | 1.538 |
| Equal-weight first six | 19.35% | 23.39% | 15.40% | 0.863 | 0.417 | 1.364 |
| Equal-weight diversified baseline | 11.23% | 17.57% | 12.76% | 0.645 | 0.281 | 1.562 |

The standard fingerprint K-Means k=6 result has the highest reported mean return
among the tested six-cluster clustering alternatives and a lower volatility than
the simple first-six baseline. This does not establish that it is the best
model: there are only three windows, the result varies by window, and the
selection rule remains historical.

The refined PCA k=6 candidate improves correlation and diversification relative
to the simple first-six baseline, but it does not improve every risk metric and
has a lower mean return. The clustering-independent diversified baseline is
competitive with refined PCA k=6, demonstrating that diversification can arise
from a transparent non-clustering selection rule as well.

The actual window-level results are in
[walk_forward_results.csv](../final_validation/walk_forward_results.csv), and
the aggregate table is in
[portfolio_summary.csv](../final_validation/portfolio_summary.csv).

## 8. Discussion

### Interpretability versus separation

The canonical k=2 model has the highest silhouette because two broad groups are
easier to separate. However, a 69|6 split compresses most of the universe into a
single behavioural category. The refined k=6 model gives up some separation to
produce six interpretable profiles with no singleton cluster and a minimum size
of six.

### Diversification versus return

Cluster-based selection generally lowers average constituent correlation relative
to the simple first-six baseline. The refined k=6 model's correlation is `0.273`
versus `0.417` for that baseline, and its diversification ratio is `1.591`
versus `1.364`. However, the clustering-independent diversified baseline has
correlation `0.281` and diversification ratio `1.562`, showing that clustering
is not the only way to obtain a more diversified six-stock portfolio.

### Stability

The refined PCA k=6 mean ARI is `0.878`, lower than deterministic Ward's
`1.000` but higher than robust K-Means and GMM. Stability alone is not enough:
Ward is perfectly stable for the fixed inputs but has weak separation and poor
out-of-sample portfolio results.

### Overall model choice

The evidence supports two different choices for two different objectives:

- For the academic six-cluster objective: refined PCA k=6.
- For a candidate deserving further portfolio testing: standard fingerprint
  K-Means k=6, because it has the strongest reported return/risk compromise
  among the six-cluster alternatives.
- For the project reference and conservative interpretation: canonical PCA k=2.

No choice is declared universally superior.

## 9. Limitations

1. Only three chronological evaluation windows are available. The last is
   shorter, so estimates have high uncertainty.
2. Mean returns, volatility, drawdown, and Sharpe values are historical sample
   statistics, not guarantees of future performance.
3. Three windows do not support strong statistical significance claims. No
   significance claim is made.
4. The final isolated walk-forward implementation uses an equal-weight stock
   proxy for beta because the cleaned stock panel does not include the benchmark
   index series.
5. Transaction costs are simplified to a 10-basis-point initial cost. No prior
   portfolio is available, so turnover is recorded as a one-time `1.0`.
6. Equal-weight daily rebalancing is a consistent comparison convention, not a
   claim that daily rebalancing is operationally optimal.
7. The universe is a fixed historical set of 75 stocks. Results may differ for
   another universe, market, date range, or corporate-action treatment.
8. Cluster labels are descriptive and can change with training window,
   representation, and algorithm.

## 10. Conclusion and future work

The project demonstrates that behavioural fingerprints can organize NSE stocks
into groups that differ in volatility, drawdown, beta, momentum, tail behaviour,
and volume response. The refined PCA k=6 model meets the academic requirement
for at least six meaningful groups more convincingly than the earlier 2- and
4-stock cluster result.

The portfolio evidence is more cautious. Cross-cluster selection improves
some diversification measures relative to a simple six-stock baseline, but a
clustering-independent diversified rule is competitive. Standard fingerprint
K-Means k=6 has the highest reported mean return among the tested six-cluster
alternatives, but the evidence is not strong enough to call it conclusively
best or to replace the canonical k=2 model.

Future work should add more rolling windows when data becomes available,
incorporate the actual benchmark series into every training-window beta
calculation, test turnover and transaction-cost sensitivity with a defined
prior portfolio, and evaluate whether cluster profiles persist across different
stock universes and market regimes.

## Reproducibility and source files

- Final validation report: [experiment_report.md](../final_validation/experiment_report.md)
- Final validation runner: [final_validation.py](../final_validation/final_validation.py)
- Final validation data audit: [data_audit.md](../final_validation/data_audit.md)
- Key results table in this package: [key_results.csv](key_results.csv)
- Presentation outline: [presentation_outline.md](presentation_outline.md)


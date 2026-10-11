# Final Validation Report

## Scope and isolation

This validation writes only to `outputs/final_validation/`. Canonical code,
canonical processed data, and previous experiment outputs were not overwritten.
Canonical hash verification passed: `True`.

## Decision A — academic clustering

The refined PCA K-Means k=6 candidate is technically defensible for an academic
six-group demonstration. It uses the exact canonical 14 features, has six
clusters, and is evaluated independently of portfolio returns. The earlier
refinement's 70% PCA setting produced sizes `13|16|6|13|12|15`; the exact
canonical 14-feature rerun produces `13|6|10|15|14|17` at k=6. Small clusters
are not removed or balanced. Inspect `cluster_profiles.csv` for volatility,
drawdown, momentum, beta, tail, and volume profiles.

## Decision B — portfolio diversification

Across three identical rolling test windows, the refined k=6 cluster
representative strategy had mean return `12.37%`,
volatility `18.73%`, drawdown `15.56%`,
correlation `0.273`, and diversification ratio
`1.591`. The full window-level values are in
`walk_forward_results.csv`; no future test values selected holdings.

Lower correlation and higher diversification do not by themselves prove
superiority. Compare the baseline rows in `portfolio_summary.csv`.

## Decision C — overall performance

There is insufficient evidence to replace canonical k=2 as a portfolio method.
Only three windows are available, the last is shorter, and the methods trade
return against risk. The comparison is descriptive, not a significance test.
No statistical significance is claimed.

## Audit limitations

The cleaned panel does not contain the benchmark index, so training-window beta
uses an equal-weight stock proxy. Daily equal-weight rebalancing is used. A
10-basis-point initial cost is charged and recorded; turnover is a one-time
`1.0` because no prior portfolio is supplied. Risk-free rate is zero for the
Sharpe ratio. The diversified baseline is deterministic and clustering-free:
six quantiles of alphabetical ticker order. These are fair reproducible
baselines, not optimized portfolios.

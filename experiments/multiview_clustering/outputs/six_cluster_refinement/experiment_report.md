# Six-Cluster Refinement Experiment

## Scope and isolation

This version writes only to `outputs/six_cluster_refinement/`. The canonical
`src/` implementation, canonical processed data, and prior experiment outputs
were not modified.

## Method and leakage controls

The 14 canonical fingerprint features are recomputed separately inside each
training window. Scaling, PCA, and clustering are fit on training data only.
Representatives are selected by distance to the training centroid (a medoid-like
rule); no test-period returns, correlations, volatility, or drawdown are used
for selection. Three chronological 504-day training / 252-day test windows use
the same dates for every method.

## Results

The CSV tables are the source of truth. `pca_sensitivity.csv` compares 70%, 80%,
88%, 90%, 95%, and full standardized features. `walk_forward_summary.csv`
reports mean and standard deviation across the three out-of-sample windows.
The plots show the complete candidate set rather than only favorable windows.

The candidate should be selected using separation, stability, cluster-size
interpretability, and the average and variability of out-of-sample risk and
return together. A lower-correlation portfolio is not automatically superior
if its return falls materially. The canonical k=2 model remains untouched and
is not replaced by this experiment.

## Findings

The best structural refinement is **PCA K-Means at the 70% requested retention
setting** (4 components, `76.1%` actual
variance). At k=6 it produces sizes `13|16|6|13|12|15`, minimum size 6,
silhouette `0.2430`, Davies-Bouldin `1.1554`, and mean seed ARI `0.8176`.
This is materially more interpretable than the previous PCA k=6 sizes
`12|4|18|12|27|2`, without balancing or deleting stocks. The 80% setting has
slightly higher silhouette (`0.2537`) and stability (`0.9802`) but retains
two 3-stock groups and a 22-stock group, so it is not preferred on structure
alone. Higher retention settings reintroduce singleton or 2-stock clusters.
The corrected consensus view is scored with precomputed-distance silhouette
only; its Euclidean DB and CH fields are intentionally unavailable.

Across the three rolling test windows, the training-only PCA k=6 pipeline has
mean annualised return `7.02%`,
volatility `18.32%`,
maximum drawdown `16.38%`,
average correlation `0.266`,
and diversification ratio `1.581`.
The six-stock clustering-independent baseline averages 19.51% return, 23.40%
volatility, 15.40% drawdown, 0.417 correlation, and 1.363 diversification
ratio. Thus PCA k=6 improves risk and diversification but gives up return.
Fingerprint K-Means is the strongest return-preserving alternative
(13.24% mean return, 19.65% volatility, 0.295 correlation), while GMM gives
the lowest mean volatility among the clustering methods but only 8.44% return.
Ward is not competitive out of sample in these windows.

**Decision:** retain PCA K-Means k=6 at the 70% requested retention as the
experimental candidate for a better-balanced cluster structure. There is not
enough evidence to replace the canonical k=2 model: only three test windows
are available, and every lower-risk clustering strategy sacrifices some
annualised return.

## Limitations

Only three windows are available in the fixed historical sample. Fingerprints
use an equal-weight mean of the available stocks as a training market proxy
because the cleaned stock panel does not contain the benchmark series. The
experiment is historical analysis, not a return forecast or investment advice.

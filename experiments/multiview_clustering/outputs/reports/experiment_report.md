# Multi-View Clustering Experiment Report

## Objective
This isolated experiment tests whether temporal/co-movement information improves
on the canonical 14-feature PCA + K-Means baseline without modifying it.

## Methods
- Statistical raw fingerprint: StandardScaler + K-Means, k=2..10.
- Statistical PCA fingerprint: PCA retaining 7 components
  (88.05% variance) + K-Means.
- Correlation view: distance `d(i,j)=1-corr(i,j)` + average-linkage
  agglomerative clustering.
- DTW view: constrained DTW on 120 sampled z-normalized return observations with
  a Sakoe-Chiba window of 12, then average-linkage clustering. This is an
  explicit computational approximation, not silently omitted full DTW.
- Consensus view: equal-weight co-association matrix from PCA, correlation, and
  DTW assignments, then average-linkage clustering.

## Experimental design
All methods evaluate k=2..10. K-Means stability uses five fixed seeds. Distance
views compare full-period and first-half solutions. Portfolio selection uses the
canonical centroid/medoid principles and is not tuned to returns.

## Candidate selections

```text
 representation  k  silhouette  davies_bouldin  mean_ari
fingerprint_raw  2    0.429896        1.134334  1.000000
fingerprint_pca  2    0.453932        1.029251  1.000000
    correlation  4    0.165685        1.275635  0.105869
            dtw  9    0.023365        2.326534  0.302166
      consensus  2    0.768884        0.563101       NaN
```

## Outputs
See `outputs/metrics/`, `outputs/clusters/`, `outputs/portfolio/`, and this
report. `method_comparison.csv` contains the complete metric table.

## Results and recommendation

The canonical PCA + K-Means baseline has silhouette 0.454, Davies-Bouldin
1.029, a 69/6 split, and five-seed ARI 1.000 at k=2. The correlation view's
best internal candidate was k=4 (silhouette 0.166, DB 1.276), but its stability
was weak (mean ARI 0.106). The constrained DTW view was also weak on internal
separation (best selected k=9 silhouette 0.023) and stability (mean ARI 0.302).

The equal-weight consensus view produced the strongest separation metrics at k=2
(silhouette 0.769, DB 0.563), but it still produced a highly imbalanced 68/7
split. Its cross-cluster portfolio had return 21.02%, volatility 21.88%,
drawdown 30.30%, and average constituent correlation 0.251, versus its
same-cluster median return 18.66%, volatility 20.71%, drawdown 26.23%, and
correlation 0.414. This is useful evidence, but the portfolio result alone is
not a reason to adopt the method.

Recommendation: retain the existing PCA + K-Means k=2 method as the production
baseline. Use correlation and consensus views as exploratory diagnostics. The
new experiment does not yet justify replacing k=2 because the apparent consensus
metric improvement comes with the same structural imbalance, and the temporal
views are not stable enough to establish a robust replacement. A future upgrade
would need stronger walk-forward validation and a pre-specified stability and
balance criterion.

## Limitations
The sample is one market and one fixed window. DTW is downsampled/constrained.
Portfolio draws share constituents, so their p-values are indicative. No result
is a forecast or investment recommendation.

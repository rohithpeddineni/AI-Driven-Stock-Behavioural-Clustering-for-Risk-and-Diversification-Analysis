# Six-Plus-Cluster Experiment

## Scope
This experiment is isolated from the canonical project. It reads canonical
processed data but writes only below `outputs/six_plus_cluster_experiment/`.

## Methods
Tested `k=6..10`: standardized fingerprint K-Means, RobustScaler K-Means,
PCA K-Means, Ward agglomerative clustering, diagonal-covariance GMM,
correlation-distance average-linkage clustering, and corrected consensus.
Distance-view silhouette is computed with `metric="precomputed"`; Euclidean
Davies-Bouldin and Calinski-Harabasz are marked unavailable for distance views.

DTW is not used as a selected method here because the prior constrained DTW view
had very weak separation and stability. Its prior outputs remain in
`outputs/previous_version/`.

## Selection rule
No candidate is selected by return alone. Compare separation, stability, cluster
sizes, profiles, and diversification. A six-cluster result is only acceptable
if it is not a fragile tiny-cluster decomposition.

## Walk-forward
The correlation view is fitted on the first 70% of returns and evaluated on the
final 30%. This avoids using evaluation-period correlations for that check.

## Results
The complete tables are in `metrics/`, `clusters/`, and `portfolio/`.

### Candidate comparison

* **PCA K-Means, k=6** is the strongest overall six-cluster candidate. It has
  silhouette `0.2075`, Davies-Bouldin `1.3231`, cluster sizes
  `12|4|18|12|27|2`, and mean seed ARI `0.9404`. It is reproducible and has
  better separation and stability than the ordinary fingerprint k=6 result.
  It is not balanced: two clusters contain only 2 and 4 stocks.
* Robust-scaling K-Means reaches higher stability at k=8 (mean ARI `0.9415`)
  but creates singleton clusters and a 27-stock cluster, so it is not preferred
  for an interpretable portfolio universe.
* Ward clustering is perfectly stable for these fixed inputs, but stability
  reflects its deterministic algorithm and its separation is weaker
  (silhouette `0.1369` at k=6).
* Correlation-distance clustering is validly scored with a precomputed
  distance silhouette (`0.1556` at k=6). Its k=6 sizes are highly imbalanced
  (`12|7|2|44|3|7`), limiting its usefulness for representative selection.
* Corrected consensus improves separation (precomputed silhouette `0.4610` at
  k=6), but produces a 39-stock cluster and a singleton. Its high silhouette
  therefore does not imply a useful diversified six-cluster partition.
* GMM is the weakest stable choice in this comparison: k=6 silhouette `0.0979`
  and a singleton cluster.

The selected PCA k=6 cross-cluster portfolio has annualised return `20.36%`,
volatility `19.16%`, maximum drawdown `26.63%`, average pairwise correlation
`0.3020`, and diversification ratio `1.5392` in the common historical sample.
This correlation is lower than the canonical k=2 Portfolio B correlation
`0.3902`, with lower volatility and drawdown than the canonical baseline
(`25.33%` and `27.93%`), but its return is higher only in this in-sample
comparison and is not sufficient evidence of superiority.

### Walk-forward result

The temporal correlation experiment fits on 797 training days and evaluates on
342 later days. For every k from 6 through 10, the selected cross-cluster
portfolio is the same six-stock set and has annualised return `1.75%`,
volatility `15.25%`, maximum drawdown `12.99%`, average correlation `0.1934`,
and diversification ratio `1.7272`. The median eligible same-cluster
portfolio has return `11.43%`, volatility `19.41%`, drawdown `16.71%`,
correlation `0.4984`, and diversification ratio `1.3174`. Thus cross-cluster
selection improves risk and correlation in this out-of-sample window, but
materially sacrifices return. It does not establish a universally better
portfolio.

### Recommendation

Adopt **PCA K-Means k=6 as the experimental candidate**, not as a replacement
for the canonical k=2 model yet. It satisfies the six-cluster requirement and
offers a credible, stable behavioural partition, but the small clusters and
the mixed return/risk evidence require additional rolling windows before
production use. The canonical implementation and outputs remain unchanged.

## Limitations
Fingerprint features are fixed canonical full-window inputs for the main
cross-sectional comparison. The walk-forward check is implemented for the
temporal correlation view; a full rolling feature rebuild would require
recomputing all canonical features per training window.

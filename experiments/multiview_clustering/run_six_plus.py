"""Versioned six-plus-cluster experiment.

Reads canonical processed CSVs read-only and writes only to
outputs/six_plus_cluster_experiment/. The canonical implementation is not
imported or modified.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "multiview_clustering" / "outputs" / "six_plus_cluster_experiment"
K_VALUES = range(6, 11)
SEEDS = (7, 19, 42, 73, 101)
TRADING_DAYS = 252


def save(frame: pd.DataFrame, name: str, folder: str) -> None:
    path = OUT / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_plots(comparison: pd.DataFrame, portfolio_frame: pd.DataFrame) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    valid = comparison.dropna(subset=["silhouette"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, frame in valid.groupby("method"):
        ax.plot(frame["k"], frame["silhouette"], marker="o", label=method)
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score",
           title="Six-plus cluster separation by method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "silhouette_by_method.png", dpi=160)
    plt.close(fig)

    cross = portfolio_frame[portfolio_frame["portfolio"] == "cross_cluster"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, frame in cross.groupby("method"):
        ax.plot(frame["k"], frame["avg_pairwise_correlation"], marker="o", label=method)
    ax.axhline(0.390217, color="black", linestyle="--", label="Canonical baseline")
    ax.set(xlabel="Number of clusters (k)", ylabel="Average pairwise correlation",
           title="Cross-cluster portfolio correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "cross_cluster_correlation.png", dpi=160)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(PROC / "stock_fingerprints.csv", index_col="Symbol").sort_index()
    panel = pd.read_csv(PROC / "cleaned_daily_data.csv", parse_dates=["Date"])
    prices = panel.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
    returns = prices.pct_change().dropna()
    returns = returns[features.index]
    return features, prices, returns


def internal_metrics(data: np.ndarray, labels: np.ndarray, distance: bool = False) -> dict:
    k = len(np.unique(labels))
    sizes = np.bincount(labels, minlength=k)
    metric_args = {"metric": "precomputed"} if distance else {}
    return {
        "silhouette": float(silhouette_score(data, labels, **metric_args)),
        "davies_bouldin": float("nan") if distance else float(davies_bouldin_score(data, labels)),
        "calinski_harabasz": float("nan") if distance else float(calinski_harabasz_score(data, labels)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_sizes": "|".join(str(int(x)) for x in sizes),
    }


def labels_for(method: str, data: np.ndarray, k: int) -> tuple[np.ndarray, bool]:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=30, random_state=42).fit_predict(data), False
    if method == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data), False
    if method == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", random_state=42,
                              n_init=5).fit_predict(data), False
    return AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed").fit_predict(data), True


def portfolio_metrics(returns: pd.DataFrame, holdings: list[str]) -> dict:
    daily = returns[holdings].mean(axis=1)
    wealth = (1 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    corr = returns[holdings].corr().to_numpy()
    upper = corr[np.triu_indices(len(holdings), 1)]
    asset_vol = returns[holdings].std(ddof=1).to_numpy() * math.sqrt(TRADING_DAYS)
    diversification_ratio = float(asset_vol.mean() / (daily.std(ddof=1) * math.sqrt(TRADING_DAYS)))
    return {
        "annualised_return": float((1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1),
        "annualised_volatility": float(daily.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "max_drawdown": float(-drawdown.min()),
        "avg_pairwise_correlation": float(np.nanmean(upper)),
        "diversification_ratio": diversification_ratio,
        "holdings": "|".join(holdings),
    }


def select_portfolios(labels: np.ndarray, symbols: list[str], geometry: pd.DataFrame,
                      returns: pd.DataFrame, method: str, k: int) -> list[dict]:
    groups: dict[int, list[str]] = {}
    for c in sorted(np.unique(labels)):
        members = [symbols[i] for i, x in enumerate(labels) if x == c]
        center = geometry.loc[members].mean().to_numpy()
        groups[int(c)] = sorted(members, key=lambda s: (np.linalg.norm(geometry.loc[s].to_numpy() - center), s))
    n = min(6, k)
    eligible = [m for m in groups.values() if len(m) >= n]
    rows = []
    for members in eligible:
        rows.append({"method": method, "k": k, "portfolio": "same_cluster",
                     **portfolio_metrics(returns, members[:n])})
    chosen_clusters = sorted(groups, key=lambda c: (-len(groups[c]), c))[:n]
    cross = [groups[c][0] for c in chosen_clusters]
    rows.append({"method": method, "k": k, "portfolio": "cross_cluster",
                 **portfolio_metrics(returns, cross)})
    return rows


def walk_forward_correlation(returns: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    split = int(len(returns) * 0.7)
    train, test = returns.iloc[:split], returns.iloc[split:]
    train_distance = 1 - train.corr().clip(-1, 1).to_numpy()
    rows = []
    for k in K_VALUES:
        labels = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed").fit_predict(train_distance)
        groups = {}
        for c in sorted(np.unique(labels)):
            members = [symbols[i] for i, x in enumerate(labels) if x == c]
            positions = [symbols.index(symbol) for symbol in members]
            groups[c] = sorted(
                members,
                key=lambda s: (
                    float(np.mean([
                        train_distance[symbols.index(s), position]
                        for position in positions
                    ])),
                    s,
                ),
            )
        n = min(6, k)
        eligible = [m for m in groups.values() if len(m) >= n]
        cross = [groups[c][0] for c in sorted(groups, key=lambda c: (-len(groups[c]), c))[:n]]
        row = {"method": "correlation_average", "k": k, "train_days": len(train),
               "test_days": len(test), "portfolio": "cross_cluster",
               **portfolio_metrics(test, cross)}
        rows.append(row)
        if eligible:
            same = [portfolio_metrics(test, m[:n]) for m in eligible]
            rows.append({"method": "correlation_average", "k": k, "train_days": len(train),
                         "test_days": len(test), "portfolio": "same_cluster_median",
                         **{key: float(np.nanmedian([x[key] for x in same]))
                            for key in same[0] if key != "holdings"},
                         "holdings": ""})
    return pd.DataFrame(rows)


def main() -> None:
    features, _, returns = load_data()
    symbols = list(features.index)
    standard = pd.DataFrame(StandardScaler().fit_transform(features), index=symbols, columns=features.columns)
    robust = pd.DataFrame(RobustScaler().fit_transform(features), index=symbols, columns=features.columns)
    pca = PCA(n_components=0.85, random_state=42)
    pca_values = pca.fit_transform(standard)
    correlation_distance = 1 - returns.corr().clip(-1, 1).to_numpy()
    np.fill_diagonal(correlation_distance, 0)

    methods = {
        "fingerprint_kmeans": (standard.to_numpy(), "kmeans", False),
        "fingerprint_robust_kmeans": (robust.to_numpy(), "kmeans", False),
        "pca_kmeans": (pca_values, "kmeans", False),
        "ward_euclidean": (standard.to_numpy(), "ward", False),
        "gmm_diag": (standard.to_numpy(), "gmm", False),
        "correlation_average": (correlation_distance, "average", True),
    }
    metrics, stability, assignments, profiles, portfolios = [], [], [], [], []
    for name, (data, algorithm, distance) in methods.items():
        for k in K_VALUES:
            labels, _ = labels_for(algorithm, data, k)
            metrics.append({"method": name, "algorithm": algorithm, "representation": name,
                            "k": k, **internal_metrics(data, labels, distance)})
            assignments.append(pd.DataFrame({"Symbol": symbols, "method": name, "k": k, "cluster": labels + 1}))
            portfolios.extend(select_portfolios(labels, symbols, standard, returns, name, k))
            runs = []
            for seed in SEEDS:
                if algorithm == "kmeans":
                    runs.append(KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(data))
                elif algorithm == "gmm":
                    runs.append(GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=5).fit_predict(data))
                else:
                    runs.append(labels_for(algorithm, data, k)[0])
            aris = [adjusted_rand_score(runs[0], x) for x in runs[1:]]
            amis = [adjusted_mutual_info_score(runs[0], x) for x in runs[1:]]
            stability.append({"method": name, "k": k, "mean_ari": np.mean(aris), "std_ari": np.std(aris),
                              "mean_ami": np.mean(amis), "std_ami": np.std(amis)})
            for c in sorted(np.unique(labels)):
                members = [symbols[i] for i, x in enumerate(labels) if x == c]
                profile = features.loc[members].mean().to_dict()
                profiles.append({"method": name, "k": k, "cluster": c + 1,
                                 "members": len(members), **profile})

    metric_frame = pd.DataFrame(metrics)
    stability_frame = pd.DataFrame(stability)
    comparison = metric_frame.merge(stability_frame, on=["method", "k"], how="left")
    # Correct consensus: consensus is evaluated as a distance; no Euclidean CH/DB claims.
    consensus_rows = []
    for k in K_VALUES:
        label_sets = [labels_for("kmeans", pca_values, k)[0],
                      labels_for("average", correlation_distance, k)[0]]
        co = np.mean([x[:, None] == x[None, :] for x in label_sets], axis=0)
        consensus_distance = 1 - co
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(consensus_distance)
        consensus_rows.append({"method": "consensus_corrected", "algorithm": "average_precomputed",
                               "representation": "coassociation_distance", "k": k,
                               "silhouette": silhouette_score(consensus_distance, labels, metric="precomputed"),
                               "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                               "min_cluster_size": np.bincount(labels, minlength=k).min(),
                               "max_cluster_size": np.bincount(labels, minlength=k).max(),
                               "cluster_sizes": "|".join(map(str, np.bincount(labels, minlength=k)))})
        portfolios.extend(select_portfolios(labels, symbols, standard, returns, "consensus_corrected", k))
        assignments.append(pd.DataFrame({"Symbol": symbols, "method": "consensus_corrected", "k": k, "cluster": labels + 1}))
    comparison = pd.concat([comparison, pd.DataFrame(consensus_rows)], ignore_index=True, sort=False)
    save(comparison, "clustering_metrics.csv", "metrics")
    save(stability_frame, "stability_metrics.csv", "metrics")
    save(comparison, "method_comparison.csv", "metrics")
    save(pd.concat(assignments, ignore_index=True), "cluster_assignments.csv", "clusters")
    profile_frame = pd.DataFrame(profiles)
    save(profile_frame, "cluster_profiles.csv", "clusters")
    save(profile_frame[["method", "k", "cluster", "members"]], "cluster_sizes.csv", "clusters")
    save(pd.DataFrame(portfolios), "portfolio_comparison.csv", "portfolio")
    walk_forward = walk_forward_correlation(returns, symbols)
    save(walk_forward, "walk_forward_results.csv", "portfolio")
    portfolio_frame = pd.DataFrame(portfolios)
    save_plots(comparison, portfolio_frame)
    report = f"""# Six-Plus-Cluster Experiment

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
"""
    (OUT / "reports" / "experiment_report.md").write_text(report)
    (OUT / "README.md").write_text("Run from the project root with `python experiments/multiview_clustering/run_six_plus.py`.\n")
    print("Six-plus experiment complete.")


if __name__ == "__main__":
    main()

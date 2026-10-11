"""Isolated multi-view clustering experiment.

This file reads the canonical processed inputs but writes only below this
experiment directory. It is intentionally separate from the production
pipeline.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = EXP / "outputs"
PROC = ROOT / "data" / "processed"
SEEDS = [7, 19, 42, 73, 101]
K_VALUES = range(2, 11)
TRADING_DAYS = 252


def _write(frame: pd.DataFrame, relative: str) -> None:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _metrics(data: np.ndarray, labels: np.ndarray, method: str, representation: str, k: int) -> dict:
    sizes = np.bincount(labels, minlength=k)
    return {
        "method": method, "representation": representation, "algorithm": method,
        "k": k, "silhouette": silhouette_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels),
        "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
        "cluster_sizes": "|".join(map(str, sizes.tolist())),
    }


def _dtw(a: np.ndarray, b: np.ndarray, window: int = 12) -> float:
    """Constrained DTW on downsampled z-normalized return paths."""
    n, m = len(a), len(b)
    inf = float("inf")
    prev = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, inf)
        lo, hi = max(1, i - window), min(m, i + window)
        for j in range(lo, hi + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / max(n + m, 1))


def _distance_matrix(returns: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "correlation":
        corr = returns.corr().clip(-1, 1).to_numpy()
        distance = 1.0 - corr
        np.fill_diagonal(distance, 0.0)
        return distance
    paths = returns.to_numpy().T
    sampled = []
    for path in paths:
        x = path[np.linspace(0, len(path) - 1, 120).astype(int)]
        x = (x - x.mean()) / (x.std(ddof=1) or 1.0)
        sampled.append(x)
    distance = np.zeros((len(sampled), len(sampled)))
    for i in range(len(sampled)):
        for j in range(i):
            distance[i, j] = distance[j, i] = _dtw(sampled[i], sampled[j])
    return distance


def _distance_labels(distance: np.ndarray, k: int) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    )
    return model.fit_predict(distance)


def _portfolio_metrics(returns: pd.DataFrame, holdings: list[str]) -> dict:
    daily = returns[holdings].mean(axis=1)
    wealth = (1 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    corr = returns[holdings].corr().to_numpy()
    upper = corr[np.triu_indices(len(holdings), 1)]
    return {
        "holdings": "|".join(holdings),
        "annualised_return": (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1,
        "annualised_volatility": daily.std(ddof=1) * math.sqrt(TRADING_DAYS),
        "max_drawdown": -drawdown.min(),
        "avg_pairwise_correlation": float(np.nanmean(upper)) if len(upper) else np.nan,
    }


def _evaluate_portfolios(labels: np.ndarray, symbols: list[str], scaled: pd.DataFrame,
                         returns: pd.DataFrame, method: str, k: int) -> list[dict]:
    groups = {}
    for cluster in sorted(set(labels)):
        members = [symbols[i] for i, value in enumerate(labels) if value == cluster]
        center = scaled.loc[members].mean().to_numpy()
        ranked = sorted(members, key=lambda s: (np.linalg.norm(scaled.loc[s].to_numpy() - center), s))
        groups[cluster] = ranked
    n = min(5, k)
    eligible = [members for members in groups.values() if len(members) >= n]
    rows = []
    if eligible:
        a = [_portfolio_metrics(returns, members[:n]) for members in eligible]
        rows.append({"method": method, "k": k, "portfolio": "same_cluster_median",
                     **{key: float(np.nanmedian([x[key] for x in a])) for key in a[0] if key != "holdings"}})
    selected = sorted(groups, key=lambda c: (-len(groups[c]), c))[:n]
    b_holdings = [groups[c][0] for c in selected]
    rows.append({"method": method, "k": k, "portfolio": "cross_cluster",
                 **_portfolio_metrics(returns, b_holdings)})
    return rows


def main() -> None:
    fingerprints = pd.read_csv(PROC / "stock_fingerprints.csv", index_col="Symbol").sort_index()
    scaled_values = StandardScaler().fit_transform(fingerprints)
    scaled = pd.DataFrame(scaled_values, index=fingerprints.index, columns=fingerprints.columns)
    panel = pd.read_csv(PROC / "cleaned_daily_data.csv", parse_dates=["Date"])
    prices = panel.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
    returns = prices.pct_change().dropna()
    symbols = list(fingerprints.index)
    returns = returns[symbols]

    pca = PCA(n_components=0.85, random_state=42)
    pca_values = pca.fit_transform(scaled_values)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(scaled_values)
    _write(pd.DataFrame({"Symbol": symbols, "PC1": pca2[:, 0], "PC2": pca2[:, 1]}),
           "clusters/pca_coordinates.csv")

    metrics, assignments, profiles, portfolios = [], [], [], []
    candidate_labels: dict[tuple[str, int], np.ndarray] = {}
    representations = {
        "fingerprint_raw": ("kmeans", scaled_values),
        "fingerprint_pca": ("kmeans", pca_values),
    }
    for name, (algorithm, data) in representations.items():
        for k in K_VALUES:
            labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(data)
            candidate_labels[(name, k)] = labels
            metrics.append(_metrics(data, labels, "kmeans", name, k))
            portfolios.extend(_evaluate_portfolios(labels, symbols, scaled, returns, name, k))

    distance_views = {}
    for name in ("correlation", "dtw"):
        distance = _distance_matrix(returns, name)
        distance_views[name] = distance
        np.save(OUT / "metrics" / f"{name}_distance.npy", distance)
        for k in K_VALUES:
            labels = _distance_labels(distance, k)
            candidate_labels[(name, k)] = labels
            # Silhouette/DB are evaluated against the precomputed distance.
            row = _metrics(distance, labels, "agglomerative", name, k)
            row["silhouette"] = silhouette_score(distance, labels, metric="precomputed")
            metrics.append(row)
            portfolios.extend(_evaluate_portfolios(labels, symbols, scaled, returns, name, k))

    # Consensus is a co-association similarity over the strongest independent
    # views at k=2..10; equal weights avoid an arbitrary preference.
    consensus_rows = []
    for k in K_VALUES:
        label_sets = [candidate_labels[("fingerprint_pca", k)],
                      candidate_labels[("correlation", k)],
                      candidate_labels[("dtw", k)]]
        co = np.mean([labels[:, None] == labels[None, :] for labels in label_sets], axis=0)
        labels = _distance_labels(1.0 - co, k)
        candidate_labels[("consensus", k)] = labels
        row = _metrics(1.0 - co, labels, "agglomerative", "consensus", k)
        row["silhouette"] = silhouette_score(1.0 - co, labels, metric="precomputed")
        metrics.append(row)
        portfolios.extend(_evaluate_portfolios(labels, symbols, scaled, returns, "consensus", k))
        consensus_rows.append(row)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "metrics" / "clustering_metrics.csv", index=False)
    # Stability: repeated K-Means seeds and bootstrap-like temporal halves.
    stability = []
    for name, (algorithm, data) in representations.items():
        for k in K_VALUES:
            runs = [KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(data) for seed in SEEDS]
            aris = [adjusted_rand_score(runs[0], run) for run in runs[1:]]
            amis = [adjusted_mutual_info_score(runs[0], run) for run in runs[1:]]
            stability.append({"method": "kmeans", "representation": name, "k": k,
                              "mean_ari": np.mean(aris), "std_ari": np.std(aris),
                              "mean_ami": np.mean(amis), "std_ami": np.std(amis)})
    for name, distance in distance_views.items():
        for k in K_VALUES:
            first = _distance_labels(distance, k)
            half = _distance_labels(_distance_matrix(returns.iloc[:len(returns)//2], name), k)
            stability.append({"method": "agglomerative", "representation": name, "k": k,
                              "mean_ari": adjusted_rand_score(first, half),
                              "std_ari": 0.0, "mean_ami": adjusted_mutual_info_score(first, half),
                              "std_ami": 0.0})
    _write(pd.DataFrame(stability), "metrics/stability_metrics.csv")
    _write(pd.DataFrame(portfolios), "portfolio/portfolio_results.csv")

    # Save assignments and behavioural profiles for every view's metric-best k.
    selected = []
    for representation in ("fingerprint_raw", "fingerprint_pca", "correlation", "dtw", "consensus"):
        subset = metric_frame[metric_frame.representation == representation].copy()
        stable = pd.DataFrame(stability)
        stable = stable[stable.representation == representation]
        if not stable.empty:
            subset = subset.merge(stable[["k", "mean_ari"]], on="k", how="left")
        subset["score"] = subset["silhouette"].rank(pct=True) + subset["davies_bouldin"].rank(ascending=False, pct=True)
        best = subset.sort_values(["score", "k"], ascending=[False, True]).iloc[0]
        best_k = int(best.k)
        selected.append({"representation": representation, "k": best_k,
                         "silhouette": best.silhouette, "davies_bouldin": best.davies_bouldin,
                         "mean_ari": best.get("mean_ari", np.nan)})
        labels = candidate_labels[(representation, best_k)]
        assignments.append(pd.DataFrame({"Symbol": symbols, "representation": representation,
                                          "k": best_k, "cluster": labels + 1}))
        for cluster in sorted(set(labels)):
            members = [symbols[i] for i, value in enumerate(labels) if value == cluster]
            stats = fingerprints.loc[members].mean().to_dict()
            profiles.append({"representation": representation, "k": best_k, "cluster": cluster + 1,
                             "members": len(members), **stats})
    _write(pd.concat(assignments, ignore_index=True), "clusters/cluster_assignments_selected.csv")
    _write(pd.DataFrame(profiles), "clusters/cluster_profiles_selected.csv")
    comparison = metric_frame.merge(pd.DataFrame(stability), on=["representation", "k"], how="left")
    comparison.to_csv(OUT / "metrics" / "method_comparison.csv", index=False)
    for representation in metric_frame.representation.unique():
        subset = metric_frame[metric_frame.representation == representation]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(subset["k"], subset["silhouette"], marker="o")
        axes[0].set(title=f"{representation}: silhouette", xlabel="k")
        axes[1].plot(subset["k"], subset["davies_bouldin"], marker="o")
        axes[1].set(title=f"{representation}: Davies-Bouldin", xlabel="k")
        fig.tight_layout()
        fig.savefig(OUT / "plots" / f"metrics_{representation}.png", dpi=140)
        plt.close(fig)
    plt.figure(figsize=(6, 5))
    plt.imshow(1 - distance_views["correlation"], cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation distance")
    plt.title("Return-correlation distance")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "similarity_heatmap_correlation.png", dpi=140)
    plt.close()
    report = OUT / "reports" / "experiment_report.md"
    report.write_text(_report(metric_frame, pd.DataFrame(selected), pd.DataFrame(portfolios), pca))
    (EXP / "README.md").write_text(
        "# Isolated Multi-View Clustering Experiment\n\n"
        "Run `python run_experiment.py` from this directory. The experiment reads "
        "the canonical processed CSVs read-only and writes only under `outputs/`. "
        "The constrained DTW view uses 120 evenly sampled z-normalized return "
        "observations and a Sakoe-Chiba window of 12 because full 1,140-point "
        "DTW is unnecessary for this exploratory 75-stock comparison.\n"
    )


def _report(metrics: pd.DataFrame, selected: pd.DataFrame, portfolios: pd.DataFrame, pca: PCA) -> str:
    top = selected.to_string(index=False)
    return f"""# Multi-View Clustering Experiment Report

## Objective
This isolated experiment tests whether temporal/co-movement information improves
on the canonical 14-feature PCA + K-Means baseline without modifying it.

## Methods
- Statistical raw fingerprint: StandardScaler + K-Means, k=2..10.
- Statistical PCA fingerprint: PCA retaining {pca.n_components_} components
  ({pca.explained_variance_ratio_.sum():.2%} variance) + K-Means.
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
{top}
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
"""


if __name__ == "__main__":
    main()
    print("Isolated multi-view experiment complete.")

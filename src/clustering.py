"""Scale fingerprints, compare raw/PCA K-Means, and persist Stage 4 outputs.

The raw standardized fingerprint space is canonical unless PCA is materially
better on both silhouette and Davies-Bouldin.  PCA coordinates used for plots
are always fitted separately from the retained-variance PCA used for comparison.

Run with ``python -m src.clustering``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src import config

Space = Literal["raw", "pca"]


def load_fingerprints() -> pd.DataFrame:
    """Load the final, feature-quality-screened fingerprint table."""
    if not config.FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(f"Missing fingerprints: {config.FINGERPRINTS_PATH}")
    fingerprints = pd.read_csv(config.FINGERPRINTS_PATH, index_col="Symbol")
    if fingerprints.empty or fingerprints.index.has_duplicates:
        raise ValueError("Fingerprint table is empty or has duplicate symbols")
    if not np.isfinite(fingerprints.to_numpy(dtype=float)).all():
        raise ValueError("Fingerprint table contains missing or infinite values")
    return fingerprints.sort_index()


def _metrics(data: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Calculate the three internal validity metrics for one clustering."""
    return (
        float(silhouette_score(data, labels)),
        float(davies_bouldin_score(data, labels)),
        float(calinski_harabasz_score(data, labels)),
    )


def _valid_k_values(n_samples: int) -> list[int]:
    """Return configured k values that can produce non-empty clusters."""
    return [k for k in config.K_RANGE if 2 <= k < n_samples]


def _select_k(metrics: pd.DataFrame, space: Space) -> int:
    """Choose k by ranked metrics, a small complexity penalty, and tie rules.

    Candidates with a cluster smaller than two members are not interpretable.
    Among the remaining candidates, the mean percentile rank of silhouette,
    inverse Davies-Bouldin, and Calinski-Harabasz is maximised; the configured
    complexity penalty favours the simpler solution when evidence is close.
    """
    subset = metrics[
        (metrics["space"] == space)
        & (metrics["min_cluster_size"] >= config.MIN_CLUSTER_SIZE)
    ].copy()
    if subset.empty:
        subset = metrics[metrics["space"] == space].copy()
    if subset.empty:
        raise ValueError(f"No valid {space} clustering candidates")
    # Percentile scores are oriented so that 1.0 is always better.
    for column, ascending in (
        ("silhouette", True), ("davies_bouldin", False), ("calinski_harabasz", True)
    ):
        subset[f"rank_{column}"] = subset[column].rank(method="min", ascending=ascending,
                                                       pct=True)
    k_min, k_max = subset["k"].min(), subset["k"].max()
    span = max(k_max - k_min, 1)
    subset["selection_score"] = (
        subset[["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]].mean(axis=1)
        - config.K_COMPLEXITY_PENALTY * (subset["k"] - k_min) / span
    )
    return int(subset.sort_values(["selection_score", "k"], ascending=[False, True]).iloc[0]["k"])


def _pca_materially_better(metrics: pd.DataFrame, raw_k: int, pca_k: int) -> bool:
    """Apply the documented, config-driven PCA-over-raw canonical-space rule."""
    raw = metrics[(metrics.space == "raw") & (metrics.k == raw_k)].iloc[0]
    pca = metrics[(metrics.space == "pca") & (metrics.k == pca_k)].iloc[0]
    db_reduction = (raw.davies_bouldin - pca.davies_bouldin) / max(abs(raw.davies_bouldin), 1e-12)
    return (
        pca.silhouette - raw.silhouette >= config.PCA_MATERIAL_SILHOUETTE_GAIN
        and db_reduction >= config.PCA_MATERIAL_DB_REDUCTION
    )


def run_clustering() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run deterministic raw/PCA K-Means and write metrics, labels, and coordinates."""
    fingerprints = load_fingerprints()
    scaled = StandardScaler().fit_transform(fingerprints)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pca_retained = PCA(n_components=config.PCA_VARIANCE_TARGET, random_state=config.RANDOM_STATE)
    retained = pca_retained.fit_transform(scaled)
    pca_plot = PCA(n_components=config.PCA_PLOT_COMPONENTS, random_state=config.RANDOM_STATE)
    coordinates = pca_plot.fit_transform(scaled)

    valid_k = _valid_k_values(len(fingerprints))
    if not valid_k:
        raise ValueError("Not enough fingerprints for configured K_RANGE")
    labels_by_space: dict[tuple[Space, int], np.ndarray] = {}
    rows: list[dict[str, float | int | str]] = []
    for space, data in (("raw", scaled), ("pca", retained)):
        for k in valid_k:
            model = KMeans(
                n_clusters=k, random_state=config.RANDOM_STATE, n_init=config.KMEANS_N_INIT
            )
            labels = model.fit_predict(data)
            labels_by_space[(space, k)] = labels
            silhouette, db, ch = _metrics(data, labels)
            rows.append({
                "space": space, "k": k, "silhouette": silhouette,
                "davies_bouldin": db, "calinski_harabasz": ch,
                "min_cluster_size": int(np.bincount(labels, minlength=k).min()),
            })
    metrics = pd.DataFrame(rows)
    raw_k = _select_k(metrics, "raw")
    pca_k = _select_k(metrics, "pca")
    canonical_space: Space = "pca" if _pca_materially_better(metrics, raw_k, pca_k) else "raw"
    canonical_k = pca_k if canonical_space == "pca" else raw_k
    metrics["ari_raw_vs_pca"] = metrics["k"].map(
        lambda k: adjusted_rand_score(labels_by_space[("raw", int(k))],
                                       labels_by_space[("pca", int(k))])
    )
    metrics["raw_selected_k"] = raw_k
    metrics["pca_selected_k"] = pca_k
    metrics["canonical_space"] = canonical_space
    metrics["canonical_k"] = canonical_k

    assignments = pd.DataFrame(index=fingerprints.index)
    assignments.index.name = "Symbol"
    assignments["pca_1"] = coordinates[:, 0]
    assignments["pca_2"] = coordinates[:, 1]
    for space in ("raw", "pca"):
        for k in valid_k:
            assignments[f"{space}_cluster_k{k}"] = labels_by_space[(space, k)] + 1
    assignments["canonical_space"] = canonical_space
    assignments["canonical_k"] = canonical_k
    assignments["cluster"] = labels_by_space[(canonical_space, canonical_k)] + 1

    scaled_frame = pd.DataFrame(scaled, index=fingerprints.index, columns=fingerprints.columns)
    scaled_frame.index.name = "Symbol"
    pca_frame = pd.DataFrame(coordinates, index=fingerprints.index,
                              columns=["pca_1", "pca_2"])
    pca_frame.index.name = "Symbol"
    retained_frame = pd.DataFrame(
        retained, index=fingerprints.index,
        columns=[f"pca_component_{i}" for i in range(1, retained.shape[1] + 1)],
    )
    retained_frame.index.name = "Symbol"
    pca_summary = pd.DataFrame({
        "component": np.arange(1, len(pca_retained.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_retained.explained_variance_ratio_,
    })
    pca_summary["cumulative_explained_variance"] = pca_summary[
        "explained_variance_ratio"
    ].cumsum()
    pca_summary["target_variance"] = config.PCA_VARIANCE_TARGET
    metrics.to_csv(config.CLUSTER_METRICS_PATH, index=False)
    assignments.to_csv(config.CLUSTER_ASSIGNMENTS_PATH)
    scaled_frame.to_csv(config.CLUSTER_SCALED_PATH)
    pca_frame.to_csv(config.CLUSTER_PCA_PATH)
    retained_frame.to_csv(config.CLUSTER_PCA_RETAINED_PATH)
    pca_summary.to_csv(config.CLUSTER_PCA_SUMMARY_PATH, index=False)
    return metrics, assignments, pca_frame


if __name__ == "__main__":
    config.configure_console()
    metrics, assignments, _ = run_clustering()
    first = metrics.iloc[0]
    print(f"Clustered {len(assignments)} stocks across {metrics['k'].nunique()} k values.")
    print(f"Canonical: {first['canonical_space']} space, k={int(first['canonical_k'])}")
    print(f"Cluster sizes: {assignments['cluster'].value_counts().sort_index().to_dict()}")
    print(f"Wrote {config.CLUSTER_METRICS_PATH.name}, {config.CLUSTER_ASSIGNMENTS_PATH.name}, "
          f"{config.CLUSTER_PCA_PATH.name}, {config.CLUSTER_PCA_RETAINED_PATH.name}, "
          f"{config.CLUSTER_PCA_SUMMARY_PATH.name}, and {config.CLUSTER_SCALED_PATH.name}")

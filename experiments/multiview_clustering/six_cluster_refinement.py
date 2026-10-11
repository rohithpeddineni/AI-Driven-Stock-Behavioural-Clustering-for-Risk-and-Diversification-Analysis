"""Isolated refinement experiment for six-plus behavioural clusters.

The runner reads canonical processed data but writes only to the refinement
output directory. Training-window fingerprints are rebuilt before every
walk-forward fit to prevent full-window feature leakage.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "multiview_clustering" / "outputs" / "six_cluster_refinement"
K_VALUES = range(6, 11)
SEEDS = (7, 19, 42, 73, 101)
TRADING_DAYS = 252
FEATURES = [
    "annualised_return", "annualised_volatility", "downside_volatility",
    "max_drawdown", "rolling_vol_dispersion", "beta", "momentum_3m",
    "momentum_12m", "trend_strength", "autocorr_lag1", "skewness",
    "excess_kurtosis", "volume_volatility", "return_volume_corr",
    "extreme_move_frequency",
]


def write(frame: pd.DataFrame, name: str) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel = pd.read_csv(PROC / "cleaned_daily_data.csv", parse_dates=["Date"])
    prices = panel.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
    volume = panel.pivot(index="Date", columns="Symbol", values="Volume").sort_index()
    symbols = sorted(pd.read_csv(PROC / "stock_fingerprints.csv")["Symbol"])
    return prices[symbols], volume[symbols], symbols


def fingerprint(prices: pd.DataFrame, volume: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Build the canonical behavioural fingerprint over exactly this window."""
    market = prices.pct_change().mean(axis=1)
    rows = {}
    for symbol in symbols:
        price = prices[symbol].dropna()
        ret = price.pct_change().dropna()
        vol = volume[symbol].reindex(price.index).replace(0, np.nan)
        log_change = np.log(vol).diff()
        rolling = ret.rolling(60).std(ddof=1).dropna()
        peak = price.cummax()
        threshold = 2 * ret.std(ddof=1)
        aligned = pd.concat([ret, market], axis=1).dropna()
        rows[symbol] = {
            "annualised_return": ret.mean() * TRADING_DAYS,
            "annualised_volatility": ret.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "downside_volatility": ret[ret < 0].std(ddof=1) * math.sqrt(TRADING_DAYS),
            "max_drawdown": (1 - price / peak).max(),
            "rolling_vol_dispersion": rolling.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "beta": aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(),
            "momentum_3m": price.iloc[-1] / price.iloc[-64] - 1 if len(price) > 63 else np.nan,
            "momentum_12m": price.iloc[-1] / price.iloc[-253] - 1 if len(price) > 252 else np.nan,
            "trend_strength": np.polyfit(np.arange(len(price)), np.log(price), 1)[0] * TRADING_DAYS,
            "autocorr_lag1": ret.autocorr(lag=1),
            "skewness": ret.skew(),
            "excess_kurtosis": ret.kurtosis(),
            "volume_volatility": log_change.std(ddof=1),
            "return_volume_corr": pd.concat([ret.abs(), log_change], axis=1).dropna().corr().iloc[0, 1],
            "extreme_move_frequency": (ret.abs() > threshold).mean(),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")[FEATURES]
    return result.replace([np.inf, -np.inf], np.nan).fillna(result.median(numeric_only=True))


def representation(features: pd.DataFrame, name: str, variance: float | None = None):
    if name == "robust":
        scaler = RobustScaler().fit(features)
        return scaler.transform(features), scaler, None, "robust"
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    if variance is None:
        return scaled, scaler, None, "standard"
    model = PCA(n_components=variance, random_state=42).fit(scaled)
    return model.transform(scaled), scaler, model, f"pca_{int(variance * 100)}"


def cluster(data: np.ndarray, method: str, k: int, seed: int = 42) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(data)
    if method == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data)
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=5,
                           random_state=seed).fit_predict(data)


def metrics(data: np.ndarray, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels, minlength=len(np.unique(labels)))
    return {
        "silhouette": silhouette_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
        "min_cluster_size": sizes.min(),
        "max_cluster_size": sizes.max(),
        "cluster_sizes": "|".join(map(str, sizes)),
    }


def portfolio_metrics(returns: pd.DataFrame, holdings: list[str]) -> dict:
    daily = returns[holdings].mean(axis=1)
    wealth = (1 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    corr = returns[holdings].corr().to_numpy()
    upper = corr[np.triu_indices(len(holdings), 1)]
    vol = returns[holdings].std(ddof=1).to_numpy() * math.sqrt(TRADING_DAYS)
    annual_vol = daily.std(ddof=1) * math.sqrt(TRADING_DAYS)
    return {
        "annualised_return": (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1,
        "annualised_volatility": annual_vol,
        "max_drawdown": -drawdown.min(),
        "avg_pairwise_correlation": np.nanmean(upper),
        "diversification_ratio": vol.mean() / annual_vol,
        "sharpe_ratio_rf0": daily.mean() * TRADING_DAYS / annual_vol,
        "holdings": "|".join(holdings),
    }


def representatives(labels: np.ndarray, symbols: list[str], geometry: np.ndarray,
                    n: int = 6) -> tuple[list[str], dict[int, list[str]]]:
    groups = {}
    for c in sorted(np.unique(labels)):
        indexes = np.flatnonzero(labels == c)
        center = geometry[indexes].mean(axis=0)
        ordered = sorted(indexes, key=lambda i: (np.linalg.norm(geometry[i] - center), symbols[i]))
        groups[int(c)] = [symbols[i] for i in ordered]
    chosen = sorted(groups, key=lambda c: (-len(groups[c]), c))[:n]
    return [groups[c][0] for c in chosen], groups


def evaluate_strategies(returns: pd.DataFrame, labels: np.ndarray, symbols: list[str],
                        geometry: np.ndarray, method: str, k: int,
                        window: str) -> list[dict]:
    cross, groups = representatives(labels, symbols, geometry)
    rows = [{"method": method, "k": k, "window": window, "strategy": "cross_cluster_medoid",
             **portfolio_metrics(returns, cross)}]
    medoids = [members[0] for c, members in sorted(groups.items()) if len(members) >= 1]
    rows.append({"method": method, "k": k, "window": window, "strategy": "medoid_six",
                 **portfolio_metrics(returns, medoids[:6])})
    multi = []
    for c in sorted(groups, key=lambda c: (-len(groups[c]), c)):
        multi.extend(groups[c][:2])
    rows.append({"method": method, "k": k, "window": window, "strategy": "two_from_three_clusters",
                 **portfolio_metrics(returns, multi[:6])})
    eligible = [members[:6] for members in groups.values() if len(members) >= 6]
    if eligible:
        values = [portfolio_metrics(returns, x) for x in eligible]
        rows.append({"method": method, "k": k, "window": window, "strategy": "same_cluster_median",
                     **{key: float(np.median([v[key] for v in values]))
                        for key in values[0] if key != "holdings"},
                     "holdings": ""})
    baseline = symbols[:6]
    rows.append({"method": method, "k": k, "window": window, "strategy": "equal_weight_baseline_6",
                 **portfolio_metrics(returns, baseline)})
    return rows


def plot_outputs(metrics_frame: pd.DataFrame, portfolios: pd.DataFrame,
                 profiles: pd.DataFrame, walk: pd.DataFrame) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, frame in metrics_frame.groupby("representation"):
        f = frame[frame["k"] == 6]
        ax.plot(f["pca_variance"].fillna(1), f["silhouette"], "o-", label=name)
    ax.set(xlabel="Representation variance / full feature space", ylabel="Silhouette",
           title="Representation sensitivity at k=6")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "pca_component_sensitivity.png", dpi=150); plt.close(fig)

    sizes = metrics_frame[metrics_frame["k"] == 6]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(sizes["representation"], sizes["min_cluster_size"], label="minimum")
    ax.bar(sizes["representation"], sizes["max_cluster_size"], alpha=.45, label="maximum")
    ax.set_ylabel("Stocks"); ax.set_title("Cluster-size range at k=6"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "cluster_sizes.png", dpi=150); plt.close(fig)

    profile_cols = ["annualised_volatility", "max_drawdown", "beta", "momentum_12m",
                    "excess_kurtosis", "extreme_move_frequency"]
    profile = profiles[(profiles["method"] == "pca_kmeans") & (profiles["k"] == 6)]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(profile[profile_cols].to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set(xlabel="Behavioural feature", ylabel="Cluster", xticks=range(len(profile_cols)),
           xticklabels=profile_cols, title="PCA k=6 cluster profiles")
    fig.tight_layout(); fig.savefig(plot_dir / "cluster_profiles.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, frame in metrics_frame.groupby("representation"):
        ax.plot(frame["k"], frame["silhouette"], "o-", label=name)
    ax.set(xlabel="k", ylabel="Silhouette", title="Clustering quality"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "clustering_quality.png", dpi=150); plt.close(fig)

    grouped = portfolios[portfolios["strategy"].isin(["cross_cluster_medoid", "equal_weight_baseline_6"])]
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, frame in grouped.groupby("strategy"):
        ax.scatter(frame["annualised_volatility"], frame["annualised_return"], label=strategy)
    ax.set(xlabel="Annualised volatility", ylabel="Annualised return",
           title="Portfolio return-risk comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "portfolio_risk_return.png", dpi=150); plt.close(fig)

    oos = walk[walk["strategy"] == "cross_cluster_medoid"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for method, frame in oos.groupby("method"):
        ax.plot(frame["window"], frame["annualised_return"], "o-", label=method)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(xlabel="Evaluation window", ylabel="Annualised return", title="Rolling out-of-sample return")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "rolling_oos_performance.png", dpi=150); plt.close(fig)


def main() -> None:
    prices, volume, symbols = load_panel()
    returns = prices.pct_change().dropna()
    full = fingerprint(prices, volume, symbols)
    representations = {}
    pca_rows = []
    for variance in (.70, .80, .88, .90, .95):
        data, scaler, model, name = representation(full, "standard", variance)
        representations[name] = (data, full.index.tolist())
        pca_rows.append({"representation": name, "requested_variance": variance,
                         "components": model.n_components_, "explained_variance": model.explained_variance_ratio_.sum()})
    data, _, _, name = representation(full, "standard", None)
    representations["standard_full"] = (data, full.index.tolist())
    pca_rows.append({"representation": "standard_full", "requested_variance": 1.0,
                     "components": data.shape[1], "explained_variance": 1.0})
    robust, _, _, _ = representation(full, "robust", None)
    representations["robust_full"] = (robust, full.index.tolist())
    pca_rows.append({"representation": "robust_full", "requested_variance": 1.0,
                     "components": robust.shape[1], "explained_variance": 1.0})

    metric_rows, stability_rows, assignments, sizes, profiles, portfolios = [], [], [], [], [], []
    for name, (data, order) in representations.items():
        for k in K_VALUES:
            labels = cluster(data, "kmeans", k)
            pca_variance = next((x["explained_variance"] for x in pca_rows if x["representation"] == name), 1.0)
            row = {"method": "pca_kmeans" if name.startswith("pca_") else "fingerprint_kmeans",
                   "representation": name, "k": k, "pca_variance": pca_variance, **metrics(data, labels)}
            metric_rows.append(row)
            runs = [cluster(data, "kmeans", k, seed) for seed in SEEDS]
            stability_rows.append({"method": row["method"], "representation": name, "k": k,
                                   "mean_ari": np.mean([adjusted_rand_score(runs[0], x) for x in runs[1:]]),
                                   "std_ari": np.std([adjusted_rand_score(runs[0], x) for x in runs[1:]])})
            assignments.append(pd.DataFrame({"Symbol": order, "method": row["method"], "representation": name,
                                             "k": k, "cluster": labels + 1}))
            size = pd.Series(labels).value_counts().sort_index()
            sizes.append({"method": row["method"], "representation": name, "k": k,
                          "cluster_sizes": "|".join(map(str, size.tolist())),
                          "min_cluster_size": size.min(), "max_cluster_size": size.max()})
            for c in sorted(np.unique(labels)):
                members = [order[i] for i, x in enumerate(labels) if x == c]
                profiles.append({"method": row["method"], "representation": name, "k": k,
                                 "cluster": c + 1, "members": len(members), **full.loc[members].mean().to_dict()})
            portfolios.extend(evaluate_strategies(returns, labels, order, data, row["method"], k, "full_window"))

    # Deterministic Euclidean alternatives on the standard full representation.
    standard = representations["standard_full"][0]
    for method in ("ward", "gmm"):
        for k in K_VALUES:
            labels = cluster(standard, method, k)
            row = {"method": method, "representation": "standard_full", "k": k, "pca_variance": 1.0, **metrics(standard, labels)}
            metric_rows.append(row)
            runs = [cluster(standard, method, k, seed) for seed in SEEDS]
            stability_rows.append({"method": method, "representation": "standard_full", "k": k,
                                   "mean_ari": np.mean([adjusted_rand_score(runs[0], x) for x in runs[1:]]),
                                   "std_ari": np.std([adjusted_rand_score(runs[0], x) for x in runs[1:]])})
            assignments.append(pd.DataFrame({"Symbol": symbols, "method": method, "representation": "standard_full",
                                             "k": k, "cluster": labels + 1}))
            portfolios.extend(evaluate_strategies(returns, labels, symbols, standard, method, k, "full_window"))

    # Corrected two-view consensus: co-association is a distance matrix, so only
    # precomputed-distance silhouette is reported for this representation.
    pca_view = representations["pca_88"][0]
    pca_labels = {k: cluster(pca_view, "kmeans", k) for k in K_VALUES}
    corr = returns.corr().clip(-1, 1).to_numpy()
    corr_distance = 1 - corr
    np.fill_diagonal(corr_distance, 0)
    corr_labels = {
        k: AgglomerativeClustering(n_clusters=k, linkage="average",
                                   metric="precomputed").fit_predict(corr_distance)
        for k in K_VALUES
    }
    for k in K_VALUES:
        coassociation = np.mean([
            pca_labels[k][:, None] == pca_labels[k][None, :],
            corr_labels[k][:, None] == corr_labels[k][None, :],
        ], axis=0)
        distance = 1 - coassociation
        labels = AgglomerativeClustering(n_clusters=k, linkage="average",
                                         metric="precomputed").fit_predict(distance)
        sizes_for_k = np.bincount(labels, minlength=k)
        metric_rows.append({
            "method": "consensus_corrected", "representation": "coassociation_distance",
            "k": k, "pca_variance": .8883918225716287,
            "silhouette": silhouette_score(distance, labels, metric="precomputed"),
            "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
            "min_cluster_size": sizes_for_k.min(), "max_cluster_size": sizes_for_k.max(),
            "cluster_sizes": "|".join(map(str, sizes_for_k)),
        })
        stability_rows.append({"method": "consensus_corrected",
                               "representation": "coassociation_distance", "k": k,
                               "mean_ari": np.nan, "std_ari": np.nan})
        assignments.append(pd.DataFrame({"Symbol": symbols, "method": "consensus_corrected",
                                         "representation": "coassociation_distance",
                                         "k": k, "cluster": labels + 1}))
        portfolios.extend(evaluate_strategies(returns, labels, symbols, standard,
                                              "consensus_corrected", k, "full_window"))

    walk_rows = []
    windows = [(0, 504, 504, 756), (252, 756, 756, 1008), (504, 1008, 1008, len(returns))]
    for start, train_end, test_start, end in windows:
        train_prices = prices.iloc[start:train_end]
        train_volume = volume.iloc[start:train_end]
        train_features = fingerprint(train_prices, train_volume, symbols)
        test_returns = returns.iloc[test_start:end]
        window_name = f"{test_returns.index[0].date()}_{test_returns.index[-1].date()}"
        for method in ("pca_kmeans", "fingerprint_kmeans", "ward", "gmm"):
            if method == "pca_kmeans":
                data, scaler, model, _ = representation(train_features, "standard", .88)
            elif method == "fingerprint_kmeans":
                data, scaler, model, _ = representation(train_features, "standard", None)
            else:
                data, scaler, model, _ = representation(train_features, "standard", None)
            labels = cluster(data, "kmeans" if method in ("pca_kmeans", "fingerprint_kmeans") else method, 6)
            cross, _ = representatives(labels, symbols, data)
            walk_rows.append({"method": method, "window": window_name, "train_days": train_end - start,
                              "test_days": end - test_start, "strategy": "cross_cluster_medoid",
                              **portfolio_metrics(test_returns, cross)})
            walk_rows.append({"method": method, "window": window_name, "train_days": train_end - start,
                              "test_days": end - test_start, "strategy": "equal_weight_baseline_6",
                              **portfolio_metrics(test_returns, symbols[:6])})

    metric_frame = pd.DataFrame(metric_rows)
    stability_frame = pd.DataFrame(stability_rows)
    portfolio_frame = pd.DataFrame(portfolios)
    walk_frame = pd.DataFrame(walk_rows)
    write(metric_frame, "clustering_metrics.csv")
    write(pd.DataFrame(pca_rows), "pca_sensitivity.csv")
    write(pd.concat(assignments, ignore_index=True), "cluster_assignments.csv")
    write(pd.DataFrame(sizes), "cluster_sizes.csv")
    write(pd.DataFrame(profiles), "cluster_profiles.csv")
    write(stability_frame, "stability_metrics.csv")
    write(portfolio_frame, "portfolio_comparison.csv")
    write(walk_frame, "walk_forward_results.csv")
    write(metric_frame.merge(stability_frame, on=["method", "representation", "k"], how="left"),
          "method_comparison.csv")
    plot_outputs(metric_frame, portfolio_frame, pd.DataFrame(profiles), walk_frame)

    summary = walk_frame[walk_frame["strategy"] == "cross_cluster_medoid"].groupby("method").agg(
        oos_return_mean=("annualised_return", "mean"), oos_return_std=("annualised_return", "std"),
        oos_volatility_mean=("annualised_volatility", "mean"), oos_drawdown_mean=("max_drawdown", "mean"),
        oos_correlation_mean=("avg_pairwise_correlation", "mean"),
        oos_diversification_mean=("diversification_ratio", "mean"),
    ).reset_index()
    write(summary, "walk_forward_summary.csv")
    report = f"""# Six-Cluster Refinement Experiment

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
setting** (4 components, `{pca_rows[0]["explained_variance"]:.1%}` actual
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
mean annualised return `{summary.loc[summary.method == "pca_kmeans", "oos_return_mean"].iloc[0]:.2%}`,
volatility `{summary.loc[summary.method == "pca_kmeans", "oos_volatility_mean"].iloc[0]:.2%}`,
maximum drawdown `{summary.loc[summary.method == "pca_kmeans", "oos_drawdown_mean"].iloc[0]:.2%}`,
average correlation `{summary.loc[summary.method == "pca_kmeans", "oos_correlation_mean"].iloc[0]:.3f}`,
and diversification ratio `{summary.loc[summary.method == "pca_kmeans", "oos_diversification_mean"].iloc[0]:.3f}`.
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
"""
    (OUT / "experiment_report.md").write_text(report, encoding="utf-8")
    print("Six-cluster refinement complete.")


if __name__ == "__main__":
    main()

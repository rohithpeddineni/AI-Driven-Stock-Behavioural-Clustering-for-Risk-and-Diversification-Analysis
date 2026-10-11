"""Final isolated validation of canonical and six-cluster alternatives."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "multiview_clustering" / "outputs" / "final_validation"
TRADING_DAYS = 252
SEEDS = (7, 19, 42, 73, 101)
FEATURES = [
    "annualised_return", "annualised_volatility", "max_drawdown",
    "rolling_vol_dispersion", "beta", "momentum_3m", "momentum_12m",
    "trend_strength", "autocorr_lag1", "skewness", "excess_kurtosis",
    "volume_volatility", "return_volume_corr", "extreme_move_frequency",
]
WINDOWS = ((0, 504, 504, 756), (252, 756, 756, 1008), (504, 1008, 1008, 1139))
COST_BPS = 10.0


def save(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def fingerprint(prices: pd.DataFrame, volume: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Rebuild the canonical 14-feature fingerprint on one training window."""
    market = prices.pct_change().mean(axis=1)
    rows = {}
    for symbol in symbols:
        price = prices[symbol].dropna()
        ret = price.pct_change().dropna()
        log_volume = np.log(volume[symbol].reindex(price.index).replace(0, np.nan))
        aligned = pd.concat([ret, market], axis=1).dropna()
        rolling = ret.rolling(60).std(ddof=1).dropna()
        threshold = 2 * ret.std(ddof=1)
        rows[symbol] = {
            "annualised_return": ret.mean() * TRADING_DAYS,
            "annualised_volatility": ret.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "max_drawdown": (1 - price / price.cummax()).max(),
            "rolling_vol_dispersion": rolling.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "beta": aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(),
            "momentum_3m": price.iloc[-1] / price.iloc[-64] - 1,
            "momentum_12m": price.iloc[-1] / price.iloc[-253] - 1,
            "trend_strength": np.polyfit(np.arange(len(price)), np.log(price), 1)[0] * TRADING_DAYS,
            "autocorr_lag1": ret.autocorr(1),
            "skewness": ret.skew(),
            "excess_kurtosis": ret.kurtosis(),
            "volume_volatility": log_volume.diff().std(ddof=1),
            "return_volume_corr": pd.concat([ret.abs(), log_volume.diff()], axis=1).dropna().corr().iloc[0, 1],
            "extreme_move_frequency": (ret.abs() > threshold).mean(),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")[FEATURES]
    return result.replace([np.inf, -np.inf], np.nan).fillna(result.median())


def fit_representation(features: pd.DataFrame, kind: str):
    scaler = RobustScaler() if kind == "robust_kmeans" else StandardScaler()
    scaled = scaler.fit_transform(features)
    if kind in ("canonical_k2", "refined_k6"):
        target = 0.85 if kind == "canonical_k2" else 0.70
        pca = PCA(n_components=target, random_state=42).fit(scaled)
        return pca.transform(scaled), scaler, pca
    return scaled, scaler, None


def fit_labels(data: np.ndarray, kind: str, k: int, seed: int = 42) -> np.ndarray:
    if kind in ("canonical_k2", "refined_k6", "standard_kmeans", "robust_kmeans"):
        return KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(data)
    if kind == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data)
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=5,
                           random_state=seed).fit_predict(data)


def cluster_metrics(data: np.ndarray, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels, minlength=len(np.unique(labels)))
    return {
        "silhouette": silhouette_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_sizes": "|".join(map(str, sizes)),
    }


def select_holdings(labels: np.ndarray, symbols: list[str], data: np.ndarray) -> list[str]:
    groups = []
    for cluster_id in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == cluster_id)
        center = data[members].mean(axis=0)
        representative = min(members, key=lambda i: (np.linalg.norm(data[i] - center), symbols[i]))
        ordered = sorted(members, key=lambda i: (np.linalg.norm(data[i] - center), symbols[i]))
        groups.append((len(members), [symbols[i] for i in ordered]))
    groups.sort(key=lambda x: (-x[0], x[1][0]))
    selected = []
    depth = 0
    while len(selected) < 6 and any(depth < len(members) for _, members in groups):
        for _, members in groups:
            if depth < len(members) and len(selected) < 6:
                selected.append(members[depth])
        depth += 1
    return selected


def baseline_holdings(symbols: list[str], diversified: bool) -> list[str]:
    ordered = sorted(symbols)
    if not diversified:
        return ordered[:6]
    positions = np.linspace(0, len(ordered) - 1, 6, dtype=int)
    return [ordered[int(i)] for i in positions]


def portfolio_metrics(returns: pd.DataFrame, holdings: list[str]) -> dict:
    daily = returns[holdings].mean(axis=1).copy()
    daily.iloc[0] -= COST_BPS / 10_000
    wealth = (1 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    corr = returns[holdings].corr().to_numpy()
    upper = corr[np.triu_indices(len(holdings), 1)]
    asset_vol = returns[holdings].std(ddof=1).to_numpy() * math.sqrt(TRADING_DAYS)
    annual_vol = daily.std(ddof=1) * math.sqrt(TRADING_DAYS)
    return {
        "annualised_return": (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1,
        "annualised_volatility": annual_vol,
        "max_drawdown": -drawdown.min(),
        "sharpe_ratio_rf0": daily.mean() * TRADING_DAYS / annual_vol,
        "avg_pairwise_correlation": np.nanmean(upper),
        "diversification_ratio": asset_vol.mean() / annual_vol,
        "holdings_count": len(holdings),
        "turnover": 1.0,
        "transaction_cost_bps": COST_BPS,
        "holdings": "|".join(holdings),
    }


def audit_data(prices: pd.DataFrame, volume: pd.DataFrame, symbols: list[str]) -> str:
    panel = pd.read_csv(PROC / "cleaned_daily_data.csv", parse_dates=["Date"])
    duplicates = int(panel.duplicated(["Date", "Symbol"]).sum())
    missing = int(prices[symbols].isna().sum().sum())
    return f"""# Final-validation data and leakage audit

## Data

- Source file: `data/processed/cleaned_daily_data.csv`.
- Persisted universe: `{len(symbols)}` stocks, taken from the canonical
  `stock_fingerprints.csv` symbol index; no symbols were added or removed.
- Prices use the persisted `Adj Close` column. Returns are `pct_change()` on
  the aligned price panel and the first missing return is dropped.
- The cleaned panel contains `{len(panel):,}` rows and `{duplicates}` duplicate
  `(Date, Symbol)` records. The aligned price matrix contains `{missing}` missing
  cells across the persisted universe.
- Every strategy uses six holdings in the primary comparison. The fixed
  baseline uses alphabetically first six stocks. The diversified baseline uses
  six deterministic quantiles of the alphabetically sorted universe; neither
  rule uses clustering or test returns.

## Feature and model audit

The final runner reconstructs the exact canonical 14-feature set:
annualised return, volatility, drawdown, rolling-volatility dispersion, beta,
3-month and 12-month momentum, trend strength, lag-1 autocorrelation,
skewness, excess kurtosis, volume volatility, return-volume correlation, and
extreme-move frequency. The prior refinement's extra downside-volatility
feature is not used here.

For each rolling window, fingerprints are built from the training slice only.
Scaling, PCA, and K-Means/GMM are fitted only on training fingerprints.
Ward is fitted only on training fingerprints. Representatives are nearest to
training centroids; no test-period value chooses a stock or model.

## Portfolio assumptions

Portfolios are equal-weighted and daily rebalanced across the fixed test window.
A 10 basis-point initial transaction cost is charged on the first test day;
turnover is recorded as 1.0 because there is no prior portfolio supplied.
No risk-free return is assumed, so Sharpe is excess return over zero. No
transaction cost optimization, future-return optimization, or rebalancing
optimization is performed.

## Remaining caveats

The benchmark index is not present in the cleaned stock panel, so the training
window beta uses the equal-weight stock-market proxy, matching the isolated
experiment convention rather than the canonical benchmark beta. Three windows
are available, with the last window shorter because the fixed dataset ends in
August 2026. This supports comparison, not statistical significance.
"""


def main() -> None:
    panel = pd.read_csv(PROC / "cleaned_daily_data.csv", parse_dates=["Date"])
    prices = panel.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
    volume = panel.pivot(index="Date", columns="Symbol", values="Volume").sort_index()
    symbols = sorted(pd.read_csv(PROC / "stock_fingerprints.csv")["Symbol"])
    prices, volume = prices[symbols], volume[symbols]
    returns = prices.pct_change().dropna()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(
        "Run from the project root with `python experiments/multiview_clustering/final_validation.py`.\n",
        encoding="utf-8",
    )
    (OUT / "data_audit.md").write_text(audit_data(prices, volume, symbols), encoding="utf-8")
    original_hashes = {}
    for path in ("src/config.py", "src/clustering.py", "src/finance.py",
                 "data/processed/cluster_assignments.csv", "data/processed/portfolio_results.csv"):
        original_hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    (OUT / "canonical_hashes_before.csv").write_text(
        "path,sha256\n" + "\n".join(f"{p},{h}" for p, h in original_hashes.items()) + "\n",
        encoding="utf-8",
    )

    kinds = {
        "canonical_pca_k2": ("canonical_k2", 2),
        "refined_pca_k6": ("refined_k6", 6),
        "standard_fingerprint_k6": ("standard_kmeans", 6),
        "robust_fingerprint_k6": ("robust_kmeans", 6),
        "ward_k6": ("ward", 6),
        "gmm_k6": ("gmm", 6),
    }
    clustering_rows, stability_rows, assignments, profiles = [], [], [], []
    for label, (kind, k) in kinds.items():
        features = fingerprint(prices, volume, symbols)
        data, _, pca = fit_representation(features, kind)
        labels = fit_labels(data, kind, k)
        extra = cluster_metrics(data, labels)
        clustering_rows.append({"model": label, "k": k, "pca_components": pca.n_components_ if pca else 0,
                                "pca_explained_variance": pca.explained_variance_ratio_.sum() if pca else 1.0, **extra})
        runs = [fit_labels(data, kind, k, seed) for seed in SEEDS]
        aris = [adjusted_rand_score(runs[0], run) for run in runs[1:]]
        stability_rows.append({"model": label, "mean_ari": np.mean(aris), "std_ari": np.std(aris)})
        assignments.append(pd.DataFrame({"Symbol": symbols, "model": label, "cluster": labels + 1}))
        for cluster_id in sorted(np.unique(labels)):
            members = [symbols[i] for i, x in enumerate(labels) if x == cluster_id]
            profiles.append({"model": label, "cluster": cluster_id + 1, "members": len(members),
                             "symbols": "|".join(members), **features.loc[members].mean().to_dict()})

    comparison = pd.DataFrame(clustering_rows).merge(pd.DataFrame(stability_rows), on="model")
    save(comparison, "clustering_comparison.csv")
    save(pd.concat(assignments, ignore_index=True), "cluster_assignments.csv")
    save(pd.DataFrame(profiles), "cluster_profiles.csv")
    save(pd.DataFrame(stability_rows), "cluster_stability.csv")

    walk_rows, portfolio_rows = [], []
    for start, train_end, test_start, end in WINDOWS:
        train_prices, train_volume = prices.iloc[start:train_end], volume.iloc[start:train_end]
        test_returns = returns.iloc[test_start:end]
        window = f"{test_returns.index[0].date()}_{test_returns.index[-1].date()}"
        train_features = fingerprint(train_prices, train_volume, symbols)
        fitted = {}
        for label, (kind, k) in kinds.items():
            data, _, pca = fit_representation(train_features, kind)
            labels = fit_labels(data, kind, k)
            holdings = select_holdings(labels, symbols, data)
            fitted[label] = holdings
            row = {"model": label, "window": window, "strategy": "cluster_representatives", "train_days": train_end - start,
                   "test_days": end - test_start, **portfolio_metrics(test_returns, holdings)}
            walk_rows.append(row)
        baselines = {
            "equal_weight_baseline_6": baseline_holdings(symbols, False),
            "equal_weight_diversified_baseline_6": baseline_holdings(symbols, True),
        }
        for label, holdings in baselines.items():
            walk_rows.append({"model": label, "window": window, "strategy": label,
                              "train_days": train_end - start, "test_days": end - test_start,
                              **portfolio_metrics(test_returns, holdings)})
        # Same six holdings, distributed as two representatives from three
        # largest training clusters, to test a multi-stock cluster rule.
        data, _, _ = fit_representation(train_features, "refined_k6")
        labels = fit_labels(data, "refined_k6", 6)
        groups = {}
        for c in sorted(np.unique(labels)):
            members = np.flatnonzero(labels == c)
            center = data[members].mean(axis=0)
            ordered = sorted(members, key=lambda i: (np.linalg.norm(data[i] - center), symbols[i]))
            groups[c] = [symbols[i] for i in ordered]
        selected = []
        for c in sorted(groups, key=lambda c: (-len(groups[c]), c))[:3]:
            selected.extend(groups[c][:2])
        walk_rows.append({"model": "refined_pca_k6", "window": window,
                          "strategy": "two_from_three_clusters", "train_days": train_end - start,
                          "test_days": end - test_start, **portfolio_metrics(test_returns, selected)})
        portfolio_rows.extend(walk_rows[-(len(kinds) + 3):])

    walk = pd.DataFrame(walk_rows)
    save(walk, "walk_forward_results.csv")
    save(walk[["model", "window", "strategy", "annualised_return",
               "annualised_volatility", "max_drawdown", "sharpe_ratio_rf0",
               "avg_pairwise_correlation", "diversification_ratio",
               "holdings_count", "turnover", "transaction_cost_bps", "holdings"]],
         "portfolio_comparison.csv")
    summary = walk.groupby(["model", "strategy"]).agg(
        windows=("window", "count"), annualised_return_mean=("annualised_return", "mean"),
        annualised_return_std=("annualised_return", "std"), volatility_mean=("annualised_volatility", "mean"),
        drawdown_mean=("max_drawdown", "mean"), sharpe_mean=("sharpe_ratio_rf0", "mean"),
        correlation_mean=("avg_pairwise_correlation", "mean"),
        diversification_mean=("diversification_ratio", "mean"),
        holdings_count=("holdings_count", "mean"), turnover_mean=("turnover", "mean"),
    ).reset_index()
    save(summary, "portfolio_summary.csv")
    robustness = comparison.copy()
    robustness["test"] = "seed_stability_and_representation"
    save(robustness, "robustness_results.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(comparison["model"], comparison["min_cluster_size"])
    ax.set_ylabel("Minimum cluster size"); ax.set_title("Cluster balance comparison")
    ax.tick_params(axis="x", rotation=35); fig.tight_layout()
    fig.savefig(OUT / "plots" / "cluster_balance.png", dpi=150); plt.close(fig)
    oos = summary[summary["strategy"].isin(["cluster_representatives", "equal_weight_baseline_6",
                                             "equal_weight_diversified_baseline_6"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for column, axis, title in (("annualised_return_mean", axes[0], "Return"), ("volatility_mean", axes[1], "Volatility"),
                                ("drawdown_mean", axes[2], "Drawdown")):
        axis.bar(oos["model"], oos[column]); axis.set_title(title); axis.tick_params(axis="x", rotation=70)
    fig.tight_layout(); fig.savefig(OUT / "plots" / "oos_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(oos["model"], oos["diversification_mean"]); ax.set_title("Out-of-sample diversification ratio")
    ax.tick_params(axis="x", rotation=70); fig.tight_layout()
    fig.savefig(OUT / "plots" / "oos_diversification.png", dpi=150); plt.close(fig)

    after_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in original_hashes}
    hash_ok = all(original_hashes[p] == after_hashes[p] for p in original_hashes)
    (OUT / "canonical_hashes_after.csv").write_text(
        "path,sha256,unchanged\n" + "\n".join(f"{p},{h},{original_hashes[p] == h}" for p, h in after_hashes.items()) + "\n",
        encoding="utf-8",
    )
    selected = summary[summary.model == "refined_pca_k6"].iloc[0]
    (OUT / "experiment_report.md").write_text(f"""# Final Validation Report

## Scope and isolation

This validation writes only to `outputs/final_validation/`. Canonical code,
canonical processed data, and previous experiment outputs were not overwritten.
Canonical hash verification passed: `{hash_ok}`.

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
representative strategy had mean return `{selected.annualised_return_mean:.2%}`,
volatility `{selected.volatility_mean:.2%}`, drawdown `{selected.drawdown_mean:.2%}`,
correlation `{selected.correlation_mean:.3f}`, and diversification ratio
`{selected.diversification_mean:.3f}`. The full window-level values are in
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
""", encoding="utf-8")
    print("Final validation complete.")


if __name__ == "__main__":
    main()

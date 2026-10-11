"""Stage 6: historical cluster risk and Portfolio A versus Portfolio B.

Selections use only Stage 4 geometry.  Returns are calculated from the cleaned
adjusted-close panel and are therefore a historical evaluation, not a forecast
or an investment recommendation.

Run with ``python -m src.finance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src import config


@dataclass(frozen=True)
class Selection:
    """The deterministic holdings selected for one cluster protocol."""

    space: str
    k: int
    cluster: int
    holdings: tuple[str, ...]
    eligible: bool
    reason: str


def _validate_inputs(
    assignments: pd.DataFrame, scaled: pd.DataFrame, prices: pd.DataFrame
) -> None:
    """Validate aligned symbols and finite price/geometry inputs."""
    if assignments.empty or scaled.empty or prices.empty:
        raise ValueError("Stage 6 inputs must all be non-empty")
    if set(assignments.index) != set(scaled.index) or not set(assignments.index).issubset(prices.columns):
        raise ValueError("Assignments, scaled fingerprints, and prices have different symbols")
    if not np.isfinite(scaled.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Stage 6 inputs contain non-finite geometry or non-positive prices")


def _adjusted_close_panel() -> pd.DataFrame:
    """Load the cached adjusted-close panel without requiring network packages."""
    if not config.CLEANED_DAILY_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned daily data: {config.CLEANED_DAILY_PATH}")
    panel = pd.read_csv(config.CLEANED_DAILY_PATH, parse_dates=["Date"])
    required = {"Date", "Symbol", "Adj Close"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Cleaned data must contain columns {sorted(required)}")
    prices = panel.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
    if prices.empty:
        raise ValueError("Cleaned adjusted-close panel is empty")
    return prices


def _portfolio_metrics(returns: pd.DataFrame) -> dict[str, float]:
    """Calculate equal-weight, daily-rebalanced annualized portfolio metrics."""
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("A portfolio needs at least one constituent")
    daily = returns.mean(axis=1).dropna()
    if daily.empty:
        raise ValueError("A portfolio has no complete return observations")
    growth = (1.0 + daily).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    corr = returns.corr()
    upper = corr.to_numpy()[np.triu_indices(corr.shape[0], 1)]
    return {
        "annualised_return": float((1.0 + daily).prod() ** (config.TRADING_DAYS_PER_YEAR / len(daily)) - 1.0),
        "annualised_volatility": float(daily.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(-drawdown.min()),
        "avg_pairwise_constituent_correlation": float(np.nanmean(upper)) if len(upper) else np.nan,
    }


def _select_members(
    space: str, k: int, labels: pd.Series, scaled: pd.DataFrame
) -> tuple[list[Selection], dict[int, list[str]]]:
    """Rank each cluster by standardized centroid distance with alphabetic ties."""
    n_holdings = min(config.PORTFOLIO_MAX_HOLDINGS, k)
    groups: dict[int, list[str]] = {}
    selections: list[Selection] = []
    for cluster in sorted(int(x) for x in labels.dropna().unique()):
        members = sorted(labels.index[labels == cluster].tolist())
        centroid = scaled.loc[members].mean(axis=0).to_numpy()
        distances = np.linalg.norm(scaled.loc[members].to_numpy() - centroid, axis=1)
        ranked = [s for _, s in sorted(zip(distances, members), key=lambda pair: (pair[0], pair[1]))]
        groups[cluster] = ranked
        eligible = len(ranked) >= n_holdings
        selections.append(Selection(
            space, k, cluster, tuple(ranked[:n_holdings]) if eligible else tuple(),
            eligible, "" if eligible else f"cluster has {len(ranked)} members; needs {n_holdings}",
        ))
    return selections, groups


def _labels_for(assignments: pd.DataFrame, space: str, k: int) -> pd.Series:
    """Read and validate one raw or retained-PCA Stage 4 label column."""
    column = f"{space}_cluster_k{k}"
    if column not in assignments:
        raise ValueError(f"Missing Stage 4 label column: {column}")
    labels = pd.to_numeric(assignments[column], errors="coerce")
    if labels.isna().any() or labels.nunique() != k:
        raise ValueError(f"{column} does not contain exactly {k} valid clusters")
    return labels.astype(int)


def _metric_row(kind: str, space: str, k: int, metrics: dict[str, float], *,
                cluster: int | str = "", holdings: Iterable[str] = (), draw: int | str = "") -> dict:
    """Build a stable, wide output row for portfolio results."""
    return {
        "space": space, "k": k, "portfolio": kind, "cluster": cluster,
        "draw": draw, "holdings": "|".join(holdings),
        **metrics,
    }


def run_finance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both portfolio protocols for every configured k and persist outputs."""
    config.ensure_directories()
    assignments = pd.read_csv(config.CLUSTER_ASSIGNMENTS_PATH, index_col="Symbol").sort_index()
    scaled = pd.read_csv(config.CLUSTER_SCALED_PATH, index_col="Symbol").sort_index()
    prices = _adjusted_close_panel()
    _validate_inputs(assignments, scaled, prices)
    returns = prices.pct_change().dropna(how="all")
    returns = returns.loc[:, sorted(set(assignments.index))]
    canonical_space = str(assignments["canonical_space"].iloc[0])
    canonical_k = int(assignments["canonical_k"].iloc[0])
    if canonical_space not in ("raw", "pca") or canonical_k not in config.K_RANGE:
        raise ValueError("Stage 4 canonical metadata is invalid")

    summary_rows: list[dict] = []
    result_rows: list[dict] = []
    selection_rows: list[dict] = []
    correlation_rows: list[dict] = []
    rng = np.random.default_rng(config.RANDOM_STATE)

    for space in ("raw", "pca"):
        for k in config.K_RANGE:
            labels = _labels_for(assignments, space, int(k))
            selections, groups = _select_members(space, int(k), labels, scaled)
            is_canonical = space == canonical_space and int(k) == canonical_k
            cluster_portfolios: list[dict[str, float]] = []
            # Cluster statistics use every member, independently of whether
            # that cluster can fill Portfolio A.
            for cluster, members in groups.items():
                cluster_metric = _portfolio_metrics(returns.loc[:, members])
                summary_rows.append({
                    "space": space, "k": k, "cluster": cluster,
                    "canonical": is_canonical, "members": len(members),
                    "annualised_return": cluster_metric["annualised_return"],
                    "annualised_volatility": cluster_metric["annualised_volatility"],
                    "max_drawdown": cluster_metric["max_drawdown"],
                    "within_cluster_avg_pairwise_correlation":
                        cluster_metric["avg_pairwise_constituent_correlation"],
                })
            for selection in selections:
                selection_rows.append({
                    "space": space, "k": k, "cluster": selection.cluster,
                    "eligible_for_a": selection.eligible, "reason": selection.reason,
                    "portfolio_a_holdings": "|".join(selection.holdings),
                })
                if not selection.eligible:
                    continue
                metric = _portfolio_metrics(returns.loc[:, list(selection.holdings)])
                cluster_portfolios.append(metric)
                row = _metric_row("portfolio_a_cluster", space, int(k), metric,
                                  cluster=selection.cluster, holdings=selection.holdings)
                row["canonical"] = is_canonical
                result_rows.append(row)

            # B uses all clusters up to five, or the five largest clusters.
            ordered_clusters = sorted(groups, key=lambda c: (-len(groups[c]), c))[: min(5, k)]
            medoids = tuple(groups[c][0] for c in ordered_clusters)
            b_metrics = _portfolio_metrics(returns.loc[:, list(medoids)])
            b_row = _metric_row("portfolio_b", space, int(k), b_metrics, holdings=medoids)
            b_row["canonical"] = is_canonical
            result_rows.append(b_row)

            if cluster_portfolios:
                for name, function in (("median", np.median), ("min", np.min), ("max", np.max)):
                    aggregate = {
                        field: float(function([m[field] for m in cluster_portfolios]))
                        for field in b_metrics
                    }
                    row = _metric_row(f"portfolio_a_{name}", space, int(k), aggregate)
                    row["canonical"] = is_canonical
                    result_rows.append(row)

            within_values: list[float] = []
            for cluster, members in groups.items():
                if len(members) > 1:
                    within_values.extend(returns.loc[:, members].corr().to_numpy()[np.triu_indices(len(members), 1)])
            cluster_returns = pd.DataFrame({
                c: returns.loc[:, members].mean(axis=1) for c, members in groups.items()
            })
            between = cluster_returns.corr().to_numpy()[np.triu_indices(len(groups), 1)]
            correlation_rows.append({
                "space": space, "k": k, "canonical": is_canonical,
                "within_cluster_avg_pairwise_correlation": float(np.nanmean(within_values)) if within_values else np.nan,
                "between_cluster_avg_pairwise_correlation": float(np.nanmean(between)) if len(between) else np.nan,
            })

            # One RNG is consumed in this explicit order: all cross draws, then
            # all within draws for each (space, k).
            cross_metrics: list[dict[str, float]] = []
            within_metrics: list[dict[str, float]] = []
            eligible_groups = [c for c in sorted(groups) if len(groups[c]) >= min(5, k)]
            for draw in range(config.PORTFOLIO_RANDOM_DRAWS):
                chosen = [str(rng.choice(groups[c])) for c in ordered_clusters]
                cross_metrics.append(_portfolio_metrics(returns.loc[:, chosen]))
                result_rows.append(_metric_row("random_cross_cluster", space, int(k), cross_metrics[-1],
                                               holdings=chosen, draw=draw))
            for draw in range(config.PORTFOLIO_RANDOM_DRAWS):
                if not eligible_groups:
                    break
                cluster = int(rng.choice(eligible_groups))
                chosen = list(rng.choice(groups[cluster], size=min(5, k), replace=False))
                within_metrics.append(_portfolio_metrics(returns.loc[:, chosen]))
                result_rows.append(_metric_row("random_within_cluster", space, int(k), within_metrics[-1],
                                               holdings=chosen, draw=draw))
            if within_metrics:
                cross_returns = [m["annualised_return"] for m in cross_metrics]
                within_returns = [m["annualised_return"] for m in within_metrics]
                u_test = mannwhitneyu(cross_returns, within_returns, alternative="two-sided")
                u = float(u_test.statistic)
                effect = 1.0 - 2.0 * u / (len(cross_returns) * len(within_returns))
                result_rows.append({
                    **_metric_row("random_comparison", space, int(k), {}),
                    "mann_whitney_u": u, "p_value_indicative": float(u_test.pvalue),
                    "rank_biserial_effect": effect, "canonical": is_canonical,
                })

    summary = pd.DataFrame(summary_rows)
    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    correlations = pd.DataFrame(correlation_rows)
    summary.to_csv(config.CLUSTER_SUMMARY_PATH, index=False)
    results.to_csv(config.PORTFOLIO_RESULTS_PATH, index=False)
    selections.to_csv(config.FINANCE_SELECTION_PATH, index=False)
    correlations.to_csv(config.FINANCE_CORRELATION_PATH, index=False)
    return summary, results, correlations


if __name__ == "__main__":
    config.configure_console()
    summary, results, _ = run_finance()
    headline = results[(results["canonical"] == True) & (results["portfolio"].isin(
        ["portfolio_a_median", "portfolio_b"]))]
    print(f"Stage 6: {len(summary)} canonical cluster rows; {len(results):,} result rows.")
    print(f"Canonical: {headline[['portfolio', 'annualised_return', 'annualised_volatility', 'max_drawdown']].to_string(index=False)}")
    print(f"Wrote {config.CLUSTER_SUMMARY_PATH.name}, {config.PORTFOLIO_RESULTS_PATH.name}, "
          f"{config.FINANCE_SELECTION_PATH.name}, and {config.FINANCE_CORRELATION_PATH.name}")

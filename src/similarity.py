"""Find stocks with similar historical behavioural fingerprints.

Similarity is calculated in the standardized fingerprint space produced by
Stage 4.  It describes historical behaviour only; it is not a forecast or a
recommendation.

Run with ``python -m src.similarity``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def load_scaled_fingerprints(path: Path = config.CLUSTER_SCALED_PATH) -> pd.DataFrame:
    """Load the standardized fingerprints used for clustering and similarity."""
    if not path.exists():
        raise FileNotFoundError(f"Missing standardized fingerprints: {path}")
    fingerprints = pd.read_csv(path, index_col="Symbol").sort_index()
    if fingerprints.empty or fingerprints.index.has_duplicates:
        raise ValueError("Standardized fingerprint table is empty or has duplicate symbols")
    if not np.isfinite(fingerprints.to_numpy(dtype=float)).all():
        raise ValueError("Standardized fingerprints contain missing or infinite values")
    if (np.linalg.norm(fingerprints.to_numpy(dtype=float), axis=1) == 0).any():
        raise ValueError("Standardized fingerprints contain a zero-length fingerprint")
    return fingerprints


def cosine_similarity(query: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    """Return cosine similarity between one fingerprint and candidate rows."""
    query_values = query.to_numpy(dtype=float)
    candidate_values = candidates.to_numpy(dtype=float)
    query_norm = float(np.linalg.norm(query_values))
    candidate_norms = np.linalg.norm(candidate_values, axis=1)
    if query_norm == 0 or (candidate_norms == 0).any():
        raise ValueError("Cosine similarity requires non-zero fingerprints")
    scores = candidate_values @ query_values / (candidate_norms * query_norm)
    return pd.Series(scores, index=candidates.index, name="similarity_score")


def _load_assignments() -> pd.DataFrame:
    """Load cluster labels so similarity results retain interpretation metadata."""
    if not config.CLUSTER_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {config.CLUSTER_ASSIGNMENTS_PATH}")
    assignments = pd.read_csv(config.CLUSTER_ASSIGNMENTS_PATH, index_col="Symbol")
    if "cluster" not in assignments.columns:
        raise ValueError("Cluster assignments must contain a 'cluster' column")
    return assignments


def find_similar_stocks(
    ticker: str,
    top_n: int = config.SIMILARITY_TOP_N,
    fingerprints: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the closest historical fingerprints for one ticker."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty stock symbol")
    table = load_scaled_fingerprints() if fingerprints is None else fingerprints.copy()
    table.index = table.index.astype(str)
    table = table.sort_index()
    query = ticker.strip().upper()
    if query not in table.index:
        raise ValueError(f"Unknown ticker {ticker!r}; available tickers are in the fingerprint table")
    if top_n > len(table) - 1:
        raise ValueError(f"top_n={top_n} exceeds the {len(table) - 1} available other stocks")
    scores = cosine_similarity(table.loc[query], table.drop(index=query))
    result = scores.rename_axis("similar_stock").reset_index()
    result["distance"] = 1.0 - result["similarity_score"]
    result = result.sort_values(
        ["similarity_score", "similar_stock"], ascending=[False, True], kind="mergesort"
    ).head(top_n).reset_index(drop=True)
    result.insert(0, "query_stock", query)
    result.insert(3, "rank", np.arange(1, len(result) + 1))

    metadata = _load_assignments() if assignments is None else assignments.copy()
    metadata.index = metadata.index.astype(str)
    if query not in metadata.index or not result["similar_stock"].isin(metadata.index).all():
        raise ValueError("Cluster assignments do not cover the query and all similar stocks")
    result["query_cluster"] = metadata.loc[query, "cluster"]
    result["similar_cluster"] = result["similar_stock"].map(metadata["cluster"])
    result["same_cluster"] = result["query_cluster"] == result["similar_cluster"]
    result["query_sector"] = config.SECTORS.get(query, "Unknown")
    result["similar_sector"] = result["similar_stock"].map(
        lambda symbol: config.SECTORS.get(symbol, "Unknown")
    )
    return result[
        [
            "query_stock", "similar_stock", "similarity_score", "distance",
            "rank", "query_cluster", "similar_cluster", "same_cluster",
            "query_sector", "similar_sector",
        ]
    ]


def run_similarity(
    ticker: str = config.SIMILARITY_QUERY_TICKER,
    top_n: int = config.SIMILARITY_TOP_N,
) -> pd.DataFrame:
    """Run the configured deterministic query and write the Stage 5 output."""
    result = find_similar_stocks(ticker, top_n)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.SIMILARITY_RESULTS_PATH, index=False)
    return result


if __name__ == "__main__":
    config.configure_console()
    results = run_similarity()
    query = results.iloc[0]["query_stock"]
    print(f"Query: {query}; returned {len(results)} historical behavioural neighbours.")
    print(results[["rank", "similar_stock", "similarity_score", "similar_cluster"]].to_string(index=False))
    print(f"Wrote {config.SIMILARITY_RESULTS_PATH.name}")

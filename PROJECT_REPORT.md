# AI-Based Stock Behavioural Clustering

## 1. Executive summary

This project converts historical NSE stock-price behaviour into numerical
**behavioural fingerprints**, groups stocks using unsupervised machine learning,
and evaluates whether selecting stocks from different behavioural groups has
historically provided better diversification than selecting stocks from one
group.

The project does **not** predict prices, generate trading signals, execute
trades, or provide investment advice.

### Final headline result

The canonical model used PCA-reduced standardized fingerprints with K-Means and
`k = 2`.

The cross-cluster portfolio performed better than the median same-cluster
portfolio over the historical study window:

| Portfolio | Annualised return | Annualised volatility | Maximum drawdown | Average constituent correlation |
|---|---:|---:|---:|---:|
| Same-cluster Portfolio A (median) | 13.72% | 27.03% | 37.20% | 0.526 |
| Cross-cluster Portfolio B | **17.49%** | **25.33%** | **27.93%** | **0.390** |

This supports the project hypothesis for this historical sample: behavioural
diversification produced a better overall balance of return, drawdown, and
correlation. It does not prove that the result will repeat in the future.

---

## 2. Project scope and research questions

The authoritative project scope is documented in
[scope.md](./scope.md). Current decisions and resolved methodology choices are
recorded in [CONTEXT.md](./CONTEXT.md).

The project answers four research questions:

| Question | Evidence |
|---|---|
| Can ML discover meaningful behavioural groups? | K-Means, PCA, and internal clustering metrics |
| Which stocks share similar historical behaviour? | Cosine-similarity search |
| Do groups differ in historical return and risk? | Cluster-level financial statistics |
| Does cross-cluster selection diversify better? | Portfolio A versus Portfolio B experiment |

The core research idea is that sector labels describe what a company does, while
behavioural clustering describes how its stock has historically moved.

---

## 3. Data and study design

### Data source

- Source: Yahoo Finance through `yfinance`
- Universe: 75 NSE-listed stocks
- Benchmark: NIFTY 50 (`^NSEI`)
- Historical window: January 2022 through August 2026
- Common aligned observations: 1,140 trading days per stock
- Main price field: Adjusted Close

Raw cached files are stored in
[data/raw/](./data/raw/). The raw cache includes the Yahoo download manifest at
[data/raw/_manifest.json](./data/raw/_manifest.json).

### Data-quality decisions

The pipeline explicitly checked:

- Duplicate dates
- Missing adjusted prices
- Non-positive prices
- Insufficient history
- Extreme daily price movements
- Zero-volume observations
- Calendar mismatches between stocks and the benchmark

Vedanta was excluded because Yahoo Finance did not adjust a large corporate
action correctly. Several stale or renamed symbols were also excluded during
universe preparation. These decisions are described in
[docs/DATA_PROCESSING.md](./docs/DATA_PROCESSING.md) and
[docs/NOTES.md](./docs/NOTES.md).

The current quality output is:

- [data_quality_report.csv](./data/processed/data_quality_report.csv)
- [cleaning_drop_log.csv](./data/processed/cleaning_drop_log.csv)

### Cleaned data output

The aligned daily panel is stored in:

- [cleaned_daily_data.csv](./data/processed/cleaned_daily_data.csv)

It contains 85,500 rows:

```text
75 stocks × 1,140 common trading days
```

The data is stored in long format:

```text
Date | Symbol | Open | High | Low | Close | Adj Close | Volume
```

---

## 4. Behavioural fingerprint engineering

The feature-engineering implementation is in
[src/features.py](./src/features.py).

The final fingerprint contains 14 features. Each stock becomes one row, so the
machine-learning input has shape:

```text
75 stocks × 14 behavioural features
```

### Final features

| Feature | What it measures |
|---|---|
| `annualised_return` | Typical historical return level |
| `annualised_volatility` | Overall return variability |
| `max_drawdown` | Worst historical peak-to-trough decline |
| `rolling_vol_dispersion` | Stability of the stock's risk level |
| `beta` | Sensitivity to NIFTY movements |
| `momentum_3m` | Recent three-month direction |
| `momentum_12m` | Longer-term twelve-month direction |
| `trend_strength` | Consistency of the long-term price trend |
| `autocorr_lag1` | Persistence or reversal of consecutive returns |
| `skewness` | Asymmetry of the return distribution |
| `excess_kurtosis` | Fat-tail and extreme-movement tendency |
| `volume_volatility` | Instability of participation volume |
| `return_volume_corr` | Relationship between large moves and volume changes |
| `extreme_move_frequency` | Frequency of stock-relative extreme moves |

Features removed for redundancy or conceptual reasons are documented in
[CONTEXT.md](./CONTEXT.md) and the full diagnostic evidence is stored in
[feature_quality_report.csv](./data/processed/feature_quality_report.csv).

The final fingerprint table is:

- [stock_fingerprints.csv](./data/processed/stock_fingerprints.csv)

---

## 5. Stage 4 — Clustering

The clustering implementation is in
[src/clustering.py](./src/clustering.py).

### Method

1. Load the 14-feature fingerprint table.
2. Standardize all features with `StandardScaler`.
3. Fit PCA:
   - Retained-variance PCA for clustering comparison
   - Separate two-dimensional PCA for visualization
4. Run K-Means for every `k` from 2 through 8.
5. Evaluate each solution using:
   - Silhouette score
   - Davies-Bouldin index
   - Calinski-Harabasz score
6. Compare raw-feature and PCA-space memberships using Adjusted Rand Index.

### Canonical model

The selected canonical model is:

```text
Space: PCA-reduced standardized fingerprints
k:     2
```

For `k = 2`, PCA was better than the raw standardized space:

| Space | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| Raw standardized features | 0.4299 | 1.1343 | 20.26 |
| PCA-reduced features | **0.4539** | **1.0293** | **23.81** |

PCA retained 7 components and approximately 88.05% cumulative explained
variance.

### Clustering outputs

- [cluster_metrics.csv](./data/processed/cluster_metrics.csv)
  - All raw/PCA metric results for `k = 2..8`
- [cluster_assignments.csv](./data/processed/cluster_assignments.csv)
  - Every stock's raw and PCA labels plus the canonical label
- [cluster_pca_coordinates.csv](./data/processed/cluster_pca_coordinates.csv)
  - Two-dimensional coordinates for plotting
- [cluster_pca_retained.csv](./data/processed/cluster_pca_retained.csv)
  - PCA coordinates used in the retained-variance comparison
- [cluster_pca_summary.csv](./data/processed/cluster_pca_summary.csv)
  - Explained variance by component
- [cluster_scaled_fingerprints.csv](./data/processed/cluster_scaled_fingerprints.csv)
  - Standardized feature matrix used by clustering and similarity search

### Canonical cluster sizes

| Cluster | Number of stocks |
|---:|---:|
| Cluster 1 | 69 |
| Cluster 2 | 6 |

This imbalance is an observed result, not a manually imposed balance. It should
be treated as an interpretation point and limitation.

---

## 6. Cluster memberships

The authoritative complete membership table is
[cluster_assignments.csv](./data/processed/cluster_assignments.csv).

### Cluster 1 — 69 stocks

```text
ABB, APOLLOHOSP, ASHOKLEY, ASIANPAINT, AUROPHARMA, AXISBANK,
BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BANKBARODA, BHARTIARTL, BPCL,
BRITANNIA, CANBK, CIPLA, COALINDIA, COLPAL, DABUR, DIVISLAB, DLF,
DMART, DRREDDY, EICHERMOT, GAIL, GODREJCP, GRASIM, HAVELLS,
HCLTECH, HDFCBANK, HDFCLIFE, HEROMOTOCO, HINDALCO, HINDUNILVR,
ICICIBANK, INFY, IOC, ITC, JSWSTEEL, JUBLFOOD, KOTAKBANK, LT, LUPIN,
M&M, MARICO, MARUTI, MOTHERSON, MPHASIS, NESTLEIND, NTPC, ONGC,
PAGEIND, PERSISTENT, PIDILITIND, PNB, POWERGRID, RELIANCE, SBILIFE,
SBIN, SHRIRAMFIN, SIEMENS, SUNPHARMA, TATACONSUM, TATASTEEL, TCS,
TECHM, TITAN, TVSMOTOR, ULTRACEMCO, WIPRO
```

### Cluster 2 — 6 stocks

```text
ADANIENT, ADANIPORTS, AMBUJACEM, ETERNAL, INDUSINDBK, TRENT
```

The result shows that behavioural groups are not simply sector groups. Cluster 1
contains stocks from banking, IT, pharma, auto, FMCG, energy, utilities, retail,
and other sectors. Cluster 2 also combines multiple sectors.

---

## 7. Stage 5 — Similarity search

The similarity implementation is in
[src/similarity.py](./src/similarity.py).

It uses cosine similarity over the standardized behavioural fingerprints. The
result describes historical similarity only.

The current configured example query is `ABB`. Its closest historical neighbours
are:

| Rank | Stock | Similarity | Sector |
|---:|---|---:|---|
| 1 | SIEMENS | 0.8475 | Capital Goods |
| 2 | PERSISTENT | 0.7439 | IT |
| 3 | DIVISLAB | 0.7438 | Pharma & Health |
| 4 | HEROMOTOCO | 0.7142 | Auto |
| 5 | TVSMOTOR | 0.6834 | Auto |

Output:

- [similarity_results.csv](./data/processed/similarity_results.csv)

The cross-sector neighbours support the idea that behavioural similarity can
differ from industry classification.

---

## 8. Stage 6 — Risk analysis and diversification

The financial-analysis implementation is in
[src/finance.py](./src/finance.py).

The analysis uses:

- Adjusted-close returns
- Equal-weight portfolios
- Daily rebalancing
- Maximum drawdown
- Annualised return
- Annualised volatility
- Average pairwise constituent correlation
- Deterministic centroid-distance stock selection

Portfolio selection does not rank stocks by return, so the result is not tuned
after seeing performance.

### Cluster-level risk and return

The canonical cluster summary is stored in:

- [cluster_summary.csv](./data/processed/cluster_summary.csv)

| Cluster | Stocks | Annualised return | Volatility | Maximum drawdown | Within-cluster correlation |
|---:|---:|---:|---:|---:|---:|
| 1 | 69 | 16.58% | 13.88% | 18.92% | 0.265 |
| 2 | 6 | 19.32% | 26.87% | 38.11% | 0.350 |

Cluster 2 had higher historical return, but also materially higher historical
volatility and drawdown.

### Portfolio A versus Portfolio B

Portfolio A selects stocks from one cluster. Portfolio B selects representative
stocks across clusters.

| Portfolio | Holdings | Annualised return | Volatility | Maximum drawdown | Avg. correlation |
|---|---|---:|---:|---:|---:|
| Portfolio A — Cluster 1 | ULTRACEMCO, BPCL | 14.91% | 21.70% | 28.78% | ~0.37 |
| Portfolio A — Cluster 2 | ADANIPORTS, AMBUJACEM | 12.53% | 32.35% | 45.62% | ~0.68 |
| Portfolio A — median | All eligible same-cluster portfolios | 13.72% | 27.03% | 37.20% | 0.526 |
| **Portfolio B — cross-cluster** | **ULTRACEMCO, ADANIPORTS** | **17.49%** | **25.33%** | **27.93%** | **0.390** |

Complete results, including every `k`, raw/PCA protocol, portfolio selection,
and robustness draws, are stored in:

- [portfolio_results.csv](./data/processed/portfolio_results.csv)
- [finance_selection.csv](./data/processed/finance_selection.csv)
- [finance_correlation_comparison.csv](./data/processed/finance_correlation_comparison.csv)

### Best historical diversification result

Portfolio B is the strongest overall diversification result in this experiment
because it had:

- Higher annualised return than the median Portfolio A
- Lower volatility than the median Portfolio A
- Lower maximum drawdown than both same-cluster cases
- Lower average constituent correlation than the median Portfolio A

However, the single Cluster 1 Portfolio A had the lowest volatility at 21.70%.
Therefore the precise conclusion is:

> Cross-cluster selection provided the best overall historical balance of return,
> drawdown, and correlation, while the stable Cluster 1 portfolio had the lowest
> volatility in this particular test.

The canonical robustness comparison produced:

- Indicative Mann–Whitney p-value: `0.0789`
- Rank-biserial effect: `-0.0454`

These p-values are indicative, not confirmatory, because the random portfolios
share stocks and are not independent samples.

---

## 9. Reproducibility

The stages can be run in order with the configured Python interpreter:

```powershell
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.data_loader
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.data_quality
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.cleaning
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.features
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.clustering
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.similarity
C:/Users/saich/AppData/Local/Programs/Python/Python313/python.exe -m src.finance
```

After the raw cache exists, the downstream stages run from local files and do
not require another Yahoo Finance download.

Configuration, tickers, dates, thresholds, random seeds, and output paths are
centralized in [src/config.py](./src/config.py).

---

## 10. Limitations

1. Results are historical and do not forecast future returns.
2. The universe is limited to 75 NSE stocks and one market.
3. The study uses one fixed historical window.
4. Yahoo Finance adjusted data can still contain corporate-action errors.
5. The canonical cluster split is highly imbalanced: 69 stocks versus 6.
6. Internal clustering metrics do not prove that clusters are economically
   permanent or causally meaningful.
7. Portfolio robustness samples share constituents, so the statistical test is
   indicative rather than definitive.
8. The portfolio experiment is a research comparison, not an optimized portfolio
   or investment recommendation.

---

## 11. Project status

Completed:

- Stage 0: environment and configuration
- Stage 1: data loading and caching
- Stage 1b: data-quality validation
- Stage 2: calendar alignment and cleaning
- Stage 3: behavioural fingerprints
- Stage 4: PCA and K-Means clustering
- Stage 5: similarity search
- Stage 6: financial risk and diversification analysis

The completed technical outputs are in
[data/processed/](./data/processed/). The historical findings summarized in this
report are ready to use in the final presentation or academic/project write-up.

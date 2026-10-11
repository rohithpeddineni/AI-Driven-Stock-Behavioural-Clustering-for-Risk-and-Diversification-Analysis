# SCOPE — AI-Based Stock Behavioural Clustering for Risk and Diversification Analysis

Authoritative specification of **what** this project is.
For **how** to build it (rules, layout, conventions), see [agent.md](agent.md).

---

## 1. Thesis

Each stock has a *behavioural fingerprint*: a fixed-length vector of statistics
describing how it has historically moved. Unsupervised ML groups stocks by that
fingerprint. Portfolio Theory then interprets whether those AI-discovered groups
differ in return and risk, and whether picking across groups diversifies better
than picking within one.

Balance: **~75% AI/ML, ~25% Financial Engineering.** Finance is the evaluation
layer, not the core.

## 2. Research Questions

| # | Question | Answered by |
|---|---|---|
| RQ1 | Can ML discover meaningful behavioural groups among stocks? | Clustering + internal validity metrics (§6) |
| RQ2 | Which stocks share similar historical fingerprints? | Similarity search (§5.4) |
| RQ3 | Do the discovered groups differ in return and risk? | Cluster-level financial stats (§5.5) |
| RQ4 | Does cross-cluster selection diversify better than same-cluster? | Portfolio A vs B comparison (§5.5) |

RQ4 is the headline result.

## 3. Data

- **Source:** Yahoo Finance via `yfinance`. Other public sources are reference only.
- **Universe:** 50–100 NSE-listed stocks spanning multiple sectors.
- **History:** 3–5 years of daily bars.
- **Fields:** Date, Open, High, Low, Close, Adjusted Close, Volume.
- **Benchmark:** one broad index (e.g. `^NSEI`) for beta and market comparison.

Returns are computed from **Adjusted Close**.

## 4. Feature Set — the Behavioural Fingerprint

> **⚠ SUPERSEDED — build from [CONTEXT.md](CONTEXT.md) §4, not this table.**
> The table below is the original proposal. Four of its entries were cut after
> analysis and a fifth after measurement: *mean daily return* (ρ≈1.0 with annualised
> return), *price-vs-moving-average ratios* (a noisy single-day snapshot, redundant
> with trend strength), *rolling-volatility mean* (near-duplicate of annualised
> volatility — the *dispersion* half was kept), and **mean volume**, which is a
> company-size proxy: clustering on it would rediscover market cap rather than
> behaviour. The final set is **14 features**, each justified in CONTEXT.md §4.

One row per stock. Every feature must have a stated reason for inclusion.

| Group | Features | Captures |
|---|---|---|
| Return | mean daily return, annualised return | Typical performance |
| Risk | volatility, rolling-volatility mean & dispersion, max drawdown | Magnitude and persistence of risk |
| Market | beta vs index | Sensitivity to the market |
| Trend | short- and medium-term momentum, trend strength, price-vs-moving-average ratios | Directional character |
| Volume | mean volume (normalised), volume volatility | Liquidity and participation |
| Distribution | skewness, kurtosis, lag-1 autocorrelation | Shape and stickiness of returns |

Features must be **scale-free** so stocks of different price levels compare fairly.

## 5. Method

### 5.1 Preparation
Clean and validate raw data, align calendars, compute returns.

### 5.2 Scaling and PCA
Standardize features. Use PCA for dimensionality reduction and 2-D visualization;
report explained variance.

### 5.3 Clustering
- **Primary:** K-Means.
- **Optional comparison:** Hierarchical Clustering, DBSCAN.
- Cluster count `k` is chosen from quantitative evidence (§6), never arbitrarily.

### 5.4 Similarity Search
Given one stock, return the most behaviourally similar stocks using cosine
similarity, Euclidean distance, or KNN over the fingerprint space.
Similarity describes **history only** — never a forecast.

### 5.5 Financial Analysis
Per cluster: average return, volatility, maximum drawdown, and the
within-cluster correlation profile. Then compare between-cluster correlation
against within-cluster correlation, and build two equally weighted portfolios:

- **Portfolio A** — stocks drawn from a single behavioural cluster.
- **Portfolio B** — stocks drawn across multiple clusters.

Compare return, volatility, correlation, and diversification characteristics.
The goal is interpretation, not an optimal portfolio.

## 6. Evaluation

**ML:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
Where practical, test stability across different time windows and seeds.

**Financial:** whether clusters differ meaningfully in return, volatility, max
drawdown, and correlation — then the Portfolio A vs B comparison.

## 7. Workflow

```text
Yahoo Finance
     ↓  fetch
Raw daily OHLCV
     ↓  clean, validate
Aligned price panel
     ↓  returns + feature engineering
Behavioural fingerprints (one row per stock)
     ↓  scale → PCA
Reduced feature space
     ↓  K-Means / alternatives
Behavioural clusters
     ├──→ Similar Stock Finder
     └──→ Return / Risk / Correlation
                 ↓
          Diversification comparison
                 ↓
          Streamlit dashboard
```

## 8. Dashboard

| Page | Shows |
|---|---|
| 1. Stock Fingerprint | Selected stock's features, price history, risk measures, assigned cluster |
| 2. AI Clusters | Cluster membership, PCA scatter, cluster sizes, per-cluster feature profiles |
| 3. Similar Stock Finder | Nearest stocks by fingerprint with side-by-side feature comparison |
| 4. Risk & Diversification | Cluster return/volatility tables, correlation heatmaps, Portfolio A vs B |

## 9. Deliverables

Data pipeline · processed feature dataset · ML pipeline · clustering and
similarity results · financial analysis · Streamlit dashboard · documentation ·
report · presentation.

## 10. Out of Scope

Exact price prediction · automated or real-money trading · complex valuation
(DCF, DDM) · derivatives and options pricing (Black-Scholes) · bonds, YTM,
duration, convexity, immunization · CAPM · APT · efficient-frontier or other
advanced portfolio optimization.

Add none of these unless explicitly requested later.

## 11. Optional Extensions

Only after the core system works end to end: market regime detection, anomaly
detection, temporal (time-varying) fingerprints, financial-news sentiment/NLP.

## 12. Success Criteria

1. Clusters are meaningful and interpretable, supported by §6 metrics.
2. The similarity search returns behaviourally defensible neighbours.
3. Cluster-level financial differences are measurable and reported.
4. The Portfolio A vs B comparison produces a clear, honestly stated answer to RQ4.
5. The whole workflow is reproducible from a clean checkout.

# MODEL TRAINING CONTEXT — AI Stock Fingerprint

## 1. Current Data

The coding agent already has raw Yahoo Finance data for each stock with:

`Date, Open, High, Low, Close, Adj Close, Volume`

There is **no target variable**. This is intentional because the core model is an **unsupervised learning** problem.

The goal is NOT to predict future stock prices.

The goal is:

> Convert each stock's historical behaviour into a numerical fingerprint, then use ML to discover stocks that behave similarly.

---

## 2. Correct Pipeline

```text
Raw Yahoo Finance Data
        ↓
Data Validation / Cleaning
        ↓
Daily Return Calculation
        ↓
Feature Engineering
        ↓
One Behavioural Fingerprint per Stock
        ↓
Feature Scaling
        ↓
PCA + Visualization
        ↓
K-Means Clustering
        ↓
Behavioural Clusters
        ↓
Similar Stock Finder
        ↓
Risk / Return / Correlation
        ↓
Diversification Experiment
```

**Important:** Do NOT train K-Means directly on every daily OHLCV row.

The final ML table should be approximately:

**75 stocks × ~16 behavioural features**

So one row represents one stock.

---

## 3. Data Validation

Before feature engineering, check:

- Valid and sorted dates
- Duplicate dates
- Missing values
- Missing `Adj Close`
- Non-positive prices
- Zero-volume days
- Extreme daily movements
- Coverage of the common date range

The existing data-quality fields such as:

`symbol, rows, first, last, coverage, missing_days, duplicate_dates, null_adj_close, nonpositive_price, zero_volume_days, extreme_moves, worst_move, adj_divergence`

are useful as **data-quality diagnostics**.

They are NOT the final behavioural fingerprint.

---

## 4. Price and Return Calculation

Use `Adj Close` for primary historical-return calculations when available.

```python
daily_return = Adj_Close.pct_change()
```

Keep OHLC and Volume because they can be used for technical/behavioural features.

Use one fixed date window for every stock.

Recommended project window:

**January 2022 – August 2026**

Record the exact start/end dates in configuration. Do not use a moving `last_5_years` window.

---

# 5. Recommended 16-Dimensional Stock Fingerprint

> **⚠ SUPERSEDED — build from [CONTEXT.md](CONTEXT.md) §4, not this list.**
> This list won over scope.md §4 on most points, including the decision to exclude
> average volume. Two of its own entries were then cut: *correlation to NIFTY*
> (since `beta = corr × σ_stock/σ_market` and σ_market is one constant across the
> universe, keeping volatility, beta and correlation means three columns carrying
> two dimensions) and *6-month momentum* (the overlapping middle of nested
> 63 ⊂ 126 ⊂ 252-day windows). *Downside volatility* was then removed by the §8
> redundancy check at +0.94 with volatility. Final set: **14 features**.

Calculate these features for every stock.

### Return / Risk

1. **Annualized Return**
   - Historical return based on adjusted prices.

2. **Annualized Volatility**
   - Standard deviation of daily returns × sqrt(252).

3. **Downside Volatility**
   - Volatility calculated from negative returns.

4. **Maximum Drawdown**
   - Largest fall from a previous running peak.

### Market Relationship

5. **Beta to NIFTY 50**
   - Requires a NIFTY 50 benchmark series with the same dates.
   - `beta = covariance(stock_return, market_return) / variance(market_return)`

6. **Correlation to NIFTY 50**
   - Correlation between stock and NIFTY daily returns.

### Momentum

7. **3-Month Momentum**
   - Approximately 63 trading-day return.

8. **6-Month Momentum**
   - Approximately 126 trading-day return.

9. **12-Month Momentum**
   - Approximately 252 trading-day return.

### Trend / Return Behaviour

10. **Trend Strength**
    - Fit a regression to log adjusted price against time.
    - Use normalized/annualized slope rather than raw rupee slope.

11. **Lag-1 Return Autocorrelation**
    - Captures persistence versus oscillating/mean-reverting behaviour.

12. **Return Skewness**
    - Measures asymmetry of the return distribution.

13. **Excess Kurtosis**
    - Captures fat-tail/extreme-movement behaviour.

### Volume Behaviour

14. **Volume Volatility**
    - Use normalized volume behaviour or log-volume changes.
    - Avoid raw average volume because it is strongly affected by company size.

15. **Return–Volume Correlation**
    - For example, correlation between absolute daily return and change in log volume.
    - Document the exact definition.

### Extreme Behaviour

16. **Extreme-Move Frequency**
    - Frequency of unusually large daily returns.
    - Use one consistent rule across all stocks.

These 16 features are the **baseline**. Before final training, check feature-feature correlation and remove/revise highly redundant features if necessary. Document any change.

---

# 6. Features That Must NOT Enter Clustering

Do NOT use:

- Symbol
- Company name
- Sector
- Industry
- Date
- Raw stock price
- Raw market capitalization
- P/E
- Any company label encoded as a number

**Sector must not be used as an ML input.**

Sector can only be used AFTER clustering to investigate:

> Do behavioural clusters simply rediscover sectors?

This is important to the research question.

---

# 7. NIFTY 50 Benchmark

Because Beta and market correlation are part of the proposed fingerprint, collect a NIFTY 50 benchmark series covering the exact same period.

Align dates before calculating Beta/correlation.

Do not use sector information in these calculations.

---

# 8. Feature Quality Checks

Before clustering:

1. Missing values
2. Infinite values
3. Near-constant features
4. Feature distributions
5. Pairwise feature correlation
6. Redundant features
7. Extreme feature values

Do not delete observations merely to make clusters look better.

If a transformation such as winsorization or a robust scaler is used, document it.

---

# 9. Scaling

Use feature scaling before distance-based clustering.

Recommended:

```python
StandardScaler()
```

Reason: Return, Beta, kurtosis, momentum, etc. have different numerical scales.

---

# 10. PCA

Run PCA on the standardized fingerprint dataset.

Use PCA mainly for:

- 2D visualization
- Understanding dominant behavioural dimensions
- Optional comparison with clustering on PCA components

Do not automatically replace the 16 original features with PCA components for clustering. Compare approaches and document the choice.

---

# 11. K-Means

Primary clustering algorithm:

**K-Means**

Test:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

Evaluate each value using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

Do not choose `k` only because the plot looks good.

Use the metrics + interpretability together.

---

# 12. Similar Stock Finder

After clustering, allow a user to select a stock, e.g.:

```text
TCS
```

Return the most behaviourally similar stocks.

Possible methods:

- Cosine similarity
- Euclidean distance
- KNN

Similarity must mean:

> similarity in historical behavioural fingerprint.

It must NOT be described as a prediction of future returns or an investment recommendation.

---

# 13. Financial Engineering Validation

The Financial Engineering component is deliberately simple and belongs mainly to **Portfolio Theory**.

After clustering, calculate:

- Return
- Volatility/risk
- Maximum drawdown
- Correlation

## Portfolio A — Same Behaviour

Select 5 stocks from one behavioural cluster.

Use equal weights.

## Portfolio B — Cross Behaviour

Select 1 stock from each of 5 different behavioural clusters.

Use equal weights.

Compare:

- Portfolio return
- Portfolio volatility
- Maximum drawdown
- Average pairwise correlation

Main question:

> Does selecting stocks from different behavioural clusters produce better historical diversification characteristics than selecting stocks from one behavioural cluster?

Do not force Portfolio B to win.

If it does not, report the actual result.

For reproducibility, use a deterministic selection rule or a fixed random seed. Do not cherry-pick stocks after seeing portfolio results.

---

# 14. Sector Analysis After Clustering

Sector labels are allowed only for interpretation.

Example:

```text
Cluster 1:
IT       30%
Banking  20%
Auto     20%
Other    30%
```

This allows the project to answer:

> Does behaviour-based clustering discover relationships that sector labels do not explain?

A particularly interesting result would be stocks from different sectors appearing in the same behavioural cluster.

---

# 15. Expected Output Files

The coding agent should produce:

### `cleaned_daily_data.csv`

```text
Date | Symbol | Open | High | Low | Close | Adj Close | Volume
```

### `stock_fingerprints.csv`

```text
Symbol | Feature_1 | ... | Feature_16
```

### `cluster_assignments.csv`

```text
Symbol | Cluster
```

### `cluster_summary.csv`

```text
Cluster | Number of Stocks | Avg Return | Avg Volatility | Avg Drawdown | ...
```

### `similarity_results.csv`

```text
Query Stock | Similar Stock | Similarity Score
```

### `portfolio_results.csv`

```text
Portfolio | Return | Volatility | Max Drawdown | Avg Correlation
```

---

# 16. Required Visualizations

At minimum:

1. Feature correlation heatmap
2. Feature distributions
3. PCA 2D scatter plot
4. Cluster visualization
5. Cluster feature profiles
6. Similar-stock comparison
7. Portfolio return comparison
8. Portfolio risk comparison
9. Portfolio correlation heatmap
10. Sector composition of each cluster (post-clustering only)

---

# 17. Recommended Project Structure

> **⚠ SUPERSEDED — the built layout follows [CONTEXT.md](CONTEXT.md) §6.**
> agent.md's layout won here, because the code already follows it and it keeps
> `notebooks/`, `tests/` and `docs/`, which the tree below drops. Specifically:
> no separate `data/cleaned/` tier (it and `processed/` would hold the same kind of
> thing); `features.py` stays one module rather than splitting into
> `feature_engineering.py` + `fingerprint.py`; `portfolio_analysis.py` → `finance.py`;
> `visualization.py` → `plots.py`; and the dashboard is `app/Home.py` + `app/pages/`
> rather than a single `streamlit_app.py`, because scope.md §8 defines four pages.
> **The §15 output filenames above were adopted exactly** — that part is authoritative.

```text
project/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── feature_engineering.py
│   ├── fingerprint.py
│   ├── clustering.py
│   ├── similarity.py
│   ├── portfolio_analysis.py
│   └── visualization.py
├── notebooks/
├── app/
│   └── streamlit_app.py
├── outputs/
├── requirements.txt
└── README.md
```

---

# 18. Execution Order

The coding agent should implement in this order:

1. Load existing Yahoo Finance raw files.
2. Validate and clean them.
3. Create the common date window.
4. Calculate daily returns.
5. Collect/align NIFTY 50 data.
6. Engineer the 16 fingerprint features.
7. Create the one-row-per-stock fingerprint dataset.
8. Run feature-quality/redundancy checks.
9. Standardize features.
10. Run PCA.
11. Sweep K-Means for `k=2..8`.
12. Evaluate clustering metrics.
13. Select and save the final clustering.
14. Build similarity search.
15. Run the same-cluster vs cross-cluster portfolio experiment.
16. Generate visualizations.
17. Save all intermediate/final datasets.
18. Build the Streamlit dashboard.
19. Document methodology and results.

---

# 19. Core Research Story

The project should communicate:

> Sector labels describe what a company does, but they do not necessarily describe how its stock moves.

Therefore:

```text
Historical prices
       ↓
Behavioural features
       ↓
Stock fingerprints
       ↓
Unsupervised ML
       ↓
Behavioural clusters
       ↓
Portfolio Theory
       ↓
Risk + Return + Correlation
       ↓
Diversification test
```

The central research question is:

> **Do behaviour-based stock clusters reveal diversification structure that sector classification alone may miss?**

---

# 20. Do Not Overcomplicate

Do NOT turn the project into:

- Stock price prediction
- LSTM forecasting
- Random Forest price prediction
- DCF
- CAPM as the main model
- Black-Scholes
- Bond analysis
- Complex efficient-frontier optimization
- Automated trading

The intended project is:

**Strong feature engineering + unsupervised ML + interpretable clustering + simple Portfolio Theory validation.**

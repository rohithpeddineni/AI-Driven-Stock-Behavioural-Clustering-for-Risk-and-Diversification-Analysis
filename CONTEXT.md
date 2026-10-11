# CONTEXT — AI-Based Stock Behavioural Clustering

**Single orientation document.** Read this first; it says what the project is, what
has been decided, what is built, and what comes next. Where the older specification
files disagree with each other, **this file records which one won and why** — those
conflicts are resolved, not open.

| File | Role |
|---|---|
| `CONTEXT.md` | This file. Current state and all resolved decisions. |
| `scope.md` | Original specification of *what* the project is. |
| `agent.md` | Operating rules and coding conventions. |
| `model_training_context.md` | Detailed ML specification (features, protocols, outputs). |
| `PLAN.md` | Stage tracker, verification log, open decisions. |
| `docs/NOTES.md` | Plain-English log of what each stage did, written as it happened. |

---

## 1. The project

Each stock gets a **behavioural fingerprint**: a fixed-length row of numbers
describing how it has historically moved. Unsupervised ML groups stocks by that
fingerprint. Portfolio theory then asks whether those AI-discovered groups actually
differ in return and risk, and whether picking across groups diversifies better than
picking within one.

Balance: **~75% AI/ML, ~25% financial engineering.** Finance is the evaluation layer,
not the core.

| # | Research question | Answered by |
|---|---|---|
| RQ1 | Can ML discover meaningful behavioural groups? | Clustering + validity metrics |
| RQ2 | Which stocks share similar fingerprints? | Similarity search |
| RQ3 | Do the groups differ in return and risk? | Cluster-level financial stats |
| RQ4 | **Does cross-cluster selection diversify better than same-cluster?** | Portfolio A vs B |

RQ4 is the headline. The framing question underneath all four:

> Sector labels describe what a company *does*. They do not necessarily describe how
> its stock *moves*. Do behaviour-based clusters reveal structure that sector
> classification misses?

### Hard constraints

1. **Nothing predicts prices.** No forecasting, no LSTM, no price targets.
2. **Never claim clusters or similarity predict future returns.** History only.
3. **Never frame output as investment advice.** No trades, no broker APIs, no real money.
4. **Sector is never an ML input** — only an interpretation aid after clustering.
   Using it as a feature would make "did clusters rediscover sectors?" circular.
5. **Out of scope:** DCF, CAPM as the main model, Black-Scholes, bonds/duration,
   efficient-frontier optimisation, automated trading.
6. **Every feature needs a stated behavioural justification.** If you cannot say what
   it captures and why it matters, it does not go in.
7. **Determinism:** fixed `random_state` everywhere; identical inputs must give
   identical outputs. No rolling "last 5 years" windows.

---

## 2. Current state

**Built and verified:** Stages 0–6. **Next:** Stage 7 (Streamlit dashboard).

| Stage | Module | Status |
|---|---|---|
| 0 | `.venv`, `requirements.txt`, git | ✅ Done |
| 1 | `src/config.py`, `src/data_loader.py` | ✅ Done — 76 symbols cached |
| 1b | `src/data_quality.py` | ✅ Done — raw data passes all checks |
| 2 | `src/cleaning.py` | ✅ Done — 85,500-row aligned panel |
| 3 | `src/features.py` | ✅ Done — 75 × 14 fingerprint table |
| 4 | `src/clustering.py` | ✅ Done — deterministic raw/PCA K-Means sweep |
| 5 | `src/similarity.py` | ✅ Done — cosine historical-neighbour search |
| 6 | `src/finance.py` | ✅ Done — cluster risk and Portfolio A/B analysis |
| 7 | `app/Home.py` + `app/pages/` | ⬜ |
| 8 | Report, presentation | ⬜ |

### Run it

```bash
python -m src.data_loader    # fetch (or read cache — no re-download)
python -m src.data_quality   # raw data quality report
python -m src.cleaning       # aligned daily panel
python -m src.features       # fingerprints + feature quality report
```

Every module runs standalone and prints a summary — that is the intended way to
learn the pipeline, one piece at a time.

---

## 3. Data — verified, not assumed

- **Source:** Yahoo Finance via `yfinance`. Google Finance has had no public API
  since 2012; NSE Bhavcopy is authoritative but unadjusted; Alpha Vantage allows 25
  calls/day; Stooq is bot-blocked. Yahoo is the only free source giving **adjusted
  close** for NSE at this scale, and adjusted close is non-negotiable — without it a
  1:5 split reads as a −80% day and wrecks every risk feature.
- **Universe:** 75 NSE stocks across 16 sectors, plus `^NSEI` as benchmark.
- **Window:** `2022-01-03` → `2026-08-14`, **1,140 aligned trading days**, held as
  fixed absolute dates in `config.START` / `config.END`.
- **Cache:** `data/raw/`, ~8 MB, one CSV per symbol plus `_manifest.json`. Fetched
  once; the pipeline then runs entirely offline, which is what makes results
  reproducible regardless of Yahoo's availability.

### Exclusions, and why

| Ticker | Reason |
|---|---|
| `PAYTM`, `NYKAA` | Listed Nov 2021 — still in post-IPO price discovery at window open. |
| `TMPV` (ex-`TATAMOTORS`) | Oct-2025 demerger: the business changed shape mid-window. |
| `LTIM` / `LTIMINDTREE` | 404 on both spellings. |
| `VEDL` | **Caught by the quality check.** 2026-04-30: ₹773.60 → ₹271.55, a −65% day, with `Close == Adj Close` — Yahoo never adjusted for the Vedanta demerger. |

### Four gotchas that will bite again

**1. The project folder name contains a zero-width space (U+200B).** It is invisible
everywhere. It breaks the SSL certificate path (`curl error 77` — *every* download
fails) and crashes any attempt to print a path on a cp1252 console. Worked around by
`config.ensure_ascii_ca_bundle()` and `config.configure_console()`.
**Renaming the folder is the real fix** — expect this to resurface in git paths,
shell quoting and Streamlit's file watcher.

**2. Symbols rot and Yahoo does not redirect.** Zomato → `ETERNAL`, Tata Motors →
`TMPV`. `data_loader.py` therefore **raises on failure rather than skipping**: a
silent skip would quietly shrink the universe between runs.

**3. Adjusted close is usually right, but not always.** VEDL proves it. Check for
unadjusted corporate actions (`Close == Adj Close` alongside a huge single-day move)
rather than assuming the adjustment happened.

**4. Yahoo pads market holidays** into stock files but not the index file — zero
volume, previous close repeated. These inject fake zero-return days that deflate
volatility. Removed by intersecting each stock's calendar with the index's, rather
than hard-coding dates, so it keeps working when data is refreshed.

---

## 4. The fingerprint — 14 features

One row per stock, `Symbol` as index. All computed from **adjusted close** over the
identical window. All **scale-free**, so a ₹200 stock and a ₹4,000 stock compare on
behaviour, not price level.

| # | Feature | Definition | Captures |
|---|---|---|---|
| 1 | `annualised_return` | `mean(r) × 252` | Typical performance |
| 2 | `annualised_volatility` | `std(r) × √252` | Risk magnitude |
| 3 | `max_drawdown` | `max(1 − P/cummax(P))` | Worst peak-to-trough fall |
| 4 | `rolling_vol_dispersion` | `std` of 60-day rolling annualised vol | Risk *stability* vs. risk *level* |
| 5 | `beta` | `cov(r, r_mkt) / var(r_mkt)` | Market sensitivity |
| 6 | `momentum_3m` | `P_t / P_{t−63} − 1` | Recent direction |
| 7 | `momentum_12m` | `P_t / P_{t−252} − 1` | Long-horizon direction |
| 8 | `trend_strength` | OLS slope of `log P` on time × 252 | Trend *consistency* |
| 9 | `autocorr_lag1` | `corr(r_t, r_{t−1})` | Persistence vs. mean reversion |
| 10 | `skewness` | Sample skew of returns | Distribution asymmetry |
| 11 | `excess_kurtosis` | Fisher excess kurtosis | Fat tails |
| 12 | `volume_volatility` | `std(Δ log Volume)` | Participation instability |
| 13 | `return_volume_corr` | `corr(\|r\|, Δ log Volume)` | Event-driven vs. steady |
| 14 | `extreme_move_frequency` | Share of days with `\|r\| > 2σ_stock` | Tail-event rate |

### Why not 16 — five features cut, with reasons

Cut **before** any code ran, by analysis:

- **Mean daily return** — the same statistic as annualised return at a different
  scale (ρ ≈ 1.0).
- **Price vs 200-day MA** — a single last-day snapshot; noisier than every
  window-aggregated feature and redundant with trend strength plus momentum.
- **Average volume** — a **company-size proxy, not a behaviour.** Self-normalised it
  trivialises to ≈1.0; cross-sectionally normalised it needs shares-outstanding, a
  fundamentals field outside `scope.md` §3. Clustering on it would rediscover market
  cap. *(This overrides `scope.md` §4, which asked for it — `model_training_context.md`
  §5.14 wins.)*
- **Correlation to NIFTY** — since `beta = corr × (σ_stock / σ_market)` and σ_market
  is a single constant across all 75 stocks, keeping volatility, beta *and*
  correlation puts three columns of only two independent dimensions into the distance
  metric K-Means uses.
- **6-month momentum** — the overlapping middle of nested 63 ⊂ 126 ⊂ 252-day windows.
  The two extremes carry more independent information.

Cut **after** measurement, by the quality check:

- **Downside volatility** — measured +0.94 with annualised volatility and +0.91 with
  rolling-vol dispersion. It was the redundancy hub; removing it cleared both flagged
  pairs and left survivors at 0.83.

### Two deliberate judgment calls

- **`annualised_return` ↔ `trend_strength` measure +0.904 — just *above* the 0.90
  threshold — and both were kept anyway.** One measures how far the stock travelled,
  the other how consistently it got there. This is a deliberate override of the rule,
  recorded here rather than hidden by nudging the threshold to 0.91. It is the highest
  remaining correlation in the table and should be revisited after clustering.
- **`autocorr_lag1` is barely above noise** (spread 0.042 vs. a sampling floor of
  1/√1139 ≈ 0.030) but has the *lowest* correlation with anything else (0.29), making
  it the most independent column in the table. Kept; first candidate to cut if it
  contributes nothing to the clusters.

### Feature-quality rule

The quality report is always computed on the **full** feature set *before* any
removal, so the evidence justifying each cut stays visible in
`feature_quality_report.csv`. Features are flagged and documented, never silently
dropped to make clustering metrics look better.

---

## 5. Protocols for the remaining stages

### Stage 4 — Clustering

`StandardScaler` on the 14 features, then **K-Means fit twice**:

1. **Primary:** on the 14-D standardized matrix.
2. **Comparison:** on PCA components retained to ~85% cumulative explained variance.

Both swept over `k = 2..8`, scored by **Silhouette, Davies-Bouldin and
Calinski-Harabasz**, with an **Adjusted Rand Index** between the two label sets at
matched `k` to quantify how much the choice of space changes membership.

Raw-feature clustering is canonical unless PCA-space is *clearly* better (materially
higher Silhouette **and** lower Davies-Bouldin — not a marginal wobble). `k` is chosen
from the metric table plus interpretability, never because a plot looks nice.

**The 2-D scatter is separate**: always fit `PCA(n_components=2)` on the standardized
matrix purely for display coordinates, and paint the canonical labels onto it. The
scatter is a window onto the feature space, never a re-run clustering — which is what
keeps labels and plot consistent.

### Stage 5 — Similarity

Cosine similarity over the scaled fingerprints; return top-N neighbours for a chosen
stock. Means **similarity of historical behaviour** — never a forecast, never a
recommendation.

### Stage 6 — Portfolio A vs B (RQ4)

**Holdings:** `n(k) = min(5, k)`. Portfolio B can never hold more stocks than there
are clusters. Clusters too small to fill Portfolio A are **skipped and logged**, not
used to shrink every portfolio.

**Selection rule, fixed before any metric is seen** — it depends only on cluster
geometry, never on returns, so it cannot be cherry-picked afterwards:
rank each cluster's members by Euclidean distance to their centroid in standardized
space; ties break alphabetically.
- **Portfolio B:** the medoid (closest to centroid) of each selected cluster. If
  `k > 5`, use the 5 largest clusters.
- **Portfolio A:** the `n(k)` closest-to-centroid members of one cluster.

**Fairness — Portfolio A is built for *every* cluster,** then reported as the median
across clusters plus the min–max range. This removes the researcher's discretion
entirely. A wide range is itself a finding: it shows single-cluster portfolios are
fragile to which cluster you happen to land in.

**Robustness —** 1,000 random cross-cluster draws vs. 1,000 within-cluster draws from
a single seeded RNG consumed in fixed order, compared with Mann-Whitney U plus effect
size. Because draws share tickers the samples are not independent, so p-values are
**indicative, not confirmatory** — state this rather than over-claiming.

**Run the whole protocol for every `k = 2..8`.** The metric-selected `k` is the
headline; the full table is the robustness appendix showing whether RQ4's answer
depends on which `k` you landed on. This is what dissolves the `k < 5` contradiction
instead of patching it.

**Metrics:** equal-weight, **rebalanced daily**. Under buy-and-hold, weights drift
with performance — a stock that doubles dominates the "equal-weight" portfolio by
year three, so any A-vs-B difference could reflect weight drift rather than genuine
diversification. Compute annualised return, annualised volatility, max drawdown, and
average pairwise correlation of constituents.

**Reporting:** publish the per-k table, the per-cluster A breakdown and the
distribution statistics alongside the headline. **If cross-cluster does not
diversify better, report that with the numbers** — a valid answer to RQ4, not a
failure. Never tune until the result flips. Anticipate a mixed result (lower
correlation but not lower volatility) and report it as partial support rather than
rounding it up or down.

---

## 6. Layout and output contract

```text
data/
  raw/                          # cached downloads, gitignored, never hand-edited
    _manifest.json
  processed/
    cleaned_daily_data.csv      # Stage 2 — long: Date|Symbol|OHLCV, 85,500 rows
    cleaning_drop_log.csv
    data_quality_report.csv     # Stage 1b
    stock_fingerprints.csv      # Stage 3 — 75 × 14
    feature_quality_report.csv  # Stage 3
    cluster_metrics.csv         # Stage 4 — k-sweep evidence, both feature spaces
    cluster_assignments.csv     # Stage 4
    similarity_results.csv      # Stage 5
    cluster_summary.csv         # Stage 6
    portfolio_results.csv       # Stage 6
src/
  config.py        # every constant — no magic numbers elsewhere
  data_loader.py   # the ONLY module allowed to touch the network
  data_quality.py  # raw-data diagnostics
  cleaning.py      # calendar alignment → cleaned_daily_data.csv
  features.py      # returns, fingerprint, feature-quality checks
  clustering.py    # scaling, PCA, k-sweep, K-Means, metrics
  similarity.py
  finance.py       # cluster stats, Portfolio A vs B
  plots.py         # shared figures; returns figures, never computes
app/
  Home.py
  pages/           # 1_Stock_Fingerprint, 2_AI_Clusters,
                   # 3_Similar_Stock_Finder, 4_Risk_and_Diversification
docs/              # NOTES.md, figures/
notebooks/         # exploration only — never imported by src/ or app/
tests/
```

**Storage rule: one stored copy of any number.** `cleaned_daily_data.csv` is long
format because the fingerprint needs Volume and OHLC as well as adjusted close, and a
wide layout would mean six separate wide tables that can silently drift apart. Wide
panels are pivoted **in memory** via `cleaning.adjusted_close_panel()` wherever
correlation matrices or portfolio maths need them, and never written to disk.

### Conventions

- Constants live in `config.py`; nothing hard-codes a ticker, date or window length.
- Network calls live only in `data_loader.py`.
- Functions return **data**; plotting lives in `plots.py`; Streamlit pages render and
  never compute.
- Drop tickers with insufficient history rather than imputing long gaps — and log
  every drop with its reason.
- Type-hint public functions; one-line docstring saying what and why; comment only
  where the *reason* is non-obvious.
- Every module has a `__main__` block that runs its stage and prints a summary.

### Ask before

Changing the ticker universe, date range or benchmark · adding a dependency outside
the stack · introducing anything from `scope.md` §10 (out of scope) or §11
(extensions) · any change shifting the 75/25 AI-to-finance balance.

---

## 7. Reporting standard

State findings plainly, including negative ones. Always attach the metric values
behind a claim. Name the limitations where they matter:

single market · single ~4.6-year window · in-sample by construction (features,
clusters and evaluation all use the same period) · Indian large-caps share common
market beta, which puts a structural ceiling on achievable diversification · no
transaction costs · small clusters give noisy estimates · testing across 7 values of
`k` and 4 metrics raises multiple-comparison risk, so lead with the full table rather
than the smallest p-value.

This project **describes historical structure. It does not forecast.**

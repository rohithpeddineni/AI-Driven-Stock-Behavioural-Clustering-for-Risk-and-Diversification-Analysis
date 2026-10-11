# DATA PROCESSING — what we did and what came out

A single reference for everything that happens **before** clustering: download →
quality check → cleaning → fingerprint. `docs/NOTES.md` is the running diary of the
build; this file is the summary you can hand to someone and have them understand the
data in ten minutes. All numbers here are the actual measured values from the current
run (2026-08-17), not targets.

---

## 1. The pipeline at a glance

| Stage | Command | Reads | Writes | Result |
|---|---|---|---|---|
| 1 | `python -m src.data_loader` | Yahoo Finance (once) | `data/raw/*.csv`, `_manifest.json` | 76 symbols cached, ~8 MB |
| 1b | `python -m src.data_quality` | `data/raw/` | `data_quality_report.csv` | 75 stocks pass; `VEDL` excluded |
| 2 | `python -m src.cleaning` | `data/raw/` | `cleaned_daily_data.csv`, `cleaning_drop_log.csv` | 85,500 rows = 75 × 1,140 days |
| 3 | `python -m src.features` | `cleaned_daily_data.csv` | `stock_fingerprints.csv`, `feature_quality_report.csv` | 75 × 14 fingerprint table |

Shape of the funnel: **~87,000 raw daily bars → 75 rows of 14 numbers.** That
compression is the whole point — you cannot cluster 75 time series, but 75 short rows
cluster easily.

Every module is standalone and prints its own summary. Run them in order the first
time; after that the cache means only the stage you changed needs re-running.

---

## 2. Stage 1 — Download

- **Source:** Yahoo Finance via `yfinance`. Chosen because it is the only free source
  giving **adjusted close** for NSE at this scale. Adjusted close is non-negotiable:
  a 1:5 split would otherwise read as a −80% day and wreck volatility, drawdown and
  kurtosis all at once.
- **Universe:** 75 NSE stocks across 16 sectors + `^NSEI` as the benchmark.
- **Window:** fixed absolute dates in `config.START` / `config.END`, never a rolling
  "last 5 years" — reruns must reproduce, not drift.
- **Columns kept:** Open, High, Low, Close, Adj Close, Volume.
- **Cache-first:** a second run downloads nothing. After the first fetch the entire
  project runs offline, so results do not change because Yahoo changed overnight.
- **Fails loudly:** a symbol that 404s raises instead of being skipped. Tickers rot
  (Zomato → `ETERNAL`, Tata Motors → `TMPV`) and Yahoo does not redirect — a silent
  skip would quietly shrink the universe between runs.
- `_manifest.json` records row count, first/last date and a hash per file, so a stale
  or half-finished cache is detectable rather than silently wrong.

### Tickers excluded, and why

| Ticker | Reason |
|---|---|
| `PAYTM`, `NYKAA` | Listed Nov 2021 — still in post-IPO price discovery at window open |
| `TMPV` (ex-`TATAMOTORS`) | Oct-2025 demerger: the business changed shape mid-window |
| `LTIM` / `LTIMINDTREE` | 404 on both spellings |
| `VEDL` | Caught by the Stage-1b check — see below |

---

## 3. Stage 1b — Quality check (before any feature work)

Asks one question: *can this data be trusted?* Checks duplicate dates, missing or
non-positive adjusted closes, insufficient history (`MIN_HISTORY_FRACTION = 0.95`),
and single-day moves beyond `EXTREME_RETURN_THRESHOLD = 40%`.

**Verdict: clean.** No duplicates, no missing closes, no non-positive prices, no
ticker short on history. Row counts 1,140–1,144. Two real problems surfaced:

**1. Vedanta (`VEDL`) was silently corrupt.** 2026-04-30: ₹773.60 → ₹271.55, a −65%
day, with `Close == Adj Close` throughout — Yahoo never adjusted for the demerger.
Left in, that one fake day would have dominated the stock's volatility, drawdown,
skewness and kurtosis; the fingerprint would have described a data error rather than a
company. **Excluded.** The lesson generalises: adjusted close is usually right, but not
always — test for it rather than assume it.

**2. Four phantom trading days.** 2026-01-15, 2026-05-01, 2026-05-28, 2026-06-26 appear
in every stock's file but *not* in the index's. Zero volume, previous close repeated —
market holidays Yahoo padded. They inject fake zero-return days that deflate every
volatility estimate. They are **not** deleted from `data/raw/`; raw data stays exactly
as downloaded and Stage 2 removes them.

**Universe after this stage: 75 stocks + benchmark.**

---

## 4. Stage 2 — Cleaning

`cleaned_daily_data.csv`: **85,500 rows = exactly 75 symbols × 1,140 trading days.**

- **Calendar alignment.** Each stock's dates are *intersected* with the index's trading
  calendar. That removes the padded holidays without hard-coding the four dates, so the
  same code keeps working when the data is refreshed and Yahoo pads different days.
  **295 rows dropped**, each logged in `cleaning_drop_log.csv` with its reason
  (~4 per symbol).
- **Nothing was forward-filled.** No gaps survived alignment.
- **Why every stock gets identical days:** a stock measured over even a slightly
  different period gets an unfair advantage or handicap in the fingerprint.
- **Long format, not wide** — one row per (Date, Symbol) with all of OHLCV. The
  fingerprint needs Volume as well as Adj Close, and six parallel wide tables can
  silently drift apart. Where wide *is* needed (correlations, portfolio maths),
  `cleaning.adjusted_close_panel()` pivots in memory. **Rule: one stored copy of any
  number; derived views computed on demand.**

---

## 5. Stage 3 — The 14-feature fingerprint

One row per stock. All from **adjusted close** (plus Volume) over the identical window.
All **scale-free**, so a ₹200 stock and a ₹4,000 stock compare on behaviour, not price.

### The features, with measured range across the 75 stocks

| # | Feature | Definition | Captures | Mean | Range (stock at each end) |
|---|---|---|---|---|---|
| 1 | `annualised_return` | `mean(r) × 252` | Typical performance | 0.167 | −0.095 WIPRO → 0.468 TVSMOTOR |
| 2 | `annualised_volatility` | `std(r) × √252` | Risk magnitude | 0.273 | 0.190 ITC → 0.490 ADANIENT |
| 3 | `max_drawdown` | `max(1 − P/cummax(P))` | Worst peak-to-trough fall | 0.372 | 0.167 MARICO → 0.733 TRENT |
| 4 | `rolling_vol_dispersion` | `std` of 60-day rolling annualised vol | Risk *stability* vs *level* | 0.064 | 0.029 NESTLEIND → 0.229 ADANIENT |
| 5 | `beta` | `cov(r, r_mkt) / var(r_mkt)` | Market sensitivity | 0.969 | 0.472 CIPLA → 1.734 ADANIENT |
| 6 | `momentum_3m` | `P_t / P_{t−63} − 1` | Recent direction | 0.060 | −0.210 ONGC → 0.321 ETERNAL |
| 7 | `momentum_12m` | `P_t / P_{t−252} − 1` | Long-horizon direction | 0.101 | −0.439 TRENT → 0.802 SHRIRAMFIN |
| 8 | `trend_strength` | OLS slope of `log P` on time × 252 | Trend *consistency* | 0.169 | −0.057 ASIANPAINT → 0.439 ETERNAL |
| 9 | `autocorr_lag1` | `corr(r_t, r_{t−1})` | Persistence vs mean reversion | −0.006 | −0.105 JSWSTEEL → 0.101 DMART |
| 10 | `skewness` | Sample skew of returns | Distribution asymmetry | −0.117 | −2.563 INDUSINDBK → 1.555 MOTHERSON |
| 11 | `excess_kurtosis` | Fisher excess kurtosis | Fat tails | 5.382 | 0.903 TVSMOTOR → 36.72 TRENT |
| 12 | `volume_volatility` | `std(Δ log Volume)` | Participation instability | 0.578 | 0.434 SBIN → 0.705 MPHASIS |
| 13 | `return_volume_corr` | `corr(\|r\|, Δ log Volume)` | Event-driven vs steady | 0.332 | 0.190 ICICIBANK → 0.432 DIVISLAB |
| 14 | `extreme_move_frequency` | Share of days with `\|r\| > 2σ_stock` | Tail-event rate | 0.052 | 0.036 TRENT → 0.063 APOLLOHOSP |

Full per-feature diagnostics (missing, infinite, std, skew, near-constant flag,
outlier count, redundancy partners) live in `feature_quality_report.csv`.
**Result: 0 missing, 0 infinite, 0 near-constant across all features.**

### Design notes worth remembering

- **Beta, not correlation-to-NIFTY.** `beta = corr × (σ_stock / σ_market)` and
  σ_market is one constant across all 75 stocks. Keeping volatility, beta *and*
  correlation would put three columns of only two real dimensions into the distance
  metric K-Means uses.
- **Volume *rates of change*, never volume levels.** Average volume is a company-size
  proxy; clustering on it would rediscover market cap. `Δ log Volume` is size-invariant.
- **Two momentum windows, not three.** 63 ⊂ 126 ⊂ 252 are nested; the extremes carry
  the independent information, the middle is overlap.
- **Trend strength earns its place beside momentum.** Momentum compares two endpoints
  and cannot tell a steady climb from a violent round trip that finishes high; the
  log-price regression uses every day.
- **Extreme-move threshold is stock-relative** (2× that stock's own σ), unlike the 40%
  used in Stage 1b. That fixed cut detects broken data — as a behavioural feature it
  would read zero for every large-cap.
- **Sector is never an input.** Using it would make "did clusters rediscover sectors?"
  circular. It is an interpretation aid only, after clustering.

---

## 6. From 16 candidates to 14 — every cut, with its reason

### Cut before any code ran (by analysis)

| Dropped | Why |
|---|---|
| Mean daily return | Same statistic as annualised return at a different scale (ρ ≈ 1.0) |
| Price vs 200-day MA | A single last-day snapshot; noisier than window-aggregated features, and redundant with trend strength + momentum |
| Average volume | Size proxy, not behaviour — see above |
| Correlation to NIFTY | Algebraically implied by beta and volatility |
| 6-month momentum | The overlapping middle of nested 63/126/252-day windows |

### Cut after measurement (by the quality check)

`downside_volatility` — measured **+0.94** with `annualised_volatility` and **+0.91**
with `rolling_vol_dispersion`. It was the redundancy hub: removing that one feature
cleared both flagged pairs at once and left the highest surviving correlation at 0.83.

**Procedural rule:** the quality report is always computed on the **full** feature set
*before* any removal, so the evidence justifying each cut stays visible in
`feature_quality_report.csv`. Features are flagged and documented, never silently
dropped to make clustering metrics look better.

### Two deliberate judgment calls, recorded not hidden

- **`annualised_return` ↔ `trend_strength` = +0.904**, just above the 0.90 threshold —
  **both kept.** One measures how far the stock travelled, the other how consistently
  it got there. Recorded here rather than hidden by nudging the threshold to 0.91.
  Highest remaining correlation in the table; revisit after clustering.
- **`autocorr_lag1` is barely above noise** (spread 0.042 vs a sampling floor of
  1/√1139 ≈ 0.030) — **kept**, because its highest correlation with anything else is
  only 0.29, making it the most independent column in the table. First candidate to cut
  if it contributes nothing to the clusters.

---

## 7. Sanity checks that the numbers mean something

- TCS, Infosys and Wipro come out with near-identical rows — same business, same
  behaviour, and the fingerprint sees it without being told the sector.
- ITC has the lowest volatility (0.190) and CIPLA the lowest beta (0.472): defensives
  look defensive.
- ADANIENT tops volatility (0.490), beta (1.734) and vol-dispersion (0.229) — one
  stock, three features, one coherent story.
- TRENT has the worst drawdown (0.733) and the fattest tails (kurtosis 36.7) while
  having the *fewest* extreme days (0.036) — proof that the tail features are measuring
  something distinct from each other rather than restating volatility.

That recognisability is the cheapest evidence available that the features work.

---

## 8. Constants that govern all of the above

All in `src/config.py`; nothing hard-codes a ticker, date or window elsewhere.

| Constant | Value | Used for |
|---|---|---|
| `START` / `END` | 2022-01-01 → 2026-08-17 | Fixed window, absolute not rolling |
| `TRADING_DAYS_PER_YEAR` | 252 | Annualisation |
| `MIN_HISTORY_FRACTION` | 0.95 | Stage 1b history requirement |
| `EXTREME_RETURN_THRESHOLD` | 0.40 | Broken-data detector (caught VEDL) |
| `ROLLING_VOL_WINDOW` | 60 | Vol-of-vol, ~3 months |
| `MOMENTUM_3M_DAYS` / `_12M_DAYS` | 63 / 252 | Momentum lookbacks |
| `EXTREME_MOVE_Z` | 2.0 | Stock-relative tail-day threshold |
| `REDUNDANCY_CORR_THRESHOLD` | 0.90 | Flags a redundant feature pair |
| `EXTREME_FEATURE_Z` | 4.0 | Flags outlying feature values |
| `RANDOM_STATE` | 42 | Determinism everywhere downstream |

---

## 9. Known limitations of the processed data

Carry these into every claim made from it:

- **Single market, single ~4.6-year window.** Nothing here says the structure holds
  elsewhere or later.
- **In-sample by construction** — features, clusters and evaluation all use the same
  period.
- **Indian large-caps share a common market beta** (mean 0.97, tight around 1), which
  puts a structural ceiling on how much diversification is achievable at all.
- **Survivorship**: the universe is today's large-caps, so companies that fell out of
  the index are absent.
- **Yahoo's adjustment is trusted everywhere except where it was tested.** VEDL proves
  a bad case can exist; the 40% detector would miss a smaller unadjusted action.

**This describes historical structure. It does not forecast.**

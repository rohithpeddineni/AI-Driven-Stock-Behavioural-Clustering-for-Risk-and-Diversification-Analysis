# NOTES — what each stage did, in plain English

Written as the build progresses. One entry per stage.

---

## Stage 0 — Environment (2026-08-17)

`.venv` created with Python 3.13, dependencies pinned in `requirements.txt`,
`.gitignore` set to exclude `data/raw/` (regenerable) and `.venv/`.

**Snag worth knowing about.** The project folder is named `Financial Engineering​`
with an invisible zero-width space (U+200B) on the end. Two things broke because of
it, both fixed in `src/config.py`:

- `ensure_ascii_ca_bundle()` — the SSL certificate file lives inside the project
  folder, so its path inherited the invisible character, and the network library
  yfinance uses could not open it. Every download failed with `curl error 77`. The
  fix copies the certificate to a plain-ASCII path first.
- `configure_console()` — Windows terminals use the cp1252 encoding, which cannot
  print that character at all, so merely printing a file path crashed the program.
  The fix forces UTF-8 output.

Renaming the folder to remove the character is the real cure; these two functions
are workarounds that let the project run in the meantime.

---

## Stage 1 — Data download (2026-08-17)

`python -m src.data_loader` fetches 76 symbols (75 stocks + the NIFTY 50 index) from
Yahoo Finance and caches one CSV per symbol in `data/raw/`. Roughly 8 MB in total.

**Run it twice and the second run downloads nothing** — it reads the cache. This is
the point of the design: after the first fetch the whole project works offline, and
your results do not change because Yahoo changed something overnight.

Each file holds Open, High, Low, Close, **Adjusted Close**, Volume for 2022-01-01
through 2026-08-14 — 1,144 trading days.

**Why adjusted close matters.** A share split cuts the price mechanically: a 1-for-5
split takes a stock from ₹4,000 to ₹800 overnight. Using the raw close, that reads
as a −80% day, which would wreck the volatility, drawdown and kurtosis features and
push the stock into a nonsense cluster of its own. Adjusted close rewrites history so
splits and dividends do not appear as price shocks. Every return in this project is
computed from it.

`data/raw/_manifest.json` records what was fetched — row count, first and last date,
and a short hash per file — so a stale or half-finished cache is detectable rather
than silently wrong.

The loader **raises rather than skips** when a symbol fails. Symbols do go stale:
Zomato became `ETERNAL` and Tata Motors became `TMPV`, and Yahoo does not redirect.
A silent skip would quietly shrink the universe between runs.

---

## Stage 1b — Data quality check (2026-08-17)

`python -m src.data_quality` inspects the cache before any feature work and writes
`data/processed/data_quality_report.csv`. It asks: can this data be trusted?

**Results: clean.** No duplicate dates, no missing adjusted closes, no non-positive
prices, no ticker below the history threshold. Row counts run 1,140–1,144.

Two real problems were found and dealt with:

**1. Vedanta (`VEDL`) was silently corrupt — now excluded.** On 2026-04-30 the price
falls from ₹773.60 to ₹271.55, a −65% day. That is the Vedanta demerger, and Yahoo
did *not* adjust for it (`Close` equals `Adj Close` throughout). Left in, that single
fake day would have dominated the stock's volatility, max drawdown, skewness and
kurtosis — the fingerprint would describe a data error rather than a company. This is
the same reason `TMPV` was excluded earlier. **Lesson: adjusted close is usually
right, but not always. Check for it rather than assume it.**

**2. Four phantom trading days.** The dates 2026-01-15, 2026-05-01, 2026-05-28 and
2026-06-26 appear in every stock's file but *not* in the index's file. They carry zero
volume and simply repeat the previous day's price — they are market holidays that
Yahoo padded. Left in, they would inject fake zero-return days, quietly deflating
every volatility estimate and distorting autocorrelation.

They are not deleted from `data/raw/` — raw data stays exactly as downloaded. Stage 2
removes them by aligning every stock to the index's trading calendar.

**Universe after this stage: 75 stocks across 16 sectors, plus the benchmark.**

---

## Stage 2 — Cleaning (2026-08-17)

`python -m src.cleaning` merges the 76 cached files into one table and writes
`data/processed/cleaned_daily_data.csv`.

**Result: 85,500 rows = exactly 75 symbols × 1,140 trading days.** Every stock now
covers precisely the same days, which is what makes the fingerprints comparable —
a stock measured over even a slightly different period would get an unfair
advantage or handicap.

295 rows were removed: the padded market holidays found in Stage 1b. They are
dropped by *intersecting* each stock's dates with the index's calendar, rather than
by hard-coding the four dates — so the same code keeps working when the data is
refreshed and Yahoo pads different days. Nothing needed forward-filling.

**Long format, not wide.** The file is one row per (date, stock) with all of OHLCV,
rather than a grid of dates × tickers. The fingerprint needs Volume and OHLC as well
as Adjusted Close, and a wide layout would mean six separate wide tables that can
silently drift apart. Where wide *is* needed — correlation matrices, portfolio maths
— `adjusted_close_panel()` pivots it in memory. The rule: one stored copy of any
number, derived views computed on demand.

---

## Stage 3 — The behavioural fingerprint (2026-08-17)

`python -m src.features` turns 1,140 days of prices per stock into **one row of 14
numbers**, and writes `stock_fingerprints.csv`. This is the step the whole project
turns on: you cannot cluster 75 time series, but 75 rows of 14 numbers cluster
easily.

**Why 14 and not 16.** The two specifications disagreed on the feature list, so each
candidate was assessed for what behaviour it captures, whether it is scale-free, and
whether it duplicates another feature. Four were cut before any code ran:

- *Mean daily return* — the same statistic as annualised return, just a different
  scale. Correlation ≈ 1.0.
- *Price vs 200-day moving average* — a single last-day snapshot, far noisier than
  the window-wide statistics, and saying what trend strength already says.
- *Average volume* — **a company-size proxy, not a behaviour.** Clustering on it
  would rediscover market cap. Volume *volatility* and *return–volume correlation*
  are kept, because rates of change are size-invariant where levels are not.
- *Correlation to NIFTY* — because `beta = correlation × (σ_stock / σ_market)`, and
  σ_market is one constant for all 75 stocks. With volatility and beta both present,
  correlation adds no independent dimension: three columns, two real dimensions.
- *6-month momentum* — the overlapping middle of 63 ⊂ 126 ⊂ 252-day windows. The two
  extremes carry more independent information.

**Then the measured data cut a fifth.** The quality check computes the correlation
matrix on all 15 remaining features and flagged three redundant pairs. Downside
volatility was the hub — +0.94 with annualised volatility, +0.91 with rolling-vol
dispersion — so removing that one feature cleared both pairs at once, leaving the
survivors at 0.83. **The report is always computed on the full set first**, so the
evidence for the removal stays visible in `feature_quality_report.csv`.

Annualised return and trend strength sit at exactly +0.90 and were *kept*: one
measures how far the stock travelled, the other how consistently it got there.

**One feature is deliberately kept despite being weak.** Lag-1 autocorrelation has a
spread of 0.042 against a sampling-noise floor of 1/√1139 ≈ 0.030 — barely above
noise. But its highest correlation with any other feature is only 0.29, making it the
most *independent* column in the table, so it earns its place. Worth revisiting if it
proves to contribute nothing to the clusters.

**Extreme-move frequency uses a stock-relative threshold** (beyond 2× that stock's own
daily σ), not the 40% used by the Stage-1b data check. That 40% cut exists to catch
broken data — as a behavioural feature it would read zero for every large-cap.

**Sanity check.** TCS, Infosys and Wipro come out with near-identical rows; ITC has a
beta of 0.61 and Tata Steel 1.29. The numbers describe behaviour you can recognise,
which is the cheapest evidence that the features are working.

## Stages 4–6 — Clustering, similarity, and financial validation (2026-10-03)

Stage 4 standardizes the 75 × 14 fingerprint table, compares raw-feature and
retained-variance PCA K-Means solutions for `k = 2..8`, and persists the metric
evidence and assignments. The canonical result is PCA space with `k = 2`; its
clusters contain 69 and 6 stocks. The selected split is retained as measured,
not balanced by hand.

Stage 5 uses cosine similarity on the standardized fingerprints. The configured
ABB query returns SIEMENS, PERSISTENT, DIVISLAB, HEROMOTOCO, and TVSMOTOR as its
closest historical behavioural neighbours. Similarity is historical description
only, not a forecast or recommendation.

Stage 6 evaluates every raw/PCA label set for cluster-level return, volatility,
drawdown, correlation, deterministic centroid-based Portfolio A/B selection, and
the seeded random robustness experiment. For the canonical PCA `k = 2` result,
Portfolio B has annualised return 0.1750, volatility 0.2533, and maximum drawdown
0.2793; the median Portfolio A has annualised return 0.1372, volatility 0.2703,
and maximum drawdown 0.3720. The cross-cluster portfolio also has lower average
constituent correlation (0.3902 versus 0.5263). These are historical results, not
investment advice.

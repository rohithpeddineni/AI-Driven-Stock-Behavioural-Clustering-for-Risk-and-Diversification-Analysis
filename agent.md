# AGENT.md

Operating rules for agents working in this repository.
The project specification lives in [scope.md](scope.md) — **read it before writing
code.** Do not restate its contents here.

**Read [CONTEXT.md](CONTEXT.md) first.** It records current state and, where scope.md
and model_training_context.md disagree, which one won and why. Those conflicts are
resolved; do not re-litigate them. CONTEXT.md wins on any point of fact.

---

## Project in one line

Learn each stock's historical behaviour, build a behavioural fingerprint, cluster
stocks by fingerprint, and use simple Portfolio Theory to interpret the clusters.

## Non-negotiable rules

1. **AI/ML is the core; finance is the evaluation layer.** Keep roughly 75/25.
2. **Never make price prediction the objective.** Nothing in this system forecasts prices.
3. **Never claim clusters or similarity predict future returns.** Describe history only.
4. **Never frame output as investment advice or a recommendation.**
5. **Never execute trades or touch real money.** No broker APIs, no order placement.
6. **Never claim an anomaly proves fraud or manipulation.**
7. **Do not add out-of-scope finance models** (scope.md §10) to inflate financial content.
8. **Every feature needs a justification.** If you cannot say what behaviour it
   captures and why it matters, do not add it.
9. **Prefer the simple method that answers the question** over the impressive one.

## Repository layout

```text
data/
  raw/            # cached yfinance downloads, never edited by hand
  processed/      # cleaned panels + fingerprint feature table
src/
  config.py       # tickers, date range, paths, constants — no magic numbers elsewhere
  data_loader.py  # fetch + cache
  data_quality.py # raw-data diagnostics, run before cleaning
  cleaning.py     # calendar alignment, missing-data handling
  features.py     # returns + fingerprint + feature-quality checks
  clustering.py   # scaling, PCA, K-Means and alternatives, metrics
  similarity.py   # nearest-stock search
  finance.py      # cluster return/risk/correlation, Portfolio A vs B
  plots.py        # shared figures
app/
  Home.py         # Streamlit entry point
  pages/          # one file per dashboard page (scope.md §8)
notebooks/        # exploration only — never a dependency of src/ or app/
tests/
docs/
```

Create directories as the work reaches them; do not scaffold empty folders ahead of time.

## Stack

Python, pandas, NumPy, yfinance, scikit-learn, SciPy, Plotly/matplotlib, Streamlit.
UMAP or PyTorch only with an explicit justification.

## Build order

Work in this sequence; each stage must run end to end before the next begins.

1. Config and data loader with on-disk caching.
2. Cleaning and validation.
3. Returns and the fingerprint feature table.
4. Scaling, PCA, clustering, and the §6 metrics.
5. Similarity search.
6. Cluster financial analysis and the Portfolio A vs B comparison.
7. Streamlit pages, in the order listed in scope.md §8.
8. Documentation, report, presentation.

## Coding conventions

- **Determinism:** fix `random_state` everywhere it exists; identical inputs must
  give identical outputs.
- **Caching:** never re-download data that is already in `data/raw/`. Network calls
  belong only in `data_loader.py`.
- **Configuration:** tickers, dates, `k`, and window lengths come from `config.py`.
- **Data shape:** the fingerprint table is one row per ticker, ticker as the index,
  one column per feature.
- **Leakage:** compute every feature over the same fixed historical window; never
  let a stock's cluster label feed back into its own features.
- **Missing data:** drop tickers with insufficient history rather than imputing
  long gaps; log every drop with the reason.
- **Functions return data, not plots.** Plotting lives in `plots.py`; Streamlit
  files render, they do not compute.
- Type-hint public functions, add a one-line docstring saying what and why,
  and comment only where the *reason* is non-obvious.

## Definition of done for a module

- Runs from a clean state without manual steps.
- Deterministic across repeated runs.
- Failure modes (empty download, delisted ticker, short history) handled explicitly.
- Outputs written to the correct `data/` subdirectory.
- Shapes and assumptions verified, not assumed.

## Reporting results

State findings plainly, including negative ones. If cross-cluster portfolios do
**not** diversify better, that is a valid answer to RQ4 — report it with the
numbers rather than tuning the setup until the result flips. Always attach the
metric values behind any claim, and name the limitation (single market, fixed
window, historical data only) where it matters.

## Ask before

- Changing the ticker universe, date range, or benchmark.
- Adding a dependency outside the stack above.
- Introducing anything from scope.md §10 or §11.
- Any design choice that shifts the 75/25 AI-to-finance balance.

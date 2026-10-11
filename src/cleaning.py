"""Turn 76 separate cached files into one validated daily panel.

Stage 2. Reads data/raw/, writes data/processed/cleaned_daily_data.csv in long
format (Date | Symbol | Open | High | Low | Close | Adj Close | Volume).

Long format is the stored form because the fingerprint needs Volume and OHLC as
well as Adj Close - keeping it wide would mean six separate wide tables that can
drift apart. Wide panels are pivoted in memory where they are needed.

Run:  python -m src.cleaning
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data_loader import _csv_path

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _load(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(_csv_path(symbol), index_col="Date", parse_dates=True)
    return frame.sort_index()


def trading_calendar(benchmark: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Days the market was genuinely open.

    Yahoo pads market holidays into the individual stock files - those rows carry
    zero volume and simply repeat the previous close. Left in, they inject fake
    zero-return days that deflate every volatility estimate and distort
    autocorrelation. The index file does not contain them, so intersecting each
    stock's dates with the benchmark's removes them by construction.
    """
    calendar = benchmark.index
    for frame in stocks.values():
        calendar = calendar.intersection(frame.index)
    return calendar.sort_values()


def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align every symbol to one calendar and return (long panel, drop log)."""
    config.ensure_directories()

    benchmark = _load(config.BENCHMARK)
    stocks = {t: _load(config.yahoo_symbol(t)) for t in config.TICKERS}

    calendar = trading_calendar(benchmark, stocks)
    drops: list[dict] = []

    # Record what alignment removed, per symbol, before anything is discarded.
    for ticker, frame in stocks.items():
        removed = len(frame.index.difference(calendar))
        if removed:
            drops.append({
                "symbol": ticker,
                "reason": "date not in common trading calendar",
                "rows_removed": removed,
            })

    minimum_rows = int(len(calendar) * config.MIN_HISTORY_FRACTION)
    panels = []

    for ticker, frame in stocks.items():
        aligned = frame.reindex(calendar)

        # Gaps are interior and tiny (1-2 days); carrying the last known price
        # forward is honest, whereas interpolating invents prices that never traded.
        filled = int(aligned["Adj Close"].isna().sum())
        if filled:
            aligned[OHLCV_COLUMNS] = aligned[OHLCV_COLUMNS].ffill()
            drops.append({
                "symbol": ticker,
                "reason": "missing bar forward-filled",
                "rows_removed": filled,
            })

        usable = int(aligned["Adj Close"].notna().sum())
        if usable < minimum_rows:
            drops.append({
                "symbol": ticker,
                "reason": f"insufficient history: {usable} < {minimum_rows} rows",
                "rows_removed": usable,
            })
            continue

        aligned = aligned.dropna(subset=["Adj Close"])
        aligned = aligned.assign(Symbol=ticker)
        panels.append(aligned.reset_index()[["Date", "Symbol", *OHLCV_COLUMNS]])

    panel = pd.concat(panels, ignore_index=True).sort_values(["Symbol", "Date"])
    drop_log = pd.DataFrame(drops, columns=["symbol", "reason", "rows_removed"])
    return panel, drop_log


def benchmark_returns() -> pd.Series:
    """Benchmark daily returns on the shared calendar, for beta."""
    panel = pd.read_csv(config.CLEANED_DAILY_PATH, parse_dates=["Date"])
    calendar = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    benchmark = _load(config.BENCHMARK).reindex(calendar).ffill()
    return benchmark["Adj Close"].pct_change().dropna()


def adjusted_close_panel() -> pd.DataFrame:
    """Wide panel (dates x tickers) of adjusted close.

    Derived on demand rather than stored: a second on-disk copy of the same
    numbers is what silently drifts out of sync with the long table.
    """
    panel = pd.read_csv(config.CLEANED_DAILY_PATH, parse_dates=["Date"])
    return panel.pivot(index="Date", columns="Symbol", values="Adj Close")


if __name__ == "__main__":
    config.configure_console()
    panel, drop_log = clean()

    panel.to_csv(config.CLEANED_DAILY_PATH, index=False)
    drop_log.to_csv(config.DROP_LOG_PATH, index=False)

    symbols = panel["Symbol"].nunique()
    dates = panel["Date"].nunique()
    print(f"Cleaned panel: {len(panel):,} rows = {symbols} symbols x {dates} trading days")
    print(f"Date range:    {panel['Date'].min().date()} -> {panel['Date'].max().date()}")

    if drop_log.empty:
        print("\nNothing dropped or filled.")
    else:
        print(f"\nAdjustments ({len(drop_log)} entries):")
        for reason, group in drop_log.groupby("reason"):
            print(f"  {reason}: {len(group)} symbols, {group['rows_removed'].sum()} rows")

    wide = adjusted_close_panel()
    print(f"\nWide panel derived in memory: {wide.shape[0]} days x {wide.shape[1]} tickers")
    print(f"Written to {config.CLEANED_DAILY_PATH.name} and {config.DROP_LOG_PATH.name}")

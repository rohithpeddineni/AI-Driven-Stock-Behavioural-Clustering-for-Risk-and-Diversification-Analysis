"""Quality report on the raw cache, before any cleaning or feature work.

Answers one question: can this data be trusted? Every check below corresponds to
a failure mode that would silently corrupt the behavioural fingerprint rather
than raise an error.

Run:  python -m src.data_quality
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data_loader import _csv_path


def _benchmark_calendar() -> pd.DatetimeIndex:
    """Trading days according to the index - the reference calendar."""
    frame = pd.read_csv(_csv_path(config.BENCHMARK), index_col="Date", parse_dates=True)
    return frame.index


def check_symbol(symbol: str, calendar: pd.DatetimeIndex) -> dict:
    """Run every quality check on one cached symbol."""
    frame = pd.read_csv(_csv_path(symbol), index_col="Date", parse_dates=True)
    adjusted = frame["Adj Close"]
    returns = adjusted.pct_change().dropna()

    # A split shows up as a huge move only if the adjustment failed; a genuine
    # NSE large-cap rarely moves 40% in a day.
    extreme = returns[returns.abs() > config.EXTREME_RETURN_THRESHOLD]

    return {
        "symbol": symbol,
        "rows": len(frame),
        "first": frame.index.min().date(),
        "last": frame.index.max().date(),
        "coverage": round(len(frame) / len(calendar), 4),
        "missing_days": len(calendar.difference(frame.index)),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "null_adj_close": int(adjusted.isna().sum()),
        "nonpositive_price": int((adjusted <= 0).sum()),
        "zero_volume_days": int((frame["Volume"] == 0).sum()),
        "extreme_moves": len(extreme),
        "worst_move": round(returns.abs().max(), 4) if len(returns) else float("nan"),
        # Divergence between Close and Adj Close means splits/dividends were
        # actually applied. Exactly 0.0 would suggest no adjustment at all.
        "adj_divergence": round(
            float((frame["Close"] - adjusted).abs().max() / adjusted.max()), 4
        ),
    }


def build_report() -> pd.DataFrame:
    """Quality table, one row per cached symbol."""
    calendar = _benchmark_calendar()
    symbols = [config.BENCHMARK] + [config.yahoo_symbol(t) for t in config.TICKERS]
    return pd.DataFrame([check_symbol(s, calendar) for s in symbols]).set_index("symbol")


def summarise(report: pd.DataFrame) -> None:
    """Print the report and flag anything that needs a human decision."""
    calendar_days = report.loc[config.BENCHMARK, "rows"]
    print(f"Benchmark trading days: {calendar_days}")
    print(f"Symbols checked:        {len(report)}")
    print(f"Window:                 {config.START} -> {config.END}\n")

    print("Row-count distribution:")
    print(report["rows"].describe()[["min", "50%", "max"]].to_string(), "\n")

    checks = {
        "Below minimum history": report[report["coverage"] < config.MIN_HISTORY_FRACTION],
        "Duplicate dates": report[report["duplicate_dates"] > 0],
        "Null adjusted close": report[report["null_adj_close"] > 0],
        "Non-positive price": report[report["nonpositive_price"] > 0],
        "Extreme daily moves": report[report["extreme_moves"] > 0],
        "Zero-volume days": report[report["zero_volume_days"] > 0],
        "No split/dividend adjustment": report[report["adj_divergence"] == 0.0],
    }

    for label, flagged in checks.items():
        if flagged.empty:
            print(f"  OK    {label}: none")
        else:
            print(f"  FLAG  {label}: {len(flagged)}")
            columns = ["rows", "coverage", "missing_days", "null_adj_close",
                       "extreme_moves", "worst_move", "zero_volume_days"]
            print(flagged[columns].to_string(), "\n")


if __name__ == "__main__":
    config.configure_console()
    report = build_report()
    config.ensure_directories()
    output = config.DATA_QUALITY_PATH
    report.to_csv(output)
    summarise(report)
    print(f"\nFull report written to {output}")

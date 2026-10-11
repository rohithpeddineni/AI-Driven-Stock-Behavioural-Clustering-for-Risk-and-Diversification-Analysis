"""Fetch daily OHLCV from Yahoo Finance and cache it on disk.

The only module in this project allowed to touch the network. Everything
downstream reads from data/raw/, so results stay reproducible whether or not
Yahoo is reachable.

Run directly to populate the cache:  python -m src.data_loader
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

config.ensure_ascii_ca_bundle()  # must precede the yfinance import

import yfinance as yf  # noqa: E402

EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
REQUEST_PAUSE_SECONDS = 0.3


def _csv_path(symbol: str) -> Path:
    """Cache location for a Yahoo symbol ('^NSEI' -> data/raw/_NSEI.csv)."""
    safe = symbol.replace("^", "_")
    return config.RAW_DIR / f"{safe}.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _download(symbol: str) -> pd.DataFrame:
    """Download one symbol, or raise with a reason.

    auto_adjust=False keeps raw OHLCV *and* 'Adj Close' side by side. Returns
    are computed from 'Adj Close' (scope.md section 3) so that splits and
    dividends do not appear as price shocks.
    """
    frame = yf.download(
        symbol,
        start=config.START,
        end=config.END,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if frame is None or frame.empty:
        raise ValueError(
            f"{symbol}: no data returned. The symbol may have been renamed or "
            f"delisted - Yahoo does not redirect (e.g. ZOMATO -> ETERNAL)."
        )

    # yfinance returns a (field, symbol) MultiIndex even for one ticker.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    frame = frame[EXPECTED_COLUMNS]
    frame.index.name = "Date"
    return frame


def fetch_symbol(symbol: str, *, force: bool = False) -> pd.DataFrame:
    """Return one symbol's OHLCV, downloading only if it is not already cached."""
    path = _csv_path(symbol)
    if path.exists() and not force:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    frame = _download(symbol)
    frame.to_csv(path)
    time.sleep(REQUEST_PAUSE_SECONDS)  # be polite to an undocumented endpoint
    return frame


def fetch_all(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch the benchmark and every ticker, then write a manifest.

    Raises if any symbol fails: a silent skip would shrink the universe between
    runs and quietly break determinism.
    """
    config.ensure_directories()

    symbols = [config.BENCHMARK] + [config.yahoo_symbol(t) for t in config.TICKERS]
    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    downloaded = 0

    for position, symbol in enumerate(symbols, start=1):
        cached = _csv_path(symbol).exists() and not force
        try:
            frames[symbol] = fetch_symbol(symbol, force=force)
        except Exception as error:  # noqa: BLE001 - reported, then re-raised below
            failures.append(f"{symbol}: {error}")
            print(f"[{position:3d}/{len(symbols)}] {symbol:16s} FAILED")
            continue

        if not cached:
            downloaded += 1
        status = "cached" if cached else "downloaded"
        print(f"[{position:3d}/{len(symbols)}] {symbol:16s} {len(frames[symbol]):5d} rows  {status}")

    if failures:
        raise RuntimeError(
            "Some symbols could not be fetched:\n  " + "\n  ".join(failures)
        )

    _write_manifest(frames)
    print(f"\n{len(frames)} symbols cached in {config.RAW_DIR} ({downloaded} newly downloaded)")
    return frames


def _write_manifest(frames: dict[str, pd.DataFrame]) -> None:
    """Record what was fetched, so a stale or partial cache is detectable."""
    entries = {}
    for symbol, frame in sorted(frames.items()):
        path = _csv_path(symbol)
        entries[symbol] = {
            "rows": int(len(frame)),
            "first_date": str(frame.index.min().date()),
            "last_date": str(frame.index.max().date()),
            "sha256": _sha256(path),
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": config.START,
        "end": config.END,
        "benchmark": config.BENCHMARK,
        "symbol_count": len(entries),
        "symbols": entries,
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    config.configure_console()
    fetch_all()

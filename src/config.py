"""Single source of truth for tickers, dates, paths and constants.

No magic numbers live anywhere else in the codebase (see agent.md).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = RAW_DIR / "_manifest.json"

# Output contract. Names follow model_training_context.md section 15 so the
# deliverables match the spec exactly; the two extra reports are additive.
CLEANED_DAILY_PATH = PROCESSED_DIR / "cleaned_daily_data.csv"
FINGERPRINTS_PATH = PROCESSED_DIR / "stock_fingerprints.csv"
CLUSTER_ASSIGNMENTS_PATH = PROCESSED_DIR / "cluster_assignments.csv"
CLUSTER_SUMMARY_PATH = PROCESSED_DIR / "cluster_summary.csv"
SIMILARITY_RESULTS_PATH = PROCESSED_DIR / "similarity_results.csv"
PORTFOLIO_RESULTS_PATH = PROCESSED_DIR / "portfolio_results.csv"
FINANCE_SELECTION_PATH = PROCESSED_DIR / "finance_selection.csv"
FINANCE_CORRELATION_PATH = PROCESSED_DIR / "finance_correlation_comparison.csv"
DATA_QUALITY_PATH = PROCESSED_DIR / "data_quality_report.csv"
FEATURE_QUALITY_PATH = PROCESSED_DIR / "feature_quality_report.csv"
CLUSTER_METRICS_PATH = PROCESSED_DIR / "cluster_metrics.csv"
CLUSTER_PCA_PATH = PROCESSED_DIR / "cluster_pca_coordinates.csv"
CLUSTER_PCA_RETAINED_PATH = PROCESSED_DIR / "cluster_pca_retained.csv"
CLUSTER_PCA_SUMMARY_PATH = PROCESSED_DIR / "cluster_pca_summary.csv"
CLUSTER_SCALED_PATH = PROCESSED_DIR / "cluster_scaled_fingerprints.csv"
DROP_LOG_PATH = PROCESSED_DIR / "cleaning_drop_log.csv"

# --------------------------------------------------------------------------
# Study window
# --------------------------------------------------------------------------
# Fixed absolute dates, never "today minus N years": identical inputs must give
# identical outputs. END is exclusive, so the last bar is 2026-08-14.

START = "2022-01-01"
END = "2026-08-17"

# A ticker must have at least this fraction of the benchmark's trading days to
# be retained. Short histories distort volatility and drawdown features.
MIN_HISTORY_FRACTION = 0.95

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# Benchmark for beta and market comparison.
BENCHMARK = "^NSEI"

# Sector labels are metadata for INTERPRETING clusters only. They are never fed
# into the clustering itself - otherwise "did the model rediscover sectors?"
# becomes a circular question.
SECTORS: dict[str, str] = {
    # Information Technology
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT",
    # Private Banks
    "HDFCBANK": "Private Bank", "ICICIBANK": "Private Bank",
    "KOTAKBANK": "Private Bank", "AXISBANK": "Private Bank",
    "INDUSINDBK": "Private Bank",
    # Public Sector Banks
    "SBIN": "PSU Bank", "PNB": "PSU Bank",
    "BANKBARODA": "PSU Bank", "CANBK": "PSU Bank",
    # NBFC and Insurance
    "BAJFINANCE": "NBFC & Insurance", "BAJAJFINSV": "NBFC & Insurance",
    "SBILIFE": "NBFC & Insurance", "HDFCLIFE": "NBFC & Insurance",
    "SHRIRAMFIN": "NBFC & Insurance",
    # Pharma and Healthcare
    "SUNPHARMA": "Pharma & Health", "CIPLA": "Pharma & Health",
    "DRREDDY": "Pharma & Health", "DIVISLAB": "Pharma & Health",
    "LUPIN": "Pharma & Health", "AUROPHARMA": "Pharma & Health",
    "APOLLOHOSP": "Pharma & Health",
    # Auto and Ancillaries
    "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto", "MOTHERSON": "Auto",
    # Metals and Mining
    # VEDL excluded: the April 2026 Vedanta demerger cut the price from 773.60 to
    # 271.55 with no adjustment applied by Yahoo (Close == Adj Close), which reads
    # as a -65% single-day return and would corrupt every risk feature.
    "TATASTEEL": "Metals & Mining", "JSWSTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining", "COALINDIA": "Metals & Mining",
    # Energy, Oil and Gas
    "RELIANCE": "Energy & Oil/Gas", "ONGC": "Energy & Oil/Gas",
    "BPCL": "Energy & Oil/Gas", "IOC": "Energy & Oil/Gas",
    "GAIL": "Energy & Oil/Gas",
    # Utilities
    "NTPC": "Utilities", "POWERGRID": "Utilities",
    # FMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG",
    # Cement and Materials
    "ULTRACEMCO": "Cement & Materials", "GRASIM": "Cement & Materials",
    "AMBUJACEM": "Cement & Materials", "ASIANPAINT": "Cement & Materials",
    "PIDILITIND": "Cement & Materials",
    # Capital Goods
    "LT": "Capital Goods", "SIEMENS": "Capital Goods",
    "ABB": "Capital Goods", "HAVELLS": "Capital Goods",
    # Consumer and Retail
    "TITAN": "Consumer & Retail", "TRENT": "Consumer & Retail",
    "DMART": "Consumer & Retail", "JUBLFOOD": "Consumer & Retail",
    "PAGEIND": "Consumer & Retail", "ETERNAL": "Consumer & Retail",
    # Infrastructure and Logistics
    "ADANIENT": "Infra & Logistics", "ADANIPORTS": "Infra & Logistics",
    # Single-name sectors
    "BHARTIARTL": "Telecom",
    "DLF": "Realty",
}

# Yahoo Finance suffix for NSE-listed equities.
NSE_SUFFIX = ".NS"

TICKERS: list[str] = sorted(SECTORS)
SIMILARITY_QUERY_TICKER = TICKERS[0]
SIMILARITY_TOP_N = 5

# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------

RANDOM_STATE = 42
TRADING_DAYS_PER_YEAR = 252

# --- Fingerprint feature windows -------------------------------------------
# 6-month momentum is deliberately absent: 63 < 126 < 252 are nested windows on
# the same price path, and the middle one is the most predictable from the other
# two. Keeping the horizon extremes carries more independent information.
ROLLING_VOL_WINDOW = 60        # volatility-of-volatility, ~3 months
MOMENTUM_3M_DAYS = 63
MOMENTUM_12M_DAYS = 252

# Extreme-move frequency counts days beyond this many of a stock's OWN standard
# deviations. It must stay stock-relative: reusing the data-quality threshold
# below would score ~0 for every large-cap and carry no information at all.
EXTREME_MOVE_Z = 2.0

# --- Data-quality thresholds (NOT features) --------------------------------
# A single-day move beyond this in an NSE large-cap is almost always a data
# error or an unadjusted corporate action, not a real price change.
EXTREME_RETURN_THRESHOLD = 0.40

# --- Feature-quality checks ------------------------------------------------
REDUNDANCY_CORR_THRESHOLD = 0.90   # |corr| above this flags a redundant pair

# Features removed AFTER seeing the measured correlation matrix, never before.
# The quality report is always computed on the full set first, so the evidence
# for each removal stays visible in feature_quality_report.csv.
REDUNDANT_FEATURES_DROPPED = {
    "downside_volatility":
        "corr +0.94 with annualised_volatility and +0.91 with "
        "rolling_vol_dispersion: it was the redundancy hub, and removing it "
        "clears both flagged pairs while leaving the survivors at 0.83.",
}
NEAR_CONSTANT_STD_THRESHOLD = 1e-8
EXTREME_FEATURE_Z = 4.0            # flag outlying feature values for review

# --- Clustering ------------------------------------------------------------
K_RANGE = range(2, 9)              # candidate cluster counts to evaluate
PCA_VARIANCE_TARGET = 0.85         # components retained for the PCA comparison
PCA_PLOT_COMPONENTS = 2            # always fit separately, for display only
PCA_MATERIAL_SILHOUETTE_GAIN = 0.02
PCA_MATERIAL_DB_REDUCTION = 0.05   # relative reduction required alongside silhouette
K_COMPLEXITY_PENALTY = 0.03
KMEANS_N_INIT = 20
MIN_CLUSTER_SIZE = 2

# --- Portfolio experiment (RQ4) --------------------------------------------
# Holdings are min(MAX_HOLDINGS, k): Portfolio B can never hold more stocks than
# there are clusters. Clusters too small to fill Portfolio A are skipped and
# logged rather than shrinking every portfolio to match the smallest cluster.
PORTFOLIO_MAX_HOLDINGS = 5
PORTFOLIO_RANDOM_DRAWS = 1000      # per arm, for the distributional check


def yahoo_symbol(ticker: str) -> str:
    """Map a bare NSE ticker to its Yahoo Finance symbol."""
    return f"{ticker}{NSE_SUFFIX}"


def ensure_ascii_ca_bundle() -> None:
    """Point the SSL stack at a CA bundle whose path is pure ASCII.

    This project directory contains a zero-width space (U+200B), and the curl
    backend yfinance uses cannot open a certificate path containing non-ASCII
    characters - every download fails with curl error 77. Copying the bundle to
    an ASCII path sidesteps it. Must run before yfinance issues any request.
    """
    import certifi

    source = certifi.where()
    if source.isascii():
        return

    destination = Path(tempfile.gettempdir()) / "fineng_cacert.pem"
    if not str(destination).isascii():
        raise RuntimeError(
            f"Temp directory path is also non-ASCII: {destination}. "
            "Rename the project directory to remove the invisible character."
        )
    if not destination.exists():
        shutil.copyfile(source, destination)

    os.environ["SSL_CERT_FILE"] = str(destination)
    os.environ["CURL_CA_BUNDLE"] = str(destination)


def configure_console() -> None:
    """Force UTF-8 stdout so printing project paths cannot crash the program.

    Windows consoles default to cp1252, which cannot encode the zero-width space
    in this directory's name - printing any absolute path raises UnicodeEncodeError.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def ensure_directories() -> None:
    """Create the data directories as the pipeline reaches them."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    configure_console()
    sectors = sorted(set(SECTORS.values()))
    print(f"{len(TICKERS)} tickers across {len(sectors)} sectors")
    print(f"Window: {START} -> {END} (END exclusive)")
    print(f"Benchmark: {BENCHMARK}")
    for sector in sectors:
        members = sorted(t for t, s in SECTORS.items() if s == sector)
        print(f"  {sector:20s} ({len(members):2d})  {' '.join(members)}")

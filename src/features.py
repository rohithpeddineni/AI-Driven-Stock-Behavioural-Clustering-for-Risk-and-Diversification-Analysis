"""Collapse each stock's price history into one row: its behavioural fingerprint.

Stage 3, and the heart of the project. 1,140 days of prices become 15 numbers
describing how a stock has moved - not what the company is, and not what it will
do next.

Every feature is scale-free (a ratio, a correlation, a normalised statistic) so a
Rs 200 stock and a Rs 4,000 stock are compared on behaviour rather than size, and
every feature is measured over the identical window so no stock benefits from a
luckier period.

Run:  python -m src.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.cleaning import adjusted_close_panel, benchmark_returns

ANNUALISE = np.sqrt(config.TRADING_DAYS_PER_YEAR)


# --------------------------------------------------------------------------
# Individual features. Each takes a stock's series and returns one number.
# --------------------------------------------------------------------------

def annualised_return(returns: pd.Series) -> float:
    """Typical performance level over the window."""
    return float(returns.mean() * config.TRADING_DAYS_PER_YEAR)


def annualised_volatility(returns: pd.Series) -> float:
    """Overall magnitude of risk."""
    return float(returns.std(ddof=1) * ANNUALISE)


def downside_volatility(returns: pd.Series) -> float:
    """Risk measured on losing days only.

    Not redundant with total volatility: two stocks can share a volatility figure
    while one earns it through upside jumps and the other through drawdowns.
    """
    losses = returns[returns < 0]
    return float(losses.std(ddof=1) * ANNUALISE)


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough fall, as a positive fraction (0.35 = a 35% fall).

    Path-dependent, so it captures something volatility cannot: a calm stock that
    fell once and never recovered scores badly here and unremarkably on vol.
    """
    running_peak = prices.cummax()
    return float((1 - prices / running_peak).max())


def rolling_volatility_dispersion(returns: pd.Series) -> float:
    """How much the stock's own risk level moves around.

    Measures risk *stability*, not risk *level*: a steadily volatile stock and one
    that alternates between calm and turbulent can share an annualised volatility
    but behave nothing alike.
    """
    rolling = returns.rolling(config.ROLLING_VOL_WINDOW).std(ddof=1) * ANNUALISE
    return float(rolling.dropna().std(ddof=1))


def beta(returns: pd.Series, market: pd.Series) -> float:
    """Sensitivity to the market: cov(stock, market) / var(market).

    Correlation-to-market is deliberately NOT a separate feature. Since
    beta = corr * (sigma_stock / sigma_market) and sigma_market is one constant
    across the whole universe, keeping volatility, beta and correlation would put
    three columns of only two independent dimensions into the distance metric.
    """
    aligned = pd.concat([returns, market], axis=1, join="inner").dropna()
    stock_returns, market_returns = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return float(stock_returns.cov(market_returns) / market_returns.var())


def momentum(prices: pd.Series, days: int) -> float:
    """Point-to-point return over the trailing window."""
    if len(prices) <= days:
        return float("nan")
    return float(prices.iloc[-1] / prices.iloc[-1 - days] - 1)


def trend_strength(prices: pd.Series) -> float:
    """Annualised slope of log price against time.

    Momentum compares two endpoints and cannot tell a steady climb from a violent
    round trip that happens to finish high. This regression uses every day, so it
    measures how *consistent* the direction was.
    """
    log_prices = np.log(prices.to_numpy(dtype=float))
    days = np.arange(len(log_prices), dtype=float)
    slope = np.polyfit(days, log_prices, 1)[0]
    return float(slope * config.TRADING_DAYS_PER_YEAR)


def lag1_autocorrelation(returns: pd.Series) -> float:
    """Do moves persist day to day, or reverse?

    Weakest signal in the set: for liquid large-caps the true value sits near zero
    and the noise floor here is 1/sqrt(1140) ~ 0.03. Flagged for review by the
    redundancy check rather than assumed informative.
    """
    return float(returns.autocorr(lag=1))


def skewness(returns: pd.Series) -> float:
    """Asymmetry: a few large gains, or a few large losses?"""
    return float(returns.skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """Fat tails - how often extreme days happen relative to a normal curve."""
    return float(returns.kurtosis())


def volume_volatility(volume: pd.Series) -> float:
    """Instability of participation, from log changes rather than levels.

    Average volume is deliberately excluded from the fingerprint: it is a company
    size proxy, so clustering on it would rediscover market cap rather than
    behaviour. Log *changes* are size-invariant.
    """
    positive = volume.replace(0, np.nan)
    return float(np.log(positive).diff().dropna().std(ddof=1))


def return_volume_correlation(returns: pd.Series, volume: pd.Series) -> float:
    """Do big price moves arrive with volume surges?

    Separates event-driven names from those that drift on steady participation.
    """
    positive = volume.replace(0, np.nan)
    log_volume_change = np.log(positive).diff()
    frame = pd.concat([returns.abs(), log_volume_change], axis=1, join="inner").dropna()
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def extreme_move_frequency(returns: pd.Series) -> float:
    """Share of days moving beyond the stock's OWN volatility band.

    The threshold is stock-relative on purpose. A fixed cut such as the 40% used
    by the data-quality check would read zero for every large-cap and carry no
    information - that constant detects broken data, not behaviour.
    """
    threshold = config.EXTREME_MOVE_Z * returns.std(ddof=1)
    return float((returns.abs() > threshold).mean())


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_fingerprints() -> pd.DataFrame:
    """One row per stock, one column per feature."""
    panel = pd.read_csv(config.CLEANED_DAILY_PATH, parse_dates=["Date"])
    prices = adjusted_close_panel()
    volumes = panel.pivot(index="Date", columns="Symbol", values="Volume")
    market = benchmark_returns()

    rows = {}
    for ticker in prices.columns:
        stock_prices = prices[ticker].dropna()
        returns = stock_prices.pct_change().dropna()

        rows[ticker] = {
            "annualised_return": annualised_return(returns),
            "annualised_volatility": annualised_volatility(returns),
            "downside_volatility": downside_volatility(returns),
            "max_drawdown": max_drawdown(stock_prices),
            "rolling_vol_dispersion": rolling_volatility_dispersion(returns),
            "beta": beta(returns, market),
            "momentum_3m": momentum(stock_prices, config.MOMENTUM_3M_DAYS),
            "momentum_12m": momentum(stock_prices, config.MOMENTUM_12M_DAYS),
            "trend_strength": trend_strength(stock_prices),
            "autocorr_lag1": lag1_autocorrelation(returns),
            "skewness": skewness(returns),
            "excess_kurtosis": excess_kurtosis(returns),
            "volume_volatility": volume_volatility(volumes[ticker]),
            "return_volume_corr": return_volume_correlation(returns, volumes[ticker]),
            "extreme_move_frequency": extreme_move_frequency(returns),
        }

    fingerprints = pd.DataFrame.from_dict(rows, orient="index")
    fingerprints.index.name = "Symbol"
    return fingerprints.sort_index()


# --------------------------------------------------------------------------
# Feature quality - runs before clustering, flags but never silently drops
# --------------------------------------------------------------------------

def feature_quality(fingerprints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature diagnostics and the pairwise correlation matrix.

    Deliberately reports rather than repairs: dropping features to improve
    clustering metrics is how you talk yourself into a result.
    """
    correlations = fingerprints.corr()

    redundant = {}
    for feature in fingerprints.columns:
        others = correlations[feature].drop(feature).abs()
        strong = others[others > config.REDUNDANCY_CORR_THRESHOLD]
        redundant[feature] = ", ".join(f"{n} ({correlations.loc[feature, n]:+.2f})"
                                       for n in strong.index)

    standardised = (fingerprints - fingerprints.mean()) / fingerprints.std(ddof=1)

    report = pd.DataFrame({
        "missing": fingerprints.isna().sum(),
        "infinite": np.isinf(fingerprints.to_numpy()).sum(axis=0),
        "mean": fingerprints.mean(),
        "std": fingerprints.std(ddof=1),
        "min": fingerprints.min(),
        "max": fingerprints.max(),
        "skew": fingerprints.skew(),
        "near_constant": fingerprints.std(ddof=1) < config.NEAR_CONSTANT_STD_THRESHOLD,
        "extreme_values": (standardised.abs() > config.EXTREME_FEATURE_Z).sum(),
        "redundant_with": pd.Series(redundant),
    })
    report.index.name = "feature"
    return report, correlations


def final_fingerprints() -> pd.DataFrame:
    """The fingerprint table clustering actually consumes.

    Redundant features are removed here, after the quality report has already
    recorded the correlations that justified removing them.
    """
    return build_fingerprints().drop(columns=list(config.REDUNDANT_FEATURES_DROPPED))


if __name__ == "__main__":
    config.configure_console()
    fingerprints = build_fingerprints()
    report, correlations = feature_quality(fingerprints)
    report.to_csv(config.FEATURE_QUALITY_PATH)

    kept = fingerprints.drop(columns=list(config.REDUNDANT_FEATURES_DROPPED))
    kept.to_csv(config.FINGERPRINTS_PATH)

    print(f"Fingerprints: {fingerprints.shape[0]} stocks x {fingerprints.shape[1]} features\n")
    print(report[["mean", "std", "min", "max", "missing", "extreme_values"]].round(4).to_string())

    flagged = report[report["redundant_with"] != ""]
    print(f"\nRedundant pairs (|corr| > {config.REDUNDANCY_CORR_THRESHOLD}):")
    if flagged.empty:
        print("  none")
    else:
        for feature, row in flagged.iterrows():
            print(f"  {feature}: {row['redundant_with']}")

    problems = report[report["near_constant"] | (report["missing"] > 0) | (report["infinite"] > 0)]
    print("\nNear-constant / missing / infinite:", "none" if problems.empty else list(problems.index))

    print("\nRemoved as redundant:")
    for feature, reason in config.REDUNDANT_FEATURES_DROPPED.items():
        print(f"  {feature} - {reason}")

    remaining = kept.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(remaining, 0.0)
    print(f"\nHighest remaining |corr| among the {kept.shape[1]} kept features: "
          f"{remaining.max():.3f}")
    print(f"Written to {config.FINGERPRINTS_PATH.name} and {config.FEATURE_QUALITY_PATH.name}")

"""
Signal construction: cross-sectional 12-1 month momentum.

Reference: Jegadeesh & Titman (1993) — "Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency."

The 12-1 convention: use the 11-month cumulative return ending one month
before the signal date. Skipping the most recent month avoids contamination
from the short-term reversal effect documented in the literature.
"""
import numpy as np
import pandas as pd


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily adjusted prices to month-end and compute simple returns.

    Using month-end resampling ('ME') ensures consistent rebalancing dates
    and avoids look-ahead from intra-month prices.
    """
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.pct_change()


def momentum_signal(
    monthly_returns: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """
    Standard 12-1 cross-sectional momentum signal.

    At month t, signal_i = cumulative return of stock i from t-12 to t-1.
    The skip parameter excludes the most recent month to avoid short-term reversal.

    Args:
        monthly_returns: DataFrame of monthly returns (dates x tickers)
        lookback: Total lookback window in months (default 12)
        skip: Months to skip at the recent end (default 1)

    Returns:
        DataFrame of signal values with same shape as monthly_returns.
        NaN where insufficient history exists.
    """
    window = lookback - skip
    # Rolling cumulative return over the window
    cumulative = (1 + monthly_returns).rolling(window).apply(np.prod, raw=True) - 1
    # Shift by skip so signal at t uses data only through t-skip
    return cumulative.shift(skip)

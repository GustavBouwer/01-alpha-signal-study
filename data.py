"""
Data fetching: S&P 500 constituents and historical price data via yfinance.

Survivorship bias note: we use the *current* S&P 500 constituent list.
Companies that were delisted or removed from the index are excluded,
which slightly inflates backtested returns. Acknowledged in limitations.
"""
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BATCH_SIZE = 50


def get_sp500_tickers() -> list[str]:
    """
    Scrape current S&P 500 constituents from Wikipedia.
    Replaces '.' with '-' to match yfinance ticker format (e.g. BRK.B → BRK-B).
    Uses a browser User-Agent to avoid Wikipedia's 403 block on automated requests.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    response = requests.get(SP500_WIKI_URL, headers=headers, timeout=10)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("Retrieved %d S&P 500 tickers", len(tickers))
    return tickers


def get_price_data(
    tickers: list[str],
    start: str,
    end: str,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers via yfinance.

    Drops any ticker missing more than 20% of observations to ensure
    signal computation has sufficient history.

    Returns:
        DataFrame indexed by date, columns are ticker symbols.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"prices_{start}_{end}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached prices from %s", cache_path)
        return pd.read_parquet(cache_path)

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"Downloading {len(tickers)} tickers in {len(batches)} batches...")

    batch_frames = []
    for batch in tqdm(batches, desc="Fetching prices", unit="batch"):
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        batch_frames.append(close)

    prices = pd.concat(batch_frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Drop tickers with excessive missing data
    min_obs = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=min_obs)
    print(f"Done. Retained {prices.shape[1]} tickers after quality filter.")

    if cache:
        prices.to_parquet(cache_path)
        logger.info("Cached to %s", cache_path)

    return prices


def get_benchmark(start: str, end: str, cache: bool = True) -> pd.Series:
    """Download SPY adjusted close prices as market benchmark."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"spy_{start}_{end}.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path).squeeze()

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    close = spy["Close"]
    # Newer yfinance returns a MultiIndex DataFrame even for a single ticker
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    prices = close.rename("SPY")
    pd.DataFrame(prices).to_parquet(cache_path)
    return prices

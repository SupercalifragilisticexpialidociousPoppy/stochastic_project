"""Market inputs: S0 (spot), sigma (annualised volatility) and r (risk-free rate).

Prices are downloaded fresh from Yahoo Finance when possible (and cached to
data/<ticker>.csv); if the download fails validation, the cached CSV is used.
"""
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["RELIANCE.NS", "SUNPHARMA.NS", "ADANIENT.NS"]
HISTORY_PERIOD = "1y"
TRADING_DAYS = 252
TBILL_YIELD = 0.0629                            # RBI 364-day T-bill, annual
RISK_FREE_RATE = math.log(1 + TBILL_YIELD)      # continuously compounded
DATA_DIR = Path(__file__).parent / "data"
MIN_ROWS = 200
MAX_STALENESS_DAYS = 7


def download_prices(ticker, period=HISTORY_PERIOD):
    """Adjusted close series from Yahoo Finance; empty Series on any failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return pd.Series(dtype=float)
        prices = hist["Close"].copy()
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
        prices.name = "Close"
        return prices
    except Exception as exc:  # yfinance often fails quietly; never crash here
        print(f"  [{ticker}] download error: {type(exc).__name__}: {exc}")
        return pd.Series(dtype=float)


def validate_prices(prices):
    """Return (ok, reason)."""
    if prices is None or len(prices) == 0:
        return False, "empty response"
    prices = prices.dropna()
    if len(prices) < MIN_ROWS:
        return False, f"only {len(prices)} rows"
    if (prices <= 0).any():
        return False, "non-positive prices"
    age = (datetime.now() - prices.index[-1].to_pydatetime()).days
    if age > MAX_STALENESS_DAYS:
        return False, f"last row is {age} days old"
    return True, "ok"


def cache_path(ticker):
    return DATA_DIR / f"{ticker}.csv"


def save_cache(ticker, prices):
    DATA_DIR.mkdir(exist_ok=True)
    tmp = cache_path(ticker).with_suffix(".csv.tmp")
    prices.to_csv(tmp, header=True)
    os.replace(tmp, cache_path(ticker))


def load_cache(ticker):
    path = cache_path(ticker)
    if not path.exists():
        return pd.Series(dtype=float)
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")


def get_prices(ticker):
    """Return (prices, status). Fresh data overwrites the cache; else use the cache."""
    fresh = download_prices(ticker)
    ok, reason = validate_prices(fresh)
    if ok:
        fresh = fresh.dropna()
        save_cache(ticker, fresh)
        return fresh, "FRESH (downloaded, cache updated)"
    cached = load_cache(ticker)
    if not cached.empty:
        return cached, (f"CACHED (download failed: {reason}; "
                        f"using data up to {cached.index[-1].date()})")
    raise RuntimeError(f"{ticker}: download failed ({reason}) and no cache at {cache_path(ticker)}")


def estimate_volatility(prices):
    """Annualised sigma from daily log returns."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return float(log_ret.std(ddof=1) * math.sqrt(TRADING_DAYS))


def current_price(prices):
    return float(prices.iloc[-1])


def get_inputs(ticker):
    prices, status = get_prices(ticker)
    return {
        "ticker": ticker,
        "S0": current_price(prices),
        "sigma": estimate_volatility(prices),
        "r": RISK_FREE_RATE,
        "as_of": str(prices.index[-1].date()),
        "n_obs": int(len(prices)),
        "status": status,
    }


def main():
    print(f"{'Ticker':<13}{'As of':<12}{'S0':>10}{'sigma':>9}{'n_obs':>7}  Status")
    for t in TICKERS:
        try:
            i = get_inputs(t)
            print(f"{t:<13}{i['as_of']:<12}{i['S0']:>10.2f}{i['sigma']:>9.4f}{i['n_obs']:>7}  {i['status']}")
        except RuntimeError as exc:
            print(f"{t:<13}ERROR: {exc}")
    print(f"Risk-free rate r = {RISK_FREE_RATE:.5f} (continuous, from {TBILL_YIELD:.2%} RBI 364-day T-bill)")


if __name__ == "__main__":
    main()

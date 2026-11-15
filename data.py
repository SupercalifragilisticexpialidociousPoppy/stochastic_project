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
TBILL_YIELD = 0.0629                            # default: RBI 364-day T-bill, annual
RISK_FREE_RATE = math.log(1 + TBILL_YIELD)      # continuously compounded
MAX_RATE_PERCENT = 30.0                         # sanity cap for --rate
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


def rate_from_percent(percent):
    """Continuously compounded r from an annual yield quoted in percent (6.29 -> 0.06100)."""
    if not (0 < percent <= MAX_RATE_PERCENT):
        raise ValueError(f"rate must be in (0, {MAX_RATE_PERCENT:g}] percent per year, got {percent}")
    return math.log(1 + percent / 100)


def add_rate_argument(parser):
    parser.add_argument("--rate", type=float, default=None, metavar="PERCENT",
                        help="annual risk-free yield in percent, e.g. 6.29 "
                             f"(default: {TBILL_YIELD:.2%} RBI 364-day T-bill)")


def resolve_rate(percent=None):
    """(r, label) for an optional user-supplied yield in percent."""
    if percent is None:
        return RISK_FREE_RATE, f"default {TBILL_YIELD:.2%} RBI 364-day T-bill"
    return rate_from_percent(percent), f"user-supplied {percent:g}% via --rate"


def get_inputs(ticker, rate_percent=None):
    prices, status = get_prices(ticker)
    r, rate_source = resolve_rate(rate_percent)
    return {
        "ticker": ticker,
        "S0": current_price(prices),
        "sigma": estimate_volatility(prices),
        "r": r,
        "rate_source": rate_source,
        "as_of": str(prices.index[-1].date()),
        "n_obs": int(len(prices)),
        "status": status,
    }


def main(rate_percent=None):
    print(f"{'Ticker':<13}{'As of':<12}{'S0':>10}{'sigma':>9}{'n_obs':>7}  Status")
    for t in TICKERS:
        try:
            i = get_inputs(t, rate_percent)
            print(f"{t:<13}{i['as_of']:<12}{i['S0']:>10.2f}{i['sigma']:>9.4f}{i['n_obs']:>7}  {i['status']}")
        except RuntimeError as exc:
            print(f"{t:<13}ERROR: {exc}")
    r, source = resolve_rate(rate_percent)
    print(f"Risk-free rate r = {r:.5f} (continuous; {source})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Show market inputs for each ticker.")
    add_rate_argument(ap)
    args = ap.parse_args()
    try:
        main(args.rate)
    except ValueError as exc:
        ap.error(str(exc))

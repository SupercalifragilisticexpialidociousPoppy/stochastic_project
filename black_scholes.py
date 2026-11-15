"""Closed-form Black-Scholes prices for European options.

Conventions: T in years, sigma annualised (decimal), r continuously compounded.
K or sigma may be NumPy arrays; the result is then an array.
"""
import sys

import numpy as np
from scipy.stats import norm


def _check_inputs(S0, K, T, sigma):
    if np.any(np.asarray(S0) <= 0):
        raise ValueError("S0 must be > 0")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("K must be > 0")
    if np.any(np.asarray(T) <= 0):
        raise ValueError("T must be > 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be > 0")


def d1_d2(S0, K, T, r, sigma):
    """Return (d1, d2)."""
    _check_inputs(S0, K, T, sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call(S0, K, T, r, sigma):
    """European call: C = S0 N(d1) - K e^{-rT} N(d2)."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0, K, T, r, sigma):
    """European put via put-call parity: P = C - S0 + K e^{-rT}."""
    return bs_call(S0, K, T, r, sigma) - S0 + K * np.exp(-r * T)


def bs_price(S0, K, T, r, sigma, option="call"):
    if option == "call":
        return bs_call(S0, K, T, r, sigma)
    if option == "put":
        return bs_put(S0, K, T, r, sigma)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def _self_check():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    failures = 0

    def check(name, ok):
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failures += not ok

    c, p = bs_call(S0, K, T, r, sigma), bs_put(S0, K, T, r, sigma)
    check(f"call = 10.4506 (got {c:.4f})", abs(c - 10.4506) < 1e-4)
    check(f"put = 5.5735 (got {p:.4f})", abs(p - 5.5735) < 1e-4)
    check("put-call parity", abs(c - p - (S0 - K * np.exp(-r * T))) < 1e-10)
    check("deep ITM call ~ S0 - K e^{-rT}",
          abs(bs_call(S0, 1.0, T, r, sigma) - (S0 - 1.0 * np.exp(-r * T))) < 1e-6)
    check("deep OTM call ~ 0", bs_call(S0, 1000.0, T, r, sigma) < 1e-6)
    check("call increases with sigma",
          bs_call(S0, K, T, r, 0.3) > bs_call(S0, K, T, r, 0.2))
    check("call increases with T",
          bs_call(S0, K, 2.0, r, sigma) > bs_call(S0, K, 1.0, r, sigma))
    arr = bs_call(S0, np.array([90.0, 100.0, 110.0]), T, r, sigma)
    check("array of strikes works and decreases", arr.shape == (3,) and arr[0] > arr[1] > arr[2])
    return failures


if __name__ == "__main__":
    sys.exit(1 if _self_check() else 0)

"""Cox-Ross-Rubinstein binomial tree for European options.

Conventions: T in years, sigma annualised (decimal), r continuously compounded.
This module is independent of the other project files.
"""
import sys

import numpy as np


def crr_params(T, r, sigma, n):
    """Return (u, d, p, dt) for an n-step CRR tree."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability p={p:.4f} not in (0,1); increase n")
    return u, d, p, dt


def terminal_prices(S0, sigma, dt, n):
    """The n+1 stock prices at maturity; index j = number of up moves.

    u^j d^(n-j) = u^(2j-n) because d = 1/u. Written via exp to avoid overflow.
    """
    j = np.arange(n + 1)
    return S0 * np.exp(sigma * np.sqrt(dt) * (2 * j - n))


def payoff(S_T, K, option):
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    if option == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def binomial_price(S0, K, T, r, sigma, n, option="call"):
    """Option price today by vectorised backward induction (O(n^2) time, O(n) memory)."""
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0 or n < 1:
        raise ValueError("S0, K, T, sigma must be > 0 and n >= 1")
    u, d, p, dt = crr_params(T, r, sigma, n)
    disc = np.exp(-r * dt)
    V = payoff(terminal_prices(S0, sigma, dt, n), K, option)
    for _ in range(n):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
    return float(V[0])


def binomial_tree_full(S0, K, T, r, sigma, n, option="call"):
    """Full lattices for small n (display only).

    Returns (stock, value), both (n+1, n+1); entry [i, j] is time step i with
    j up moves (j <= i); other entries are NaN.
    """
    u, d, p, dt = crr_params(T, r, sigma, n)
    disc = np.exp(-r * dt)
    stock = np.full((n + 1, n + 1), np.nan)
    value = np.full((n + 1, n + 1), np.nan)
    for i in range(n + 1):
        for j in range(i + 1):
            stock[i, j] = S0 * u ** (2 * j - i)
    value[n, :] = payoff(stock[n, :], K, option)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            value[i, j] = disc * (p * value[i + 1, j + 1] + (1 - p) * value[i + 1, j])
    return stock, value


def _self_check():
    # Imported here only so the module itself stays independent of black_scholes.
    from black_scholes import bs_call, bs_put

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    failures = 0

    def check(name, ok):
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failures += not ok

    c, p = binomial_price(S0, K, T, r, sigma, 1000, "call"), binomial_price(S0, K, T, r, sigma, 1000, "put")
    check(f"n=1000 call within 0.01 of BS (got {c:.4f}, BS {bs_call(S0, K, T, r, sigma):.4f})",
          abs(c - bs_call(S0, K, T, r, sigma)) < 0.01)
    check(f"n=1000 put within 0.01 of BS (got {p:.4f}, BS {bs_put(S0, K, T, r, sigma):.4f})",
          abs(p - bs_put(S0, K, T, r, sigma)) < 0.01)
    for n in (7, 50, 333):
        cn, pn = binomial_price(S0, K, T, r, sigma, n, "call"), binomial_price(S0, K, T, r, sigma, n, "put")
        check(f"put-call parity at n={n}", abs(cn - pn - (S0 - K * np.exp(-r * T))) < 1e-10)
    u, d, pr, dt = crr_params(T, r, sigma, 1)
    by_hand = np.exp(-r * dt) * (pr * max(S0 * u - K, 0) + (1 - pr) * max(S0 * d - K, 0))
    check("n=1 matches one-step formula", abs(binomial_price(S0, K, T, r, sigma, 1) - by_hand) < 1e-12)
    _, val = binomial_tree_full(S0, K, T, r, sigma, 3)
    check("full tree root equals binomial_price at n=3",
          abs(val[0, 0] - binomial_price(S0, K, T, r, sigma, 3)) < 1e-12)
    return failures


if __name__ == "__main__":
    sys.exit(1 if _self_check() else 0)

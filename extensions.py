"""Extra experiments E9-E11, run on the SAME inputs as the report (outputs/run_info.json).

  E9   averaging the prices for n and n+1 (cancels the odd/even oscillation)
  E10  running time for large n (does the slope reach 2?)
  E11  American put: early exercise on the same CRR tree

Usage:  python extensions.py        (run compare.py first; nothing is downloaded here)
Figures and tables go to outputs/ with the same helpers as compare.py.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binomial import binomial_price, binomial_price_american
from black_scholes import bs_call, bs_put
from compare import (C_ALT, C_ALT2, C_BIN, C_BS, PRIMARY, ROOT, abs_err, pair, save_fig, save_table,
                     setup_dirs, time_call)

N_AVG = [10, 50, 100, 200, 500, 1000]
N_AVG_DENSE = list(range(10, 1001, 10))
MONEYNESS_AVG = [1.0, 1.05, 1.1]
N_BIG = [1000, 2000, 4000, 8000, 16000]
N_AMER = 1000
MONEYNESS_AMER = [0.9, 1.0, 1.1, 1.2]
T_AMER = [0.25, 1.0]


def load_inputs():
    """Inputs of the run the report is based on (written by compare.py)."""
    info = json.loads((ROOT / "outputs" / "run_info.json").read_text())
    return info["stocks"][info.get("primary", PRIMARY)]


def averaged(S0, K, T, r, sigma, n, option="call"):
    return 0.5 * (binomial_price(S0, K, T, r, sigma, n, option) + binomial_price(S0, K, T, r, sigma, n + 1, option))


def run_e9_averaging(inp):
    rows = []
    for m in MONEYNESS_AVG:
        S0, K, T, r, sigma = pair(inp, m=m)
        bs = float(bs_call(S0, K, T, r, sigma))
        for n in N_AVG:
            plain = abs(binomial_price(S0, K, T, r, sigma, n, "call") - bs)
            plain_2n = abs(binomial_price(S0, K, T, r, sigma, 2 * n, "call") - bs)
            avg = abs(averaged(S0, K, T, r, sigma, n) - bs)
            rows.append({"K/S0": m, "n": n, "Plain err (n)": plain, "Plain err (2n)": plain_2n,
                         "Averaged err (n, n+1)": avg, "Plain / averaged": plain / avg})
    save_table(pd.DataFrame(rows), "e9_averaging",
               f"Call error of one tree (n and 2n steps) and of the average of n and n+1 steps, "
               f"{inp['ticker']}, T={pair(inp)[2]}.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, (1.0, 1.1)):
        S0, K, T, r, sigma = pair(inp, m=m)
        bs = float(bs_call(S0, K, T, r, sigma))
        ns = np.array(N_AVG_DENSE)
        plain = abs_err([binomial_price(S0, K, T, r, sigma, n, "call") for n in ns], bs)
        avg = abs_err([averaged(S0, K, T, r, sigma, n) for n in ns], bs)
        ax.loglog(ns, np.maximum(plain, 1e-8), "o", ms=3, color=C_BIN, label="one tree, n steps")
        ax.loglog(ns, np.maximum(avg, 1e-8), "s", ms=3, color=C_ALT, label="average of n and n+1")
        ax.set_xlabel("Number of steps n")
        ax.set_title(f"Call, K = {m:g} S0")
        ax.legend()
    axes[0].set_ylabel("|Binomial - Black-Scholes| (INR)")
    save_fig(fig, "e9_averaging")


def run_e10_timing_large_n(inp):
    S0, K, T, r, sigma = pair(inp)
    times = [time_call(lambda n=n: binomial_price(S0, K, T, r, sigma, n, "call"), repeats=3) for n in N_BIG]
    big = [i for i, n in enumerate(N_BIG) if n >= 4000]
    slope = np.polyfit(np.log(np.array(N_BIG)[big]), np.log(np.array(times)[big]), 1)[0]
    df = pd.DataFrame({"n": N_BIG, "Seconds per price": times})
    save_table(df, "e10_timing_large_n", "Median running time per binomial price for large n.")
    save_table(pd.DataFrame([{"Fitted slope (n >= 4000)": slope}]), "e10_slope",
               "Log-log slope of binomial running time against n, large n.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_BIG, times, "o-", color=C_BIN, label=f"Binomial (slope {slope:.2f} for n>=4000)")
    ref = times[-1] * (np.array(N_BIG) / N_BIG[-1]) ** 2
    ax.loglog(N_BIG, ref, "k--", lw=1, label=r"$O(n^2)$ reference")
    ax.set_xlabel("Number of steps n")
    ax.set_ylabel("Seconds per price")
    ax.set_title("Running time, large n")
    ax.legend()
    save_fig(fig, "e10_timing_large_n")


def run_e11_american(inp):
    rows = []
    for T in T_AMER:
        for m in MONEYNESS_AMER:
            S0, K, _, r, sigma = pair(inp, m=m)
            eu_bs = float(bs_put(S0, K, T, r, sigma))
            eu = binomial_price(S0, K, T, r, sigma, N_AMER, "put")
            am = binomial_price_american(S0, K, T, r, sigma, N_AMER, "put")
            rows.append({"T (years)": T, "K/S0": m, "BS European put": eu_bs, "Binomial European put": eu,
                         "Binomial American put": am, "Early-exercise premium": am - eu,
                         "Premium (pct)": 100 * (am - eu) / eu, "Intrinsic K - S0": max(K - S0, 0.0)})
    save_table(pd.DataFrame(rows), "e11_american",
               f"American vs European put on {inp['ticker']} (n={N_AMER}); the premium uses the tree "
               f"European price at the same n.")
    S0, K, T, r, sigma = pair(inp)
    conv = pd.DataFrame([{"n": n, "American put": binomial_price_american(S0, K, T, r, sigma, n, "put")}
                         for n in (100, 250, 500, 1000, 2000)])
    save_table(conv, "e11_american_convergence", f"ATM American put on {inp['ticker']} (T={T}) against n.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ms = np.linspace(0.8, 1.2, 17)
    for T, color_eu, color_am in ((0.25, C_BS, C_BIN), (1.0, C_ALT2, C_ALT)):
        eu = [binomial_price(S0, m * S0, T, r, sigma, N_AMER, "put") for m in ms]
        am = [binomial_price_american(S0, m * S0, T, r, sigma, N_AMER, "put") for m in ms]
        ax.plot(ms, eu, "--", color=color_eu, label=f"European, T={T}")
        ax.plot(ms, am, "-", color=color_am, label=f"American, T={T}")
    ax.plot(ms, np.maximum(ms * S0 - S0, 0), "k:", lw=1, label="Intrinsic K - S0")
    ax.set_xlabel("Moneyness K/S0")
    ax.set_ylabel("Put price (INR)")
    ax.set_title(f"Early exercise, {inp['ticker']} (n={N_AMER})")
    ax.legend()
    save_fig(fig, "e11_american")


def main():
    setup_dirs()
    inp = load_inputs()
    for name, fn in (("E9", run_e9_averaging), ("E10", run_e10_timing_large_n), ("E11", run_e11_american)):
        fn(inp)
        print(f"done {name}")


if __name__ == "__main__":
    main()

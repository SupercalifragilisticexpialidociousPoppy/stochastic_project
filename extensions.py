"""Extra experiments E9-E10, run on the SAME inputs as the report (outputs/run_info.json).

  E9   averaging the prices for n and n+1 (cancels the odd/even oscillation)
  E10  running time for large n (does the slope reach 2?)

Usage:  python extensions.py        (run compare.py first; nothing is downloaded here)
Figures and tables go to outputs/ with the same helpers as compare.py.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binomial import binomial_price
from black_scholes import bs_call
from compare import (C_ALT, C_BIN, PRIMARY, ROOT, abs_err, pair, save_fig, save_table,
                     setup_dirs, time_call)

N_AVG = [10, 50, 100, 200, 500, 1000]
N_AVG_DENSE = list(range(10, 1001, 10))
MONEYNESS_AVG = [1.0, 1.05, 1.1]
N_BIG = [1000, 2000, 4000, 8000, 16000]


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



def main():
    setup_dirs()
    inp = load_inputs()
    for name, fn in (("E9", run_e9_averaging), ("E10", run_e10_timing_large_n)):
        fn(inp)
        print(f"done {name}")


if __name__ == "__main__":
    main()

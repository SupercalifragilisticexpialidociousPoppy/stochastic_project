"""Run every experiment (E1-E8), save figures and tables to outputs/.

Usage:  python compare.py
Figures: outputs/figures/<name>.pdf and .png   Tables: outputs/tables/<name>.csv and .tex
"""
import json
import time
import timeit
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from binomial import binomial_price, binomial_tree_full
from black_scholes import bs_call, bs_put

# ----------------------------------------------------------------- configuration
PRIMARY = "RELIANCE.NS"
T_DEFAULT = 0.25                       # 3 months
MONEYNESS_DEFAULT = 1.0                # K = m * S0 (1.0 = at the money)
N_DEFAULT = 500
N_LIST = [5, 10, 20, 50, 100, 200, 500, 1000]
N_DENSE = list(range(5, 401))
MONEYNESS_LIST = np.linspace(0.8, 1.2, 9)
SIGMA_LIST = np.linspace(0.10, 0.60, 11)
T_LIST = [1 / 12, 0.25, 0.5, 1.0]
TIMING_REPEATS = 7

ROOT = Path(__file__).parent
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"

# Okabe-Ito colour-blind-safe palette
C_BIN, C_BS, C_ALT, C_ALT2 = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------- helpers
def setup_dirs():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"{name}.pdf")
    fig.savefig(OUT_FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def save_table(df, name, caption):
    df.to_csv(OUT_TAB / f"{name}.csv", index=False)
    latex = df.to_latex(index=False, float_format="%.6g", caption=caption, label=f"tab:{name}")
    # Centre the table and shrink it to the text width only if it is too wide
    # (needs the `adjustbox` LaTeX package).
    latex = latex.replace("\\begin{table}\n", "\\begin{table}\n\\centering\n\\small\n", 1)
    latex = latex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}\n", "\\end{tabular}\n\\end{adjustbox}\n", 1)
    (OUT_TAB / f"{name}.tex").write_text(latex)


def time_call(fn, repeats=TIMING_REPEATS):
    """Median seconds per call; batch size chosen so one batch takes >= 0.2 s."""
    number, _ = timeit.Timer(fn).autorange()
    runs = timeit.repeat(fn, number=number, repeat=repeats)
    return float(np.median(runs) / number)


def abs_err(a, b):
    return np.abs(np.asarray(a) - np.asarray(b))


def rel_err(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))


def pair(inp, m=MONEYNESS_DEFAULT, T=T_DEFAULT):
    """(S0, K, T, r, sigma) for a stock's inputs at moneyness m."""
    return inp["S0"], m * inp["S0"], T, inp["r"], inp["sigma"]


def write_run_info(inputs):
    info = {"run_at": datetime.now().isoformat(timespec="seconds"),
            "primary": PRIMARY, "T_default": T_DEFAULT, "n_default": N_DEFAULT,
            "stocks": inputs}
    (ROOT / "outputs" / "run_info.json").write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------ experiments
def run_e1_convergence(inp):
    S0, K, T, r, sigma = pair(inp)
    bsc, bsp = float(bs_call(S0, K, T, r, sigma)), float(bs_put(S0, K, T, r, sigma))
    rows = []
    for n in N_LIST:
        bc = binomial_price(S0, K, T, r, sigma, n, "call")
        bp = binomial_price(S0, K, T, r, sigma, n, "put")
        rows.append({"n": n, "Binomial call": bc, "BS call": bsc,
                     "Call abs err": abs(bc - bsc), "Call rel err": abs(bc - bsc) / bsc,
                     "Binomial put": bp, "BS put": bsp,
                     "Put abs err": abs(bp - bsp), "Put rel err": abs(bp - bsp) / bsp})
    df = pd.DataFrame(rows)
    save_table(df, "e1_convergence",
               f"Binomial vs Black--Scholes, ATM options on {inp['ticker']} (T={T}, sigma={sigma:.4f}).")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind, bs in zip(axes, ("call", "put"), (bsc, bsp)):
        ax.plot(df["n"], df[f"Binomial {kind}"], "o-", color=C_BIN, label="Binomial")
        ax.axhline(bs, color=C_BS, ls="--", label=f"Black-Scholes ({bs:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("Number of steps n")
        ax.set_ylabel("Price (INR)")
        ax.set_title(f"ATM European {kind}, {inp['ticker']}")
        ax.legend()
    save_fig(fig, "e1_convergence")


def run_e2_error_decay(inp):
    S0, K, T, r, sigma = pair(inp)
    bsc = float(bs_call(S0, K, T, r, sigma))
    errs = np.array([abs(binomial_price(S0, K, T, r, sigma, n, "call") - bsc) for n in N_DENSE])
    ns = np.array(N_DENSE)
    rows = []
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # Right panel: the SIGNED error alternates with the parity of n (the odd/even oscillation),
    # which the absolute-error panel can hide when the size of the error is nearly constant.
    zoom = (ns >= 50) & (ns <= 100)
    signed = np.array([binomial_price(S0, K, T, r, sigma, n, "call") - bsc for n in ns[zoom]])
    ax2.plot(ns[zoom], signed, "o-", ms=3, color=C_BIN)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Number of steps n")
    ax2.set_ylabel("Binomial - Black-Scholes (INR)")
    ax2.set_title("Signed error alternates with odd/even n")
    for label, mask, color, marker in (("odd n", ns % 2 == 1, C_BIN, "o"),
                                       ("even n", ns % 2 == 0, C_ALT, "s")):
        x, y = np.log(ns[mask]), np.log(np.maximum(errs[mask], 1e-16))
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({"Subset": label, "Fitted slope": slope})
        ax.loglog(ns[mask], errs[mask], marker, ms=3, color=color, alpha=0.7,
                  label=f"{label} (slope {slope:.2f})")
    ref = errs[ns == 100][0] * (100 / ns)
    ax.loglog(ns, ref, "k--", lw=1, label=r"$O(1/n)$ reference")
    ax.set_xlabel("Number of steps n")
    ax.set_ylabel("|Binomial - Black-Scholes| (INR)")
    ax.set_title(f"Error decay, ATM call on {inp['ticker']}")
    ax.legend()
    save_fig(fig, "e2_error_loglog")
    save_table(pd.DataFrame(rows), "e2_slope",
               "Fitted log-log slope of the absolute error against n (expected about $-1$).")


def run_e3_strike(inp):
    rows = []
    for m in MONEYNESS_LIST:
        S0, K, T, r, sigma = pair(inp, m=m)
        bc, bp = float(bs_call(S0, K, T, r, sigma)), float(bs_put(S0, K, T, r, sigma))
        nc = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "call")
        np_ = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "put")
        rows.append({"K/S0": m, "K": K, "BS call": bc, "Binomial call": nc,
                     "Call abs err": abs(nc - bc), "BS put": bp, "Binomial put": np_,
                     "Put abs err": abs(np_ - bp)})
    df = pd.DataFrame(rows)
    save_table(df, "e3_strike", f"Effect of strike (n={N_DEFAULT}, T={T_DEFAULT}) on {inp['ticker']}.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["K/S0"], df["BS call"], "-", color=C_BS, label="BS call")
    axes[0].plot(df["K/S0"], df["Binomial call"], "o", color=C_BIN, label="Binomial call")
    axes[0].plot(df["K/S0"], df["BS put"], "-", color=C_ALT, label="BS put")
    axes[0].plot(df["K/S0"], df["Binomial put"], "s", color=C_ALT2, label="Binomial put")
    axes[0].set_xlabel("Moneyness K/S0"); axes[0].set_ylabel("Price (INR)"); axes[0].legend()
    axes[1].plot(df["K/S0"], df["Call abs err"], "o-", color=C_BIN, label="Call")
    axes[1].plot(df["K/S0"], df["Put abs err"], "s-", color=C_ALT2, label="Put")
    axes[1].set_xlabel("Moneyness K/S0"); axes[1].set_ylabel("Absolute error (INR)"); axes[1].legend()
    save_fig(fig, "e3_strike")


def run_e4_volatility(inp):
    rows = []
    for s in SIGMA_LIST:
        S0, K, T, r, _ = pair(inp)
        bc, bp = float(bs_call(S0, K, T, r, s)), float(bs_put(S0, K, T, r, s))
        nc = binomial_price(S0, K, T, r, s, N_DEFAULT, "call")
        np_ = binomial_price(S0, K, T, r, s, N_DEFAULT, "put")
        rows.append({"sigma": s, "BS call": bc, "Binomial call": nc, "Call abs err": abs(nc - bc),
                     "BS put": bp, "Binomial put": np_, "Put abs err": abs(np_ - bp)})
    df = pd.DataFrame(rows)
    save_table(df, "e4_volatility", f"Effect of volatility (ATM, n={N_DEFAULT}, T={T_DEFAULT}) on {inp['ticker']}.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["sigma"], df["BS call"], "-", color=C_BS, label="BS call")
    axes[0].plot(df["sigma"], df["Binomial call"], "o", color=C_BIN, label="Binomial call")
    axes[0].set_xlabel("Volatility sigma"); axes[0].set_ylabel("Price (INR)"); axes[0].legend()
    axes[1].plot(df["sigma"], df["Call abs err"], "o-", color=C_BIN, label="Call")
    axes[1].plot(df["sigma"], df["Put abs err"], "s-", color=C_ALT2, label="Put")
    axes[1].set_xlabel("Volatility sigma"); axes[1].set_ylabel("Absolute error (INR)"); axes[1].legend()
    save_fig(fig, "e4_volatility")


def run_e5_maturity(inp):
    rows = []
    for T in T_LIST:
        S0, K, _, r, sigma = pair(inp)
        bc, bp = float(bs_call(S0, K, T, r, sigma)), float(bs_put(S0, K, T, r, sigma))
        nc = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "call")
        np_ = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "put")
        rows.append({"T (years)": T, "BS call": bc, "Binomial call": nc, "Call abs err": abs(nc - bc),
                     "BS put": bp, "Binomial put": np_, "Put abs err": abs(np_ - bp)})
    df = pd.DataFrame(rows)
    save_table(df, "e5_maturity", f"Effect of maturity (ATM, n={N_DEFAULT}) on {inp['ticker']}.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["T (years)"], df["BS call"], "-", color=C_BS, label="BS call")
    axes[0].plot(df["T (years)"], df["Binomial call"], "o", color=C_BIN, label="Binomial call")
    axes[0].plot(df["T (years)"], df["BS put"], "-", color=C_ALT, label="BS put")
    axes[0].plot(df["T (years)"], df["Binomial put"], "s", color=C_ALT2, label="Binomial put")
    axes[0].set_xlabel("Maturity T (years)"); axes[0].set_ylabel("Price (INR)"); axes[0].legend()
    axes[1].plot(df["T (years)"], df["Call abs err"], "o-", color=C_BIN, label="Call")
    axes[1].plot(df["T (years)"], df["Put abs err"], "s-", color=C_ALT2, label="Put")
    axes[1].set_xlabel("Maturity T (years)"); axes[1].set_ylabel("Absolute error (INR)"); axes[1].legend()
    save_fig(fig, "e5_maturity")


def run_e6_stocks(all_inputs):
    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_ALT, C_ALT2, "#E69F00", C_BIN]  # same stock colours as the price-history figure
    for color, (ticker, inp) in zip(colors, all_inputs.items()):
        S0, K, T, r, sigma = pair(inp)
        bc, bp = float(bs_call(S0, K, T, r, sigma)), float(bs_put(S0, K, T, r, sigma))
        nc = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "call")
        np_ = binomial_price(S0, K, T, r, sigma, N_DEFAULT, "put")
        rows.append({"Stock": ticker, "S0": S0, "sigma": sigma,
                     "BS call": bc, "Binomial call": nc,
                     "Call abs err": abs(nc - bc), "Call rel err": abs(nc - bc) / bc,
                     "BS put": bp, "Binomial put": np_,
                     "Put abs err": abs(np_ - bp), "Put rel err": abs(np_ - bp) / bp})
        rel = [abs(binomial_price(S0, K, T, r, sigma, n, "call") - bc) / bc for n in N_LIST]
        ax.loglog(N_LIST, rel, "o-", color=color, label=f"{ticker} (sigma={sigma:.2f})")
    save_table(pd.DataFrame(rows), "e6_stocks",
               f"Same moneyness (ATM), T={T_DEFAULT}, n={N_DEFAULT}, for each stock's own S0 and sigma.")
    ax.set_xlabel("Number of steps n")
    ax.set_ylabel("Relative error of ATM call")
    ax.set_title("Convergence by stock")
    ax.legend()
    save_fig(fig, "e6_stocks_convergence")


def run_e7_timing(inp):
    S0, K, T, r, sigma = pair(inp)
    t_bs = time_call(lambda: bs_call(S0, K, T, r, sigma))
    rows = [{"Method": "Black-Scholes", "n": 0, "Seconds per price": t_bs}]
    times = []
    for n in N_LIST:
        t = time_call(lambda n=n: binomial_price(S0, K, T, r, sigma, n, "call"))
        times.append(t)
        rows.append({"Method": "Binomial", "n": n, "Seconds per price": t})
    df = pd.DataFrame(rows)
    save_table(df, "e7_timing", "Median running time per option price.")
    big = [i for i, n in enumerate(N_LIST) if n >= 100]
    slope = np.polyfit(np.log(np.array(N_LIST)[big]), np.log(np.array(times)[big]), 1)[0]
    save_table(pd.DataFrame([{"Fitted slope (n >= 100)": slope}]), "e7_slope",
               "Log-log slope of binomial running time against n.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_LIST, times, "o-", color=C_BIN, label=f"Binomial (slope {slope:.2f} for n>=100)")
    ax.axhline(t_bs, color=C_BS, ls="--", label=f"Black-Scholes ({t_bs * 1e6:.0f} us)")
    ax.set_xlabel("Number of steps n")
    ax.set_ylabel("Seconds per price")
    ax.set_title("Running time")
    ax.legend()
    save_fig(fig, "e7_timing")


def run_e8_parity(inp):
    S0, K, T, r, sigma = pair(inp)
    target = S0 - K * np.exp(-r * T)
    bs_res = float(bs_call(S0, K, T, r, sigma) - bs_put(S0, K, T, r, sigma) - target)
    bn_res = (binomial_price(S0, K, T, r, sigma, N_DEFAULT, "call")
              - binomial_price(S0, K, T, r, sigma, N_DEFAULT, "put") - target)
    save_table(pd.DataFrame([{"Model": "Black-Scholes", "Residual": bs_res},
                             {"Model": f"Binomial (n={N_DEFAULT})", "Residual": bn_res}]),
               "e8_parity", "Put-call parity residual $C-P-(S_0-Ke^{-rT})$.")


def run_example_tree(inp):
    S0, K, T, r, sigma = pair(inp)
    stock, value = binomial_tree_full(S0, K, T, r, sigma, 3, "call")
    rows = [{"Step": i, "Up moves": j, "Stock price": stock[i, j], "Call value": value[i, j]}
            for i in range(4) for j in range(i + 1)]
    save_table(pd.DataFrame(rows), "example_tree_n3",
               f"Three-step CRR tree for an ATM call on {inp['ticker']}.")


def run_price_history():
    """Price history of every cached stock, rebased to 100 (for the slides)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, t in zip((C_ALT, C_ALT2, "#E69F00", C_BIN), data.TICKERS):
        prices = data.load_cache(t)
        if prices.empty:
            continue
        sigma = data.estimate_volatility(prices)
        ax.plot(prices.index, 100 * prices / prices.iloc[0], color=color, lw=1.6,
                label=f"{t.replace('.NS', '')} (sigma = {sigma:.2f})")
    ax.set_ylabel("Price, rebased to 100")
    ax.set_title("One year of daily closes")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    save_fig(fig, "price_history")


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    setup_dirs()
    inputs = {}
    for t in data.TICKERS:
        try:
            inputs[t] = data.get_inputs(t)
            print(f"{t}: {inputs[t]['status']}")
        except RuntimeError as exc:
            print(f"WARNING: skipping {t}: {exc}")
    if PRIMARY not in inputs:
        raise SystemExit(f"No data for primary ticker {PRIMARY}; connect once to build the cache.")
    write_run_info(inputs)
    primary = inputs[PRIMARY]
    run_price_history()
    for name, fn, arg in (("E1", run_e1_convergence, primary), ("E2", run_e2_error_decay, primary),
                          ("E3", run_e3_strike, primary), ("E4", run_e4_volatility, primary),
                          ("E5", run_e5_maturity, primary), ("E6", run_e6_stocks, inputs),
                          ("E7", run_e7_timing, primary), ("E8", run_e8_parity, primary),
                          ("tree", run_example_tree, primary)):
        fn(arg)
        print(f"done {name}")
    print(f"All outputs written to {ROOT / 'outputs'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

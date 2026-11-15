# European Option Pricing: Binomial Tree vs Black–Scholes

MAC F244 project. We price European calls and puts on NSE stocks with two methods and compare them:

- **Black–Scholes (BS):** a closed-form formula.
- **Cox–Ross–Rubinstein (CRR) binomial tree:** a numerical method.

The comparison looks at how the binomial price converges to BS, how sensitive both are to strike, volatility, maturity and stock, and how fast each one runs.

> Project scope, team split and schedule are in [`prd.md`](prd.md). This file is the **technical spec**: what every Python file contains, how it works, and how to run it.

---

## 1. Quick start

Target machine: **Ubuntu 22.04, Python 3.10** (system `python3`).

```bash
# one-time setup (from the project root)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# every session
source .venv/bin/activate
python data.py           # refresh market data (falls back to cached CSVs if offline)
python black_scholes.py  # self-checks for the BS model
python binomial.py       # self-checks for the binomial model
python compare.py        # run all experiments -> outputs/
```

`compare.py` calls `data.py` itself, so in practice `python compare.py` is the only command you need.

### Why a virtual environment?
The system Python has old apt-installed packages (numpy 1.21, scipy 1.8, matplotlib 3.5), and pandas and yfinance are not installed. If we `pip install --user` pandas or yfinance, pip will pull numpy 2.x, which can break the apt-installed scipy and matplotlib. A project-local `.venv` keeps a clean, consistent set of packages and leaves the system untouched. `python3-venv` is already installed on this machine.

---

## 2. Dependencies

`requirements.txt`:

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
yfinance>=1.8
jinja2>=3.1
```

| Package | Used in | What for |
|---|---|---|
| `numpy` | all | arrays, `exp`, `log`, `sqrt`, vectorised tree roll-back |
| `scipy` | `black_scholes.py` | `scipy.stats.norm.cdf`, the standard normal CDF N(·) |
| `pandas` | `data.py`, `compare.py` | price series, CSV I/O, result tables, `DataFrame.to_latex()` |
| `matplotlib` | `compare.py` | all figures (non-interactive `Agg` backend, so it works without a display) |
| `yfinance` | `data.py` only | downloading NSE price history from Yahoo Finance |
| `jinja2` | `compare.py` (indirectly) | required by `DataFrame.to_latex()` to write the `.tex` tables |

Standard library: `math`, `time`, `timeit`, `json`, `os`, `pathlib`, `datetime`.

**Notes on yfinance (checked against yfinance 1.8.0, released Nov 2026):**
- Prices are **adjusted for splits and dividends by default** (`auto_adjust=True`). The `Close` column is therefore the adjusted close. There is no separate `Adj Close` column.
- `yf.download()` can return **MultiIndex columns** even for one ticker. We avoid this by using `yf.Ticker(t).history(...)`, which returns plain columns (`Open, High, Low, Close, Volume, ...`).
- On failure (no network, rate limit, bad ticker), yfinance usually **does not raise**. It prints an error and returns an **empty DataFrame**. Our code must check for emptiness itself (see §4.3).
- Rate limiting raises `YFRateLimitError` in recent versions. We catch it like any other exception.
- yfinance pulls in `curl_cffi`, `requests`, `lxml`, `beautifulsoup4` and a few others automatically.

**Import graph** (no cycles; the three model files are independent of each other):

```
compare.py ──► data.py
           ──► black_scholes.py
           ──► binomial.py
```

---

## 3. Project layout

```
stochastic/
├── desc.md                # this file (technical spec)
├── prd.md                 # project plan, team split, schedule
├── requirements.txt
├── data.py                # market data -> (S0, sigma, r)
├── black_scholes.py       # closed-form BS call/put
├── binomial.py            # CRR binomial tree call/put
├── compare.py             # experiments, figures, tables
├── data/                  # cached price CSVs (one per ticker), committed to the repo
│   ├── RELIANCE.NS.csv
│   ├── SUNPHARMA.NS.csv
│   └── ADANIENT.NS.csv
├── outputs/
│   ├── figures/           # every figure as .pdf (report) and .png (slides)
│   ├── tables/            # every table as .csv and .tex
│   └── run_info.json      # which data was used, S0/sigma/r per stock, timestamp
└── report/
    ├── report.tex
    └── refs.bib
```

---

## 4. `data.py`: market inputs

**Job:** turn a ticker into model inputs `S0` (spot) and `sigma` (annualised volatility), and provide `r` (risk-free rate). It also keeps a local CSV cache so the project still runs when Yahoo Finance is unavailable.

**Imports:** `math`, `datetime`, `pathlib`, `numpy`, `pandas`, `yfinance`.

### 4.1 Constants

| Name | Value | Meaning |
|---|---|---|
| `TICKERS` | `["RELIANCE.NS", "SUNPHARMA.NS", "ADANIENT.NS"]` | NSE stocks used in the project. `.NS` is Yahoo's suffix for the National Stock Exchange. |
| `HISTORY_PERIOD` | `"1y"` | how much history to download (rolling: the last year up to today) |
| `TRADING_DAYS` | `252` | trading days per year, used to annualise volatility |
| `TBILL_YIELD` | `0.0629` | RBI 364-day Treasury Bill yield (Oct 2026), annual |
| `RISK_FREE_RATE` | `math.log(1 + TBILL_YIELD)` ≈ `0.06100` | the same rate as a **continuously compounded** rate, which is the convention both BS and CRR use (`e^{-rT}`) |
| `DATA_DIR` | `Path(__file__).parent / "data"` | cache folder |
| `MIN_ROWS` | `200` | sanity threshold: fewer rows than this counts as a bad download |
| `MAX_STALENESS_DAYS` | `7` | a download whose last row is older than this counts as stale |

### 4.2 Functions

| Function | Signature | Returns | Logic |
|---|---|---|---|
| `download_prices` | `(ticker: str, period: str = HISTORY_PERIOD) -> pd.Series` | adjusted close series; empty if it failed | `yf.Ticker(ticker).history(period=period, auto_adjust=True)["Close"]`, wrapped in `try/except Exception` (returns an empty Series on any error). Removes the timezone from the index with `.tz_localize(None)` so the CSV stays clean. |
| `validate_prices` | `(prices: pd.Series) -> tuple[bool, str]` | `(ok, reason)` | `ok` only if: not empty; at least `MIN_ROWS` rows after `dropna()`; all prices > 0; last date within `MAX_STALENESS_DAYS` of today. Otherwise `reason` explains why, e.g. `"only 12 rows"`. |
| `cache_path` | `(ticker: str) -> Path` | `data/<ticker>.csv` | |
| `save_cache` | `(ticker, prices) -> None` | | writes to `data/<ticker>.csv.tmp`, then `os.replace()` it onto the real file. A crash halfway through therefore never leaves a corrupted CSV. |
| `load_cache` | `(ticker) -> pd.Series` | cached series; empty if no file | `pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")` |
| `get_prices` | `(ticker) -> tuple[pd.Series, str]` | `(prices, status)` | **main data logic, see §4.3** |
| `estimate_volatility` | `(prices: pd.Series) -> float` | annualised σ | `log_ret = np.log(prices / prices.shift(1)).dropna()`, then `sigma = log_ret.std(ddof=1) * sqrt(TRADING_DAYS)` |
| `current_price` | `(prices) -> float` | `S0` | `float(prices.iloc[-1])`, the most recent close |
| `get_inputs` | `(ticker) -> dict` | `{"ticker", "S0", "sigma", "r", "as_of", "n_obs", "status"}` | calls `get_prices`, then the two estimators. `as_of` is the date of the last price. Raises `RuntimeError` if there is neither fresh nor cached data. |
| `main` | `() -> None` | | runs `get_inputs` for every ticker and prints a summary table (see §4.4) |

`if __name__ == "__main__": main()`

### 4.3 Fetch logic: rolling data with cache fallback

We always try to get the **latest** year of data. If that works, it overwrites the cache. If it fails, we use the last good cached copy and say so clearly.

```
get_prices(ticker):
    fresh = download_prices(ticker)
    ok, reason = validate_prices(fresh)
    if ok:
        save_cache(ticker, fresh)
        return fresh, "FRESH (downloaded, cache updated)"
    cached = load_cache(ticker)
    if not cached.empty:
        return cached, f"CACHED (download failed: {reason}; using data up to {cached.index[-1].date()})"
    raise RuntimeError(f"{ticker}: download failed ({reason}) and no cache at {cache_path(ticker)}")
```

**On "pinging yfinance first":** there is no official Yahoo Finance health or status endpoint, and a separate ping wouldn't prove the real download will succeed. For example, Yahoo can answer a ping and still rate-limit or return empty data for a ticker. The most reliable check is the one above: **attempt the real download and validate what comes back** (non-empty, enough rows, positive prices, recent last date). It costs nothing extra, because we need the download anyway, and it catches every failure mode: no internet, Yahoo down, rate limit, delisted ticker, or a partial or stale response.

**A consequence of rolling data:** S0, σ and every price change slightly each time fresh data is downloaded. To keep the report consistent:
- `compare.py` records the exact data used in `outputs/run_info.json` (status, `as_of` date, S0, σ, r per ticker), and the report quotes that `as_of` date.
- Do the **final** `python compare.py` run just before writing the report, then don't rerun it, or rerun and refresh the numbers in the report.

### 4.4 Example output of `python data.py`

```
Ticker        Status                                        As of        S0         sigma    n_obs
RELIANCE.NS   FRESH (downloaded, cache updated)             2026-11-06   1452.30    0.2214   249
SUNPHARMA.NS  FRESH (downloaded, cache updated)             2026-11-06   1860.00    0.1959   249
ADANIENT.NS   CACHED (download failed: empty response;      2026-11-05   2398.10    0.3920   248
              using data up to 2026-11-05)
Risk-free rate r = 0.06100 (continuous, from 6.29% RBI 364-day T-bill)
```

The numbers above are illustrative, not real.

---

## 5. `black_scholes.py`: closed-form model

**Job:** compute the Black–Scholes price of a European call or put.

**Imports:** `numpy`, `scipy.stats.norm`.

### 5.1 Functions

| Function | Signature | Returns | Formula |
|---|---|---|---|
| `_check_inputs` | `(S0, K, T, sigma) -> None` | | raises `ValueError` unless `S0 > 0`, `K > 0`, `T > 0`, `sigma > 0` |
| `d1_d2` | `(S0, K, T, r, sigma) -> (d1, d2)` | two floats or arrays | `d1 = (ln(S0/K) + (r + σ²/2)T) / (σ√T)`, `d2 = d1 − σ√T` |
| `bs_call` | `(S0, K, T, r, sigma) -> float` | call price | `C = S0·N(d1) − K·e^{−rT}·N(d2)` |
| `bs_put` | `(S0, K, T, r, sigma) -> float` | put price | put–call parity: `P = C − S0 + K·e^{−rT}` |
| `bs_price` | `(S0, K, T, r, sigma, option="call") -> float` | price | dispatches to `bs_call` or `bs_put`; raises `ValueError` for any other `option` string |

**Design notes:**
- These are pure functions with no I/O and no global state, so they are easy to test and time.
- They use `np.log`, `np.exp` and `np.sqrt`, so `K` or `sigma` can be passed as **NumPy arrays** and you get a whole sweep in one call. `compare.py` uses this for the strike and volatility sweeps.
- `option` is the string `"call"` or `"put"`. The same convention is used in `binomial.py`.

### 5.2 Self-check (`if __name__ == "__main__":`)

These are known textbook values (Hull) for S0=100, K=100, T=1, r=0.05, σ=0.2:

| Check | Expected |
|---|---|
| `bs_call(...)` | 10.4506 (±1e-4) |
| `bs_put(...)` | 5.5735 (±1e-4) |
| parity: `C − P − (S0 − K e^{−rT})` | ≈ 0 (±1e-10) |
| deep ITM call (K=1) | ≈ `S0 − K e^{−rT}` |
| deep OTM call (K=1000) | ≈ 0 |
| monotonicity | call price increases with σ and with T |

The script prints `PASS` or `FAIL` for each line and exits with a non-zero code if anything fails.

---

## 6. `binomial.py`: CRR binomial tree

**Job:** price a European call or put by building a recombining up/down tree and working backwards from maturity.

**Imports:** `numpy`.

### 6.1 Functions

| Function | Signature | Returns | Logic |
|---|---|---|---|
| `crr_params` | `(T, r, sigma, n) -> (u, d, p, dt)` | tree parameters | `dt = T/n`, `u = e^{σ√dt}`, `d = 1/u`, `p = (e^{r·dt} − d)/(u − d)`. Raises `ValueError` if `not 0 < p < 1`, which only happens for extreme inputs with tiny `n`. |
| `terminal_prices` | `(S0, sigma, dt, n) -> np.ndarray` | the `n+1` stock prices at maturity | `j = np.arange(n+1)` (number of up moves); `S_T = S0 * u**(2j − n)`, because `d = 1/u` means `u^j·d^(n−j) = u^(2j−n)`. Written as `S0 * np.exp(σ√dt * (2j − n))` to avoid overflow. |
| `payoff` | `(S_T, K, option) -> np.ndarray` | payoff at maturity | call: `np.maximum(S_T − K, 0)`; put: `np.maximum(K − S_T, 0)` |
| `binomial_price` | `(S0, K, T, r, sigma, n, option="call") -> float` | option price today | **see §6.2** |
| `binomial_tree_full` | `(S0, K, T, r, sigma, n, option="call") -> (stock, value)` | two `(n+1)×(n+1)` arrays (upper-triangular) | same maths, but keeps every node. **Only for small `n` (≤ 5)**: used to print or draw the example tree for the slides and report. |

### 6.2 Backward induction (vectorised)

```
u, d, p, dt = crr_params(T, r, sigma, n)
disc = exp(−r·dt)
V = payoff(terminal_prices(S0, sigma, dt, n), K, option)    # length n+1, index j = number of up moves
for step in range(n, 0, −1):                         # n times
    V = disc * (p * V[1:] + (1 − p) * V[:-1])         # length shrinks by 1 each step
return V[0]
```

At each step, a node with `j` up moves has an "up" child `j+1` and a "down" child `j`. The node's value is the discounted risk-neutral average of those two children. `V[1:]` lines up all the up-children and `V[:-1]` all the down-children, so a whole layer is computed in one NumPy operation.

- **Time:** O(n²) arithmetic, but only `n` Python loop iterations, so n=1000 takes a few milliseconds.
- **Memory:** O(n), because only one layer is stored at a time and the full 2-D tree is never built.

### 6.3 Self-check (`if __name__ == "__main__":`)

| Check | Expected |
|---|---|
| Hull inputs (S0=K=100, T=1, r=0.05, σ=0.2), n=1000, call | within 0.01 of BS 10.4506 |
| same, put | within 0.01 of BS 5.5735 |
| put–call parity on binomial prices (any n) | holds to ±1e-10, because CRR satisfies parity exactly |
| `n=1` call, by hand | matches the one-step formula `e^{−rT}[p·max(S0u−K,0) + (1−p)·max(S0d−K,0)]` |
| `binomial_tree_full(n=3)` | its root value equals `binomial_price(n=3)` |

The binomial self-check imports `black_scholes` **only inside the `__main__` block**, to compare against BS. The `binomial` module itself stays independent.

---

## 7. `compare.py`: experiments, figures, tables

**Job:** load the market inputs, run every experiment, and save every figure and table used in the slides and report. Running `python compare.py` regenerates all of `outputs/`.

**Imports:** `json`, `time`, `timeit`, `pathlib`, `datetime`, `numpy`, `pandas`, `matplotlib` (with `matplotlib.use("Agg")` before importing `pyplot`), plus `data`, `black_scholes` and `binomial`.

### 7.1 Configuration (constants at the top of the file)

| Name | Default | Meaning |
|---|---|---|
| `PRIMARY` | `"RELIANCE.NS"` | the stock used for E1–E5, E7 and E8 |
| `T_DEFAULT` | `0.25` | 3-month maturity (in years) |
| `MONEYNESS_DEFAULT` | `1.0` | `K = m · S0`, so 1.0 means at-the-money |
| `N_DEFAULT` | `500` | steps for single-price comparisons |
| `N_LIST` | `[5, 10, 20, 50, 100, 200, 500, 1000]` | steps for the convergence plots |
| `N_DENSE` | `range(5, 401)` | every n from 5 to 400, for the odd/even plot in E2 |
| `MONEYNESS_LIST` | `np.linspace(0.8, 1.2, 9)` | strike sweep (E3) |
| `SIGMA_LIST` | `np.linspace(0.10, 0.60, 11)` | volatility sweep (E4) |
| `T_LIST` | `[1/12, 0.25, 0.5, 1.0]` | maturity sweep (E5): 1, 3, 6 and 12 months |
| `TIMING_REPEATS` | `7` | timing repeats; we report the median |
| `OUT_FIG`, `OUT_TAB` | `outputs/figures`, `outputs/tables` | created automatically |

**Strike convention:** `K = m · S0` exactly, with no rounding to a real strike grid. This keeps moneyness identical across stocks, so they can be compared fairly (see `prd.md` §3.5).

### 7.2 Helper functions

| Function | Purpose |
|---|---|
| `setup_dirs()` | creates the output folders |
| `save_fig(fig, name)` | saves `outputs/figures/<name>.pdf` and `.png` (dpi 200), then closes the figure |
| `save_table(df, name, caption)` | saves `outputs/tables/<name>.csv` and `<name>.tex` using `df.to_latex(index=False, float_format="%.4f", caption=caption, label=f"tab:{name}")` |
| `time_call(fn, repeats)` | runs `timeit.repeat(fn, number=k, repeat=repeats)`, picking `k` so one batch takes about 0.2 s, and returns the median seconds per call |
| `abs_err(a, b)`, `rel_err(a, b)` | `|a − b|` and `|a − b| / b` |
| `write_run_info(inputs)` | dumps `outputs/run_info.json` with the run timestamp, plus status, as_of, S0, σ, r and n_obs per ticker |

### 7.3 Experiments

Each experiment is a function `run_eN(inputs) -> None` that saves its own outputs. `main()` calls them in order.

| ID | Function | What it computes | Saved outputs |
|---|---|---|---|
| E1 | `run_e1_convergence` | ATM call and put on `PRIMARY`: binomial price for every `n` in `N_LIST`, plus the BS price | `e1_convergence.csv/.tex` (n, binom call, BS call, abs err, rel err, and the same for the put); figure `e1_convergence` (binomial price vs n on a log x-axis, BS as a dashed horizontal line; one panel each for call and put) |
| E2 | `run_e2_error_decay` | abs error vs n for every `n` in `N_DENSE`; fits the slope of `log(err)` against `log(n)` with `np.polyfit`, separately for odd and even n; also the **signed** error for n = 50–100 | figure `e2_error_loglog` (left: log–log abs error, odd and even n in different markers, an O(1/n) reference line; right: signed error vs n, showing the sign alternating with parity); `e2_slope.csv` (fitted slopes, expected ≈ −1) |
| E3 | `run_e3_strike` | for each moneyness in `MONEYNESS_LIST`: BS and binomial (`N_DEFAULT`) call and put prices and errors | `e3_strike.csv/.tex`; figure `e3_strike` (price vs K/S0 for both models; second panel shows the error vs K/S0) |
| E4 | `run_e4_volatility` | same, sweeping σ over `SIGMA_LIST` with ATM strike | `e4_volatility.csv/.tex`; figure `e4_volatility` (price vs σ; error vs σ) |
| E5 | `run_e5_maturity` | same, over `T_LIST` | `e5_maturity.csv/.tex`; figure `e5_maturity` |
| E6 | `run_e6_stocks` | for every ticker: its own S0 and σ, ATM, `T_DEFAULT`; BS and binomial prices, abs error and **relative error**; plus rel error vs n (`N_LIST`) for each stock | `e6_stocks.csv/.tex` (ticker, S0, σ, BS call, binom call, abs err, rel err, and the same for the put); figure `e6_stocks_convergence` (rel error vs n, one line per stock, log–log) |
| E7 | `run_e7_timing` | median time per price: BS once, binomial at each `n` in `N_LIST`; also the slope of log(time) vs log(n) | `e7_timing.csv/.tex`; figure `e7_timing` (log–log time vs n, BS as a horizontal line; expected binomial slope ≈ 2 for large n) |
| E8 | `run_e8_parity` | for both models at `N_DEFAULT`: `C − P − (S0 − K e^{−rT})` | `e8_parity.csv/.tex` (residuals ≈ 0) |
| — | `run_example_tree` | `binomial_tree_full(n=3)` on `PRIMARY` | `example_tree_n3.csv/.tex` (stock and option value per node), used for the tree slide and figure |

### 7.4 `main()` flow

```
setup_dirs()
inputs = {t: data.get_inputs(t) for t in data.TICKERS}    # prints FRESH/CACHED per ticker
write_run_info(inputs)
run_e1 ... run_e8, run_example_tree
print summary: where outputs went, total runtime, and the data status per ticker
```

If one ticker has neither fresh nor cached data, `main()` prints a warning and **skips that ticker** in E6 instead of crashing. If `PRIMARY` itself has no data, it stops with a clear error.

### 7.5 Results from the first full run (6 Nov 2026 data)

Reliance, ATM, T = 0.25, σ ≈ 0.240, r = 0.0610. These numbers change slightly whenever fresh data is downloaded, but the pattern should not.

- **E1:** the binomial price converges to BS, alternating around it with the parity of n (the listed n are all even except n=5, so they lie below BS except n=5). The call relative error is 4.2% at n=5, 0.22% at n=100, and 0.022% at n=1000.
- **E2:** fitted log–log slopes are −1.00 (odd n) and −1.00 (even n), so the error is **O(1/n)**. At-the-money, the *size* of the error is almost the same for odd and even n, so the absolute-error curve is a single smooth line. The **odd/even oscillation is in the sign**: the error alternates between negative and positive from one n to the next (see the right panel). For strikes away from the money (K = 1.05–1.1·S0, checked by hand) the oscillation also shows up in the size of the error. It is a known property of CRR, not a bug.
- **E3–E5:** the two models agree. The call price increases with σ and T and decreases with K.
- **E6:** the three stocks have σ ≈ 0.196 (SUNPHARMA), 0.240 (RELIANCE) and 0.375 (ADANIENT). The call relative error at n=500 rises with σ but only slightly: 4.19×10⁻⁴, 4.32×10⁻⁴, 4.54×10⁻⁴. The **put** relative error goes the other way (6.19×10⁻⁴, 5.94×10⁻⁴, 5.57×10⁻⁴), because the put is relatively more expensive at high σ while its absolute error equals the call's. The absolute error in INR scales with the share price. The three convergence curves lie almost on top of each other, so the main conclusion is that convergence barely depends on the stock. The range of σ available among liquid large caps is narrow (most are 0.20–0.25); the full range 0.10–0.60 is covered synthetically in E4. (HDFCBANK was dropped because its σ ≈ 0.232 was almost identical to RELIANCE's.)
- **E7:** BS takes about 70 µs per price (mostly SciPy overhead). The binomial takes about 15 µs at n=5 and 2.2 ms at n=1000, about 30× slower than BS. The fitted log–log slope for n ≥ 100 is **≈ 1.1, not 2**: the work is O(n²) in principle, but NumPy does each layer in one operation, so at n ≤ 1000 the per-step Python overhead dominates. The slope should approach 2 for much larger n.
- **E8:** parity residuals are about 1e-13 (BS) and 2e-11 (binomial), i.e. zero up to floating-point rounding.

---

## 7b. Building the report

The report is `report/report.tex` (bibliography `report/refs.bib`). It reads the figures from `outputs/figures/` and the tables from `outputs/tables/`, so run `python compare.py` first.

LaTeX is **not installed by default** on this machine. One-time install (needs sudo):

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

Build (from the `report/` folder):

```bash
cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report
```

Packages used: `geometry`, `amsmath`, `graphicx`, `booktabs`, `caption`, `hyperref`, and `adjustbox` (the generated tables use it to shrink to the page width). All are in the packages above. Alternatively, upload `report/`, `outputs/figures/` and `outputs/tables/` to Overleaf with the same relative layout.

The numbers quoted in the report text were written from the run of 6 Nov 2026. If you re-download data and rerun, check them against `outputs/tables/` (the tables and figures update automatically; the sentences do not).

---

## 7c. Slides

`slides/option_pricing.pptx` is the presentation (12 slides for the talk, then 3 backup slides on strike, volatility and maturity; speaker notes are on every slide). It uses the figures in `outputs/figures/` and one native PowerPoint chart pair for the payoff diagrams. Colour code on every slide and figure: **blue = binomial tree, orange = Black–Scholes**.

`slides/make_deck.js` is the generator (Node + `pptxgenjs`, installed locally in `slides/node_modules`). It is only needed if you want to regenerate the deck after rerunning `compare.py`; for ordinary edits, just open the `.pptx` in PowerPoint. Some numbers on the slides are typed in (they come from the 6 Nov 2026 run, same as the report), so check them against `outputs/tables/` after a rerun: slide 4 (tree values are read from the CSV automatically, but `u`, `d`, `p` and the 78.42 / 73.22 text are typed), slide 5 (inputs table), slides 6–10 and 12.

```bash
cd slides && npm install pptxgenjs   # once
NODE_PATH=$PWD/node_modules node make_deck.js
```

(The generator also writes the theme colours into the file through a helper that ships with the Claude pptx skill, via the `PPTX_SKILL_DIR` environment variable. Without it the deck still builds but shapes use PowerPoint's stock colours.)

---

## 8. Conventions used across all files

| Item | Convention |
|---|---|
| Time `T` | years (3 months = 0.25) |
| Volatility `sigma` | annualised, as a decimal (0.22 = 22%) |
| Rate `r` | continuously compounded, as a decimal; always `data.RISK_FREE_RATE` |
| Prices | INR, adjusted close |
| `option` argument | the string `"call"` or `"put"` |
| Errors | absolute error in INR; **relative error** (fraction of the BS price) when comparing stocks |
| Randomness | none: every computation is deterministic |

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: pandas` / `yfinance` | the venv isn't active; run `source .venv/bin/activate` |
| `data.py` shows `CACHED (download failed ...)` | Yahoo is down, rate-limited, or there's no internet. The cached CSV is used; rerun later to refresh. |
| `RuntimeError: ... no cache` | first run while offline. Connect once to build `data/`. |
| binomial and BS don't converge | check that both use the same `r` (continuous), and that `sigma` is annual and `T` is in years |
| `ValueError: p not in (0,1)` | `n` is too small for the given `T` and `σ`; use a larger `n` |
| no plots appear | intended: figures are saved to `outputs/figures/`, not shown on screen |

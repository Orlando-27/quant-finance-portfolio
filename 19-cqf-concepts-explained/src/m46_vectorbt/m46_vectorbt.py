#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 46 -- VECTORISED BACKTESTING: MOVING AVERAGE CROSSOVER
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Vectorised backtesting avoids explicit loops over trading days by expressing
the entire strategy as array operations.  This yields two key advantages:

  1. Speed: numpy/pandas operations run in optimised C, not Python loops.
  2. Clarity: signal generation and P&L calculation are expressed concisely.

The canonical strategy studied here is the Dual Moving Average Crossover:

    Signal_t = +1  if  SMA(fast, t) > SMA(slow, t)   [long]
    Signal_t = -1  if  SMA(fast, t) < SMA(slow, t)   [short]
    Signal_t =  0  before both MAs are defined        [flat]

where SMA(n, t) = (1/n) * sum_{i=0}^{n-1} P_{t-i}

RETURN CALCULATION
------------------
Let r_t = log(P_t / P_{t-1}) be the log-return at time t.
The strategy return is:

    R_t^strat = Signal_{t-1} * r_t

The lag (Signal_{t-1}) is critical: the signal at close of day t-1
determines the position held during day t.  Using same-day signals
introduces look-ahead bias.

TRANSACTION COSTS
-----------------
Each position change incurs a round-trip cost c (in log-return units).
Let delta_t = |Signal_t - Signal_{t-1}| be the position change indicator.

    R_t^net = R_t^strat - (c / 2) * delta_t

For equities, c ~ 5-10 bps (0.0005-0.001) round-trip.  High turnover
strategies are more sensitive to transaction costs.

PARAMETER SWEEP (OPTIMISATION)
--------------------------------
We sweep a grid of (fast, slow) MA pairs and compute the in-sample Sharpe.
This generates a performance heatmap that reveals the sensitivity of results
to parameter choice.

WALK-FORWARD VALIDATION
-----------------------
To avoid in-sample overfitting, we apply walk-forward analysis:
  1. Train window: optimise (fast, slow) on first W_train days
  2. Test window:  trade with optimal parameters for next W_test days
  3. Slide: advance both windows by W_test days and repeat
The out-of-sample Sharpe is the performance metric that matters.

PERFORMANCE METRICS
--------------------
Sharpe Ratio:     SR = (mean(R) / std(R)) * sqrt(252)
Sortino Ratio:    SoR = (mean(R) / std(R^-)) * sqrt(252)   R^- = min(R, 0)
Calmar Ratio:     CR = CAGR / Max Drawdown
Max Drawdown:     MDD = max_t (peak_t - valley_t) / peak_t
Turnover:         % of days with position change
Hit Rate:         fraction of non-flat days with positive return
Profit Factor:    sum(wins) / sum(|losses|)

REFERENCES
----------
[1] Chan, E. (2008). Quantitative Trading. Wiley.
[2] Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
[3] Aronson, D. (2006). Evidence-Based Technical Analysis. Wiley.
[4] Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners."
    Journal of Finance 48(1):65-91.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m46_vectorbt/m46_vectorbt.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m46")
os.makedirs(FIGS, exist_ok=True)

# =============================================================================
# DARK THEME
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
TEXT   = "#e6edf3"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
AMBER  = "#d29922"
VIOLET = "#bc8cff"
TEAL   = "#39d353"

plt.rcParams.update({
    "figure.facecolor" : DARK,  "axes.facecolor"  : PANEL,
    "axes.edgecolor"   : GRID,  "axes.labelcolor" : TEXT,
    "axes.titlecolor"  : TEXT,  "xtick.color"     : TEXT,
    "ytick.color"      : TEXT,  "text.color"      : TEXT,
    "grid.color"       : GRID,  "grid.linestyle"  : "--",
    "grid.alpha"       : 0.5,   "legend.facecolor": PANEL,
    "legend.edgecolor" : GRID,  "font.family"     : "monospace",
    "font.size"        : 9,     "axes.titlesize"  : 10,
})

def section(n, msg): print(f"  [{n:02d}] {msg}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def sma(prices: np.ndarray, n: int) -> np.ndarray:
    """
    Simple Moving Average via cumsum (O(N), no loop).

    SMA(n)_t = (1/n) * sum_{i=0}^{n-1} P_{t-i}

    First n-1 values are NaN (insufficient history).
    """
    out = np.full(len(prices), np.nan)
    cs  = np.cumsum(prices)
    out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def ema(prices: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential Moving Average.
    alpha = 2 / (span + 1)
    EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
    """
    alpha = 2.0 / (span + 1)
    out   = np.full(len(prices), np.nan)
    out[span-1] = prices[:span].mean()
    for i in range(span, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i-1]
    return out

def crossover_signal(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """
    +1 when fast > slow, -1 when fast < slow, 0 otherwise (NaN region).
    Signal is lagged by 1 day before applying to returns.
    """
    sig = np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))
    sig[np.isnan(fast) | np.isnan(slow)] = 0.0
    return sig

def strategy_returns(ret: np.ndarray, signal: np.ndarray,
                     cost_bps: float = 5.0) -> np.ndarray:
    """
    Compute net strategy returns with transaction costs.

    Parameters
    ----------
    ret      : log-returns array length N
    signal   : position signal length N (applied at t-1 to r_t)
    cost_bps : one-way transaction cost in basis points

    Returns
    -------
    net strategy log-returns length N
    """
    c       = cost_bps * 1e-4                   # convert bps to decimal
    pos     = np.concatenate([[0], signal[:-1]]) # lag by 1 day
    gross   = pos * ret
    delta   = np.abs(np.diff(pos, prepend=0))   # position change
    costs   = c * delta
    return gross - costs

def performance(ret: np.ndarray, ann: int = 252) -> dict:
    """
    Comprehensive performance metrics for a return series.
    """
    r    = ret[ret != 0] if (ret != 0).any() else ret
    mu   = ret.mean()
    sig  = ret.std() + 1e-9
    neg  = ret[ret < 0]
    sig_d= neg.std() + 1e-9 if len(neg) > 0 else 1e-9

    cum  = np.exp(np.cumsum(ret))
    peak = np.maximum.accumulate(cum)
    dd   = (peak - cum) / (peak + 1e-9)
    mdd  = dd.max()

    cagr = np.exp(mu * ann) - 1
    sr   = (mu / sig)   * np.sqrt(ann)
    sor  = (mu / sig_d) * np.sqrt(ann)
    cal  = cagr / (mdd + 1e-9)

    nz   = ret[ret != 0]
    hit  = (nz > 0).mean() if len(nz) > 0 else 0.5
    wins = nz[nz > 0].sum() if (nz > 0).any() else 0
    loss = -nz[nz < 0].sum() if (nz < 0).any() else 1e-9
    pf   = wins / (loss + 1e-9)

    turn = (np.abs(np.diff(np.sign(ret), prepend=0)) > 0).mean()

    return {"sharpe": sr, "sortino": sor, "calmar": cal,
            "mdd": mdd, "cagr": cagr, "hit": hit,
            "pf": pf, "turnover": turn}


# =============================================================================
# 1.  DATA
# =============================================================================
print()
print("=" * 65)
print("  MODULE 46: VECTORISED BACKTESTING")
print("  MA Crossover | Param Sweep | Walk-Forward | Cost Analysis")
print("=" * 65)

raw   = yf.download("SPY", start="2010-01-01", end="2023-12-31",
                    auto_adjust=True, progress=False)
close = raw["Close"].squeeze().dropna()
ret   = np.log(close / close.shift(1)).dropna().values
dates = close.index[1:]
N     = len(ret)
P     = close.values[1:]

section(1, f"SPY: {N} days  [{dates[0].date()} -- {dates[-1].date()}]")

# =============================================================================
# 2.  BASELINE STRATEGY  fast=20, slow=50, SMA crossover
# =============================================================================
FAST_BASE, SLOW_BASE = 20, 50
COST_BPS             = 5.0

fast_ma = sma(P, FAST_BASE)
slow_ma = sma(P, SLOW_BASE)
sig_base = crossover_signal(fast_ma, slow_ma)
ret_base = strategy_returns(ret, sig_base, COST_BPS)
ret_bnh  = ret.copy()                           # buy-and-hold benchmark

perf_base = performance(ret_base)
perf_bnh  = performance(ret_bnh)

section(2, f"Baseline SMA({FAST_BASE},{SLOW_BASE})  "
           f"Sharpe={perf_base['sharpe']:.3f}  "
           f"MDD={perf_base['mdd']:.3f}  "
           f"(BnH Sharpe={perf_bnh['sharpe']:.3f})")

# =============================================================================
# 3.  PARAMETER SWEEP: Sharpe heatmap over (fast, slow) grid
# =============================================================================
FAST_RANGE = [5, 10, 15, 20, 30, 40, 50]
SLOW_RANGE = [20, 30, 50, 75, 100, 120, 150, 200]

sharpe_grid = np.full((len(FAST_RANGE), len(SLOW_RANGE)), np.nan)

for i, f in enumerate(FAST_RANGE):
    for j, s in enumerate(SLOW_RANGE):
        if f >= s:
            continue
        fm  = sma(P, f);  sm_ = sma(P, s)
        sig = crossover_signal(fm, sm_)
        r_s = strategy_returns(ret, sig, COST_BPS)
        sharpe_grid[i, j] = performance(r_s)["sharpe"]

# Best in-sample parameters
best_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)
best_fast = FAST_RANGE[best_idx[0]]
best_slow = SLOW_RANGE[best_idx[1]]
best_sharpe = sharpe_grid[best_idx]

section(3, f"Best IS params: SMA({best_fast},{best_slow})  "
           f"Sharpe={best_sharpe:.3f}")

# =============================================================================
# 4.  EMA vs SMA COMPARISON
# =============================================================================
ema_fast = ema(P, FAST_BASE)
ema_slow = ema(P, SLOW_BASE)
sig_ema  = crossover_signal(ema_fast, ema_slow)
ret_ema  = strategy_returns(ret, sig_ema, COST_BPS)
perf_ema = performance(ret_ema)

section(4, f"EMA({FAST_BASE},{SLOW_BASE})  "
           f"Sharpe={perf_ema['sharpe']:.3f}  "
           f"MDD={perf_ema['mdd']:.3f}")

# =============================================================================
# 5.  TRANSACTION COST SENSITIVITY
# =============================================================================
costs_range = [0, 1, 2, 5, 10, 20, 50]
sharpe_vs_cost = []
for c in costs_range:
    r_c = strategy_returns(ret, sig_base, c)
    sharpe_vs_cost.append(performance(r_c)["sharpe"])

breakeven_cost = None
for k in range(len(costs_range)-1):
    if sharpe_vs_cost[k] >= 0 >= sharpe_vs_cost[k+1]:
        # Linear interpolation
        x0, x1 = costs_range[k], costs_range[k+1]
        y0, y1 = sharpe_vs_cost[k], sharpe_vs_cost[k+1]
        breakeven_cost = x0 - y0 * (x1 - x0) / (y1 - y0)
        break

section(5, f"Cost sensitivity: Sharpe@0bps={sharpe_vs_cost[0]:.3f}  "
           f"@5bps={sharpe_vs_cost[3]:.3f}  "
           f"breakeven~{breakeven_cost:.1f}bps" if breakeven_cost else
           f"Cost sensitivity computed  no breakeven in range")

# =============================================================================
# 6.  WALK-FORWARD VALIDATION
# =============================================================================
W_TRAIN = 504   # ~2 years
W_TEST  = 63    # ~1 quarter
FAST_WF = [10, 20, 30, 50]
SLOW_WF = [50, 75, 100, 150, 200]

wf_rets   = np.zeros(N)
wf_params = []

for start in range(0, N - W_TRAIN - W_TEST, W_TEST):
    train_end = start + W_TRAIN
    test_end  = min(train_end + W_TEST, N)
    P_tr      = P[start:train_end]
    ret_tr    = ret[start:train_end]

    # In-sample optimisation
    best_sr_tr = -np.inf
    best_f_tr, best_s_tr = FAST_WF[0], SLOW_WF[0]
    for f, s in product(FAST_WF, SLOW_WF):
        if f >= s:
            continue
        fm_  = sma(P_tr, f);  sm__ = sma(P_tr, s)
        sig_ = crossover_signal(fm_, sm__)
        r_   = strategy_returns(ret_tr, sig_, COST_BPS)
        sr_  = performance(r_)["sharpe"]
        if sr_ > best_sr_tr:
            best_sr_tr = sr_
            best_f_tr, best_s_tr = f, s

    # Out-of-sample application
    P_full  = P[start:test_end]
    ret_oos = ret[train_end:test_end]
    fm_oos  = sma(P_full, best_f_tr)[-len(ret_oos):]
    sm_oos  = sma(P_full, best_s_tr)[-len(ret_oos):]
    sig_oos = crossover_signal(fm_oos, sm_oos)
    r_oos   = strategy_returns(ret_oos, sig_oos, COST_BPS)
    wf_rets[train_end:test_end] = r_oos
    wf_params.append((best_f_tr, best_s_tr))

perf_wf = performance(wf_rets[W_TRAIN:])

section(6, f"Walk-forward OOS  Sharpe={perf_wf['sharpe']:.3f}  "
           f"MDD={perf_wf['mdd']:.3f}  "
           f"CAGR={perf_wf['cagr']*100:.1f}%")

# =============================================================================
# 7.  REGIME-FILTERED STRATEGY
#     Only trade when 200-day SMA slope > 0 (uptrend filter)
# =============================================================================
sma200 = sma(P, 200)
trend_filter = np.where(
    np.isnan(sma200) | np.isnan(np.roll(sma200, 20)),
    0.0,
    np.where(sma200 > np.roll(sma200, 20), 1.0, 0.0)
)
sig_filtered = sig_base * trend_filter
ret_filtered = strategy_returns(ret, sig_filtered, COST_BPS)
perf_filt    = performance(ret_filtered)

section(7, f"Regime-filtered  Sharpe={perf_filt['sharpe']:.3f}  "
           f"MDD={perf_filt['mdd']:.3f}  "
           f"Turnover={perf_filt['turnover']:.3f}")

# =============================================================================
# 8.  MONTHLY RETURN HEATMAP
# =============================================================================
import pandas as pd
ret_s   = pd.Series(ret_base, index=dates)
monthly = ret_s.resample("ME").sum()
yr_mo   = pd.DataFrame({
    "year": monthly.index.year,
    "month": monthly.index.month,
    "ret": monthly.values
})
years  = sorted(yr_mo["year"].unique())
months = list(range(1, 13))
heat   = np.full((len(years), 12), np.nan)
for _, row in yr_mo.iterrows():
    yi = years.index(int(row["year"]))
    heat[yi, int(row["month"])-1] = row["ret"] * 100

section(8, f"Monthly return heatmap: {len(years)} years x 12 months")

# =============================================================================
# FIGURE 1: STRATEGY OVERVIEW
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle("Module 46 -- Vectorised Backtesting: MA Crossover Strategy",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.45)

# 1A: Price + MAs + signals
ax = fig.add_subplot(gs[0, :])
t_ = np.arange(N)
ax.plot(t_, P,        color=TEXT,   lw=0.5, alpha=0.5, label="SPY Price")
ax.plot(t_, fast_ma,  color=ACCENT, lw=1.0, label=f"SMA({FAST_BASE})")
ax.plot(t_, slow_ma,  color=AMBER,  lw=1.0, label=f"SMA({SLOW_BASE})")
# Shade long/short regions
long_m  = sig_base == 1
short_m = sig_base == -1
ax.fill_between(t_, P.min()*0.99, P.max()*1.01,
                where=long_m,  alpha=0.07, color=GREEN,  label="Long")
ax.fill_between(t_, P.min()*0.99, P.max()*1.01,
                where=short_m, alpha=0.07, color=RED,    label="Short")
ax.set_title("SPY Price with MA Crossover Signals")
ax.set_xlabel("Day"); ax.set_ylabel("Price (USD)")
ax.legend(fontsize=7, ncol=6); ax.grid(True)

# 1B: Cumulative returns
ax = fig.add_subplot(gs[1, 0])
cum_base  = np.exp(np.cumsum(ret_base))   - 1
cum_ema_  = np.exp(np.cumsum(ret_ema))    - 1
cum_filt_ = np.exp(np.cumsum(ret_filtered)) - 1
cum_wf_   = np.exp(np.cumsum(wf_rets))   - 1
cum_bnh_  = np.exp(np.cumsum(ret_bnh))   - 1
ax.plot(t_, cum_bnh_  * 100, color=TEXT,   lw=1.0, alpha=0.6, label="Buy & Hold")
ax.plot(t_, cum_base * 100, color=ACCENT, lw=1.2, label=f"SMA({FAST_BASE},{SLOW_BASE})")
ax.plot(t_, cum_ema_  * 100, color=AMBER,  lw=1.2, label=f"EMA({FAST_BASE},{SLOW_BASE})")
ax.plot(t_, cum_filt_ * 100, color=GREEN,  lw=1.2, label="Regime-filtered")
ax.plot(t_, cum_wf_   * 100, color=VIOLET, lw=1.2, label="Walk-forward")
ax.set_title("Cumulative Return (%)")
ax.set_xlabel("Day"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1C: Drawdown
ax = fig.add_subplot(gs[1, 1])
for ret_arr, label, col in [
    (ret_base,     f"SMA base  MDD={perf_base['mdd']:.2f}", ACCENT),
    (ret_filtered, f"Filtered  MDD={perf_filt['mdd']:.2f}", GREEN),
    (ret_bnh,      f"BnH       MDD={perf_bnh['mdd']:.2f}",  TEXT),
]:
    cum_a = np.exp(np.cumsum(ret_arr))
    peak_ = np.maximum.accumulate(cum_a)
    dd_   = (peak_ - cum_a) / (peak_ + 1e-9)
    ax.fill_between(t_, -dd_*100, 0, alpha=0.4, color=col, label=label)
ax.set_title("Drawdown (%)")
ax.set_xlabel("Day"); ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1D: Cost sensitivity
ax = fig.add_subplot(gs[2, 0])
ax.plot(costs_range, sharpe_vs_cost, color=ACCENT, lw=2.0, marker="o", ms=6)
ax.axhline(0, color=RED, lw=1.0, ls="--", label="Sharpe = 0")
if breakeven_cost:
    ax.axvline(breakeven_cost, color=AMBER, lw=1.2, ls="--",
               label=f"Breakeven ~{breakeven_cost:.1f} bps")
ax.set_title("Transaction Cost Sensitivity")
ax.set_xlabel("One-way cost (bps)"); ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=7); ax.grid(True)

# 1E: Performance table
ax = fig.add_subplot(gs[2, 1])
ax.axis("off")
strategies = ["BnH", f"SMA({FAST_BASE},{SLOW_BASE})",
              f"EMA({FAST_BASE},{SLOW_BASE})", "Filtered", "Walk-Fwd"]
perfs = [perf_bnh, perf_base, perf_ema, perf_filt, perf_wf]
rows  = []
for name, p in zip(strategies, perfs):
    rows.append([name,
                 f"{p['sharpe']:.3f}",
                 f"{p['sortino']:.3f}",
                 f"{p['mdd']:.3f}",
                 f"{p['cagr']*100:.1f}%",
                 f"{p['hit']:.3f}",
                 f"{p['pf']:.2f}"])
cols = ["Strategy", "Sharpe", "Sortino", "MDD", "CAGR", "Hit", "PF"]
tbl  = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if r > 0 else GRID)
    cell.set_text_props(color=TEXT)
    cell.set_edgecolor(GRID)
ax.set_title("Performance Summary", pad=80)

fig.savefig(os.path.join(FIGS, "m46_fig1_strategy_overview.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: PARAMETER SWEEP HEATMAP
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle("Module 46 -- Parameter Sweep: Sharpe Heatmap",
             fontsize=12, color=TEXT, y=1.01)

ax = axes[0]
im = ax.imshow(sharpe_grid, cmap="RdYlGn", aspect="auto",
               vmin=-0.5, vmax=max(1.0, np.nanmax(sharpe_grid)))
ax.set_xticks(range(len(SLOW_RANGE))); ax.set_xticklabels(SLOW_RANGE, fontsize=8)
ax.set_yticks(range(len(FAST_RANGE))); ax.set_yticklabels(FAST_RANGE, fontsize=8)
ax.set_xlabel("Slow MA period"); ax.set_ylabel("Fast MA period")
ax.set_title("In-Sample Sharpe Ratio Grid")
for i in range(len(FAST_RANGE)):
    for j in range(len(SLOW_RANGE)):
        if not np.isnan(sharpe_grid[i, j]):
            ax.text(j, i, f"{sharpe_grid[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if sharpe_grid[i,j] > 0.3 else TEXT)
ax.scatter(best_idx[1], best_idx[0], marker="*", color=AMBER, s=200, zorder=5,
           label=f"Best: ({best_fast},{best_slow}) SR={best_sharpe:.2f}")
ax.legend(fontsize=7, loc="lower right")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Sharpe", color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)

# Walk-forward param history
ax = axes[1]
if wf_params:
    fast_wf_hist, slow_wf_hist = zip(*wf_params)
    windows = np.arange(len(wf_params))
    ax.step(windows, fast_wf_hist, color=ACCENT, lw=1.5, label="Optimal fast", where="post")
    ax.step(windows, slow_wf_hist, color=AMBER,  lw=1.5, label="Optimal slow", where="post")
    ax.set_title("Walk-Forward: Selected Parameters by Window")
    ax.set_xlabel("Walk-forward window"); ax.set_ylabel("MA period")
    ax.legend(fontsize=8); ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m46_fig2_param_sweep_walkforward.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: MONTHLY RETURN HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK)
ax.set_facecolor(PANEL)
fig.suptitle("Module 46 -- Monthly Return Heatmap: SMA Crossover Strategy",
             fontsize=12, color=TEXT)

vmax_ = np.nanpercentile(np.abs(heat), 95)
im    = ax.imshow(heat.T, cmap="RdYlGn", aspect="auto",
                  vmin=-vmax_, vmax=vmax_)
ax.set_yticks(range(12))
ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Year"); ax.set_ylabel("Month")
ax.set_title("Monthly Returns (%) -- SMA Crossover")
for i in range(len(years)):
    for j in range(12):
        if not np.isnan(heat[i, j]):
            ax.text(i, j, f"{heat[i,j]:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(heat[i,j]) > vmax_*0.5 else TEXT)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.set_label("Return (%)", color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m46_fig3_monthly_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("  MODULE 46 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Vectorised backtest: signal = np.where(fast>slow, 1, -1)")
print("  [2] Lag rule: position_{t-1} applied to return_t (no look-ahead)")
print("  [3] Net return = gross - cost * |delta_position|")
print(f"  [4] SMA({FAST_BASE},{SLOW_BASE}) Sharpe={perf_base['sharpe']:.3f}  "
      f"MDD={perf_base['mdd']:.3f}")
print(f"  [5] Best IS params: SMA({best_fast},{best_slow}) Sharpe={best_sharpe:.3f}")
print(f"  [6] Walk-forward OOS Sharpe={perf_wf['sharpe']:.3f} "
      f"(IS decay expected)")
print(f"  [7] Regime filter Sharpe={perf_filt['sharpe']:.3f}  "
      f"MDD={perf_filt['mdd']:.3f}")
print(f"  [8] Cost sensitivity: Sharpe@0bps={sharpe_vs_cost[0]:.3f}  "
      f"@5bps={sharpe_vs_cost[3]:.3f}")
print(f"  NEXT: M47 -- Event-Driven Backtesting: Commissions & Slippage")
print()

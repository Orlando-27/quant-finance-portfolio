#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 49 -- SHARPE, SORTINO, DRAWDOWNS & PERFORMANCE ATTRIBUTION
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Performance measurement goes far beyond a single Sharpe ratio.  A rigorous
tearsheet decomposes returns into risk-adjusted metrics, drawdown analysis,
factor attribution, and statistical significance tests.

RISK-ADJUSTED RETURN METRICS
------------------------------
Let r_t be the strategy daily log-return and r_f the risk-free rate.

Sharpe Ratio:
    SR = (mean(r - r_f) / std(r)) * sqrt(252)
    Measures reward per unit of total volatility.  Penalises upside and
    downside equally -- appropriate for symmetric return distributions.

Sortino Ratio:
    SoR = (mean(r - r_f) / sigma_d) * sqrt(252)
    sigma_d = std(r[r < MAR])   where MAR = minimum acceptable return
    Only penalises downside deviations -- preferred for skewed strategies.

Calmar Ratio:
    CR = CAGR / |Max Drawdown|
    Relates annualised compounded growth to worst peak-to-trough loss.

Omega Ratio:
    Omega(L) = E[max(r - L, 0)] / E[max(L - r, 0)]
             = integral_{L}^{inf} (1 - F(r)) dr
               / integral_{-inf}^{L} F(r) dr
    Captures the full return distribution; Omega > 1 => more gains than
    losses above threshold L.

Information Ratio:
    IR = mean(r - r_benchmark) / std(r - r_benchmark) * sqrt(252)
    Measures active return per unit of tracking error.

DRAWDOWN ANALYSIS
-----------------
Max Drawdown:
    MDD = max_{t in [0,T]} (W_peak(t) - W_t) / W_peak(t)
    where W_peak(t) = max_{s in [0,t]} W_s

Drawdown Duration: number of days spent below the previous peak.
Average Drawdown: mean of all drawdown episodes.
Recovery Time: days from trough to full recovery.

BRINSON-HOOD-BEEBOWER (BHB) ATTRIBUTION
-----------------------------------------
The BHB model decomposes active return into:
    R_active = R_allocation + R_selection + R_interaction

    R_allocation = sum_k (w_k^p - w_k^b) * r_k^b
    R_selection  = sum_k w_k^b * (r_k^p - r_k^b)
    R_interaction= sum_k (w_k^p - w_k^b) * (r_k^p - r_k^b)

where w_k^p, w_k^b are portfolio and benchmark sector weights,
and r_k^p, r_k^b are portfolio and benchmark sector returns.

STATISTICAL SIGNIFICANCE
-------------------------
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012):
    DSR = SR* * N(z)
    z = (SR_obs - E[SR_max]) / std(SR_max)

The DSR corrects for selection bias when testing multiple strategies.
Under the null hypothesis of zero Sharpe, the expected maximum Sharpe
from N_trials independent strategies is:

    E[SR_max] = (1 - gamma_e) * N^{-1}(1 - 1/N_trials)
              + gamma_e * N^{-1}(1 - 1/(N_trials * e))

REFERENCES
----------
[1] Sharpe, W. (1966). "Mutual Fund Performance." JB 39(1):119-138.
[2] Sortino, F. & Price, L. (1994). "Performance Measurement in a
    Downside Risk Framework." JPM 21(1):59-64.
[3] Brinson, G., Hood, R. & Beebower, G. (1986). "Determinants of
    Portfolio Performance." FAJ 42(4):39-44.
[4] Bailey, D. & Lopez de Prado, M. (2012). "The Sharpe Ratio
    Efficient Frontier." JOIS 5(3):13-38.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m49_pyfolio/m49_pyfolio.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm
import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m49")
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
# PERFORMANCE ENGINE
# =============================================================================
class PerformanceEngine:
    """
    Comprehensive performance analytics computed from a daily return series.

    All metrics are implemented from first principles without external
    performance libraries, following the mathematical derivations above.
    """

    def __init__(self, returns: np.ndarray, benchmark: np.ndarray = None,
                 rf: float = 0.04 / 252, ann: int = 252):
        self.r   = returns
        self.b   = benchmark if benchmark is not None else np.zeros_like(returns)
        self.rf  = rf
        self.ann = ann
        self._compute()

    def _compute(self):
        r = self.r; rf = self.rf; ann = self.ann

        # Basic moments
        self.mean   = r.mean()
        self.vol    = r.std()
        self.skew   = stats.skew(r)
        self.kurt   = stats.kurtosis(r)   # excess kurtosis

        # Excess return
        er = r - rf

        # Sharpe
        self.sharpe = (er.mean() / (r.std() + 1e-9)) * np.sqrt(ann)

        # Sortino (MAR = rf)
        down = r[r < rf]
        self.sigma_d = down.std() + 1e-9
        self.sortino = (er.mean() / self.sigma_d) * np.sqrt(ann)

        # CAGR
        self.cagr = np.exp(self.mean * ann) - 1

        # Cumulative wealth and drawdown
        self.wealth = np.exp(np.cumsum(r))
        peak        = np.maximum.accumulate(self.wealth)
        self.dd     = (peak - self.wealth) / (peak + 1e-9)
        self.mdd    = self.dd.max()

        # Calmar
        self.calmar = self.cagr / (self.mdd + 1e-9)

        # Omega ratio (threshold = 0)
        gains  = np.maximum(r, 0).sum()
        losses = np.maximum(-r, 0).sum()
        self.omega = gains / (losses + 1e-9)

        # Information ratio vs benchmark
        active = r - self.b
        self.ir = (active.mean() / (active.std() + 1e-9)) * np.sqrt(ann)

        # Hit rate & profit factor
        nz = r[r != 0]
        self.hit = (nz > 0).mean() if len(nz) > 0 else 0.5
        wins  = nz[nz > 0].sum() if (nz > 0).any() else 0.0
        losss = -nz[nz < 0].sum() if (nz < 0).any() else 1e-9
        self.profit_factor = wins / losss

        # VaR and CVaR
        self.var95  = np.percentile(r, 5)
        self.var99  = np.percentile(r, 1)
        self.cvar95 = r[r <= self.var95].mean()
        self.cvar99 = r[r <= self.var99].mean()

        # Tail ratio
        self.tail_ratio = abs(np.percentile(r, 95)) / (abs(np.percentile(r, 5)) + 1e-9)

        # Drawdown episodes
        self.dd_episodes = self._drawdown_episodes()

        # Monthly returns
        self.monthly_rets = self._monthly_returns()

    def _drawdown_episodes(self):
        """Identify all drawdown episodes: (start, trough, end, depth, duration)."""
        in_dd    = False
        episodes = []
        start = trough = trough_val = 0
        peak_val = self.wealth[0]

        for i, w in enumerate(self.wealth):
            if not in_dd:
                if w < peak_val:
                    in_dd = True; start = i
                    trough = i; trough_val = w
                else:
                    peak_val = w
            else:
                if w < trough_val:
                    trough = i; trough_val = w
                if w >= peak_val:
                    depth    = (peak_val - trough_val) / peak_val
                    duration = i - start
                    episodes.append((start, trough, i, depth, duration))
                    in_dd = False; peak_val = w

        episodes.sort(key=lambda x: x[3], reverse=True)
        return episodes

    def _monthly_returns(self):
        """Aggregate daily returns to monthly."""
        n = len(self.r)
        days_per_month = 21
        monthly = []
        for i in range(0, n, days_per_month):
            chunk = self.r[i:i+days_per_month]
            monthly.append(chunk.sum())
        return np.array(monthly)

    def deflated_sharpe(self, n_trials: int = 100) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012).
        Corrects observed Sharpe for selection bias across n_trials.
        """
        T    = len(self.r)
        sr   = self.sharpe / np.sqrt(self.ann)   # daily SR
        # Expected max SR under N(0,1) trial SRs
        euler_gamma = 0.5772
        e_max = (1 - euler_gamma) * norm.ppf(1 - 1/n_trials) + \
                euler_gamma * norm.ppf(1 - 1/(n_trials * np.e))
        # Variance of SR estimate
        var_sr = (1 + 0.5*sr**2 - self.skew*sr +
                  (self.kurt/4)*sr**2) / T
        dsr = norm.cdf((sr - e_max) / (np.sqrt(var_sr) + 1e-9))
        return float(dsr)

    def summary(self) -> dict:
        return {
            "Sharpe"        : self.sharpe,
            "Sortino"       : self.sortino,
            "Calmar"        : self.calmar,
            "Omega"         : self.omega,
            "IR"            : self.ir,
            "CAGR"          : self.cagr,
            "Vol (ann)"     : self.vol * np.sqrt(self.ann),
            "Max DD"        : self.mdd,
            "Skewness"      : self.skew,
            "Excess Kurt"   : self.kurt,
            "VaR(95%)"      : self.var95,
            "CVaR(95%)"     : self.cvar95,
            "VaR(99%)"      : self.var99,
            "CVaR(99%)"     : self.cvar99,
            "Tail Ratio"    : self.tail_ratio,
            "Hit Rate"      : self.hit,
            "Profit Factor" : self.profit_factor,
        }


# =============================================================================
# PRINT HEADER + DATA
# =============================================================================
print()
print("=" * 65)
print("  MODULE 49: PERFORMANCE TEARSHEET")
print("  Sharpe | Sortino | Calmar | Omega | BHB | DSR | Drawdowns")
print("=" * 65)

# Multiple strategies for comparison
TICKERS_STRAT = {
    "SPY (B&H)"  : "SPY",
    "QQQ (B&H)"  : "QQQ",
    "TLT (B&H)"  : "TLT",
    "GLD (B&H)"  : "GLD",
    "60/40"      : None,    # synthetic
}

raw_prices = {}
for label, tk in TICKERS_STRAT.items():
    if tk is not None:
        d = yf.download(tk, start="2015-01-01", end="2023-12-31",
                        auto_adjust=True, progress=False)
        raw_prices[label] = np.log(d["Close"].squeeze().dropna() /
                                   d["Close"].squeeze().dropna().shift(1)).dropna().values

# 60/40 synthetic
min_len = min(len(v) for v in raw_prices.values())
raw_prices["60/40"] = (0.60 * raw_prices["SPY (B&H)"][:min_len] +
                       0.40 * raw_prices["TLT (B&H)"][:min_len])

# Align all to same length
min_len = min(len(v) for v in raw_prices.values())
for k in raw_prices:
    raw_prices[k] = raw_prices[k][:min_len]

spy_ret = raw_prices["SPY (B&H)"]
N = min_len

section(1, f"Strategies loaded: {list(raw_prices.keys())}  T={N} days")

# =============================================================================
# 1.  BUILD PERFORMANCE ENGINES
# =============================================================================
engines = {label: PerformanceEngine(ret, benchmark=spy_ret)
           for label, ret in raw_prices.items()}

print()
print(f"  {'Metric':18s}", end="")
for label in engines:
    print(f"  {label:12s}", end="")
print()
print(f"  {'-'*90}")

key_metrics = ["Sharpe","Sortino","Calmar","Omega","Max DD","CAGR","Vol (ann)",
               "Skewness","CVaR(95%)","Hit Rate"]
for m in key_metrics:
    print(f"  {m:18s}", end="")
    for eng in engines.values():
        v = eng.summary()[m]
        print(f"  {v:+12.4f}", end="")
    print()
print()

# DSR for each strategy
dsr_vals = {label: eng.deflated_sharpe(n_trials=50)
            for label, eng in engines.items()}

section(2, "Performance engines computed for all strategies")

# =============================================================================
# 2.  DRAWDOWN ANALYSIS
# =============================================================================
spy_eng = engines["SPY (B&H)"]
top5_dd = spy_eng.dd_episodes[:5]
section(3, f"SPY top-5 drawdowns extracted  worst MDD={top5_dd[0][3]:.3f}")
for i, (s, tr, e, depth, dur) in enumerate(top5_dd):
    print(f"       DD{i+1}: depth={depth:.3f}  duration={dur}d  "
          f"start_bar={s}  trough_bar={tr}  end_bar={e}")

# =============================================================================
# 3.  BHB ATTRIBUTION (sector decomposition of SPY-like portfolio)
# =============================================================================
# Sector ETFs as proxy for portfolio and benchmark weights
SECTORS = ["XLK","XLF","XLE","XLV","XLI","XLP","XLU","XLB","XLY","XLC"]

sector_data = {}
for tk in SECTORS:
    try:
        d = yf.download(tk, start="2015-01-01", end="2023-12-31",
                        auto_adjust=True, progress=False)
        r = np.log(d["Close"].squeeze().dropna() /
                   d["Close"].squeeze().dropna().shift(1)).dropna().values
        sector_data[tk] = r[:min_len]
    except Exception:
        pass

available_sectors = list(sector_data.keys())
n_sec = len(available_sectors)

# Benchmark weights: equal-weight
w_bench = np.ones(n_sec) / n_sec

# Portfolio weights: overweight tech & healthcare, underweight energy & utilities
w_port  = np.array([0.25 if s in ["XLK","XLV"]
                    else 0.05 if s in ["XLE","XLU"]
                    else 1.0/n_sec
                    for s in available_sectors])
w_port /= w_port.sum()

# Annual sector returns
sec_rets_p = np.array([sector_data[s].mean() * 252 for s in available_sectors])
sec_rets_b = sec_rets_p.copy()   # same universe, different weights

r_alloc   = ((w_port - w_bench) * sec_rets_b).sum()
r_select  = (w_bench * (sec_rets_p - sec_rets_b)).sum()
r_interact= ((w_port - w_bench) * (sec_rets_p - sec_rets_b)).sum()
r_active  = r_alloc + r_select + r_interact

section(4, f"BHB attribution  allocation={r_alloc*100:.2f}%  "
           f"selection={r_select*100:.2f}%  "
           f"interaction={r_interact*100:.2f}%  "
           f"active={r_active*100:.2f}%")

# =============================================================================
# 4.  ROLLING METRICS
# =============================================================================
WIN = 63   # rolling quarter

def rolling_sharpe(ret, window=63, ann=252):
    out = np.full(len(ret), np.nan)
    for i in range(window, len(ret)):
        w = ret[i-window:i]
        out[i] = (w.mean() / (w.std() + 1e-9)) * np.sqrt(ann)
    return out

def rolling_vol(ret, window=21, ann=252):
    out = np.full(len(ret), np.nan)
    for i in range(window, len(ret)):
        out[i] = ret[i-window:i].std() * np.sqrt(ann)
    return out

roll_sr  = rolling_sharpe(spy_ret, WIN)
roll_vol = rolling_vol(spy_ret, 21)
t_       = np.arange(N)

section(5, f"Rolling Sharpe and vol computed  window={WIN}d")

# =============================================================================
# 5.  RETURN DISTRIBUTION ANALYSIS
# =============================================================================
# Test for normality: Jarque-Bera
jb_stat, jb_pval = stats.jarque_bera(spy_ret)
# Fat-tail ratio: kurtosis relative to normal
normal_sim = np.random.normal(spy_ret.mean(), spy_ret.std(), N)

section(6, f"JB normality test: stat={jb_stat:.1f}  p={jb_pval:.4f}  "
           f"(reject normality: {jb_pval < 0.05})")

# =============================================================================
# FIGURE 1: FULL TEARSHEET
# =============================================================================
fig = plt.figure(figsize=(16, 16), facecolor=DARK)
fig.suptitle("Module 49 -- Performance Tearsheet: Multi-Strategy Comparison",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.35, hspace=0.45)

colors_s = [ACCENT, GREEN, AMBER, VIOLET, TEAL]

# 1A: Cumulative returns
ax = fig.add_subplot(gs[0, :])
for (label, ret), col in zip(raw_prices.items(), colors_s):
    eng  = engines[label]
    cum  = (eng.wealth - 1) * 100
    ax.plot(t_, cum, color=col, lw=1.2,
            label=f"{label}  SR={eng.sharpe:.2f}  MDD={eng.mdd:.2f}")
ax.set_title("Cumulative Return (%)")
ax.set_xlabel("Day"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=7, ncol=5); ax.grid(True)

# 1B: Drawdowns
ax = fig.add_subplot(gs[1, 0])
for (label, ret), col in zip(raw_prices.items(), colors_s):
    eng = engines[label]
    ax.fill_between(t_, -eng.dd*100, 0, alpha=0.35, color=col,
                    label=f"{label} MDD={eng.mdd:.2f}")
ax.set_title("Drawdown (%)"); ax.set_xlabel("Day"); ax.set_ylabel("DD (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1C: Rolling Sharpe
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_, roll_sr, color=ACCENT, lw=1.0, label=f"SPY rolling {WIN}d Sharpe")
ax.axhline(0, color=GRID, lw=0.8)
ax.axhline(spy_eng.sharpe, color=GREEN, lw=1.0, ls="--",
           label=f"Full-period SR={spy_eng.sharpe:.2f}")
ax.fill_between(t_, roll_sr, 0, where=roll_sr > 0, alpha=0.3, color=GREEN)
ax.fill_between(t_, roll_sr, 0, where=roll_sr < 0, alpha=0.3, color=RED)
ax.set_title(f"Rolling {WIN}d Sharpe Ratio (SPY)")
ax.set_xlabel("Day"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=7); ax.grid(True)

# 1D: Return distribution
ax = fig.add_subplot(gs[2, 0])
ax.hist(spy_ret*100, bins=80, color=ACCENT, alpha=0.7, density=True,
        edgecolor="none", label="SPY daily returns")
ax.hist(normal_sim*100, bins=80, color=AMBER, alpha=0.4, density=True,
        edgecolor="none", label="Normal(mu,sigma)")
ax.axvline(spy_eng.var95*100, color=RED,   lw=1.2, ls="--",
           label=f"VaR(95%)={spy_eng.var95*100:.2f}%")
ax.axvline(spy_eng.var99*100, color=VIOLET, lw=1.2, ls="--",
           label=f"VaR(99%)={spy_eng.var99*100:.2f}%")
ax.set_title(f"Return Distribution  Skew={spy_eng.skew:.2f}  Kurt={spy_eng.kurt:.2f}")
ax.set_xlabel("Daily return (%)"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

# 1E: Performance radar / bar comparison
ax = fig.add_subplot(gs[2, 1])
metric_names = ["Sharpe", "Sortino", "Calmar", "Omega"]
x_pos = np.arange(len(metric_names))
width = 0.15
for k, ((label, eng), col) in enumerate(zip(engines.items(), colors_s)):
    vals = [eng.summary()[m] for m in metric_names]
    ax.bar(x_pos + k*width, vals, width, color=col, label=label, edgecolor=DARK)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xticks(x_pos + width * len(engines)/2)
ax.set_xticklabels(metric_names, fontsize=8)
ax.set_title("Risk-Adjusted Metrics Comparison")
ax.legend(fontsize=6, ncol=2); ax.grid(True, axis="y")

# 1F: Monthly return heatmap (SPY)
ax = fig.add_subplot(gs[3, :])
mr   = spy_eng.monthly_rets * 100
n_yr = len(mr) // 12
if n_yr > 0:
    heat = mr[:n_yr*12].reshape(n_yr, 12)
    vmax_h = np.percentile(np.abs(heat), 90)
    im   = ax.imshow(heat.T, cmap="RdYlGn", aspect="auto",
                     vmin=-vmax_h, vmax=vmax_h)
    ax.set_yticks(range(12))
    ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=7)
    yr_labels = [str(2015 + i) for i in range(n_yr)]
    ax.set_xticks(range(n_yr))
    ax.set_xticklabels(yr_labels, fontsize=7)
    for i in range(n_yr):
        for j in range(12):
            ax.text(i, j, f"{heat[i,j]:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(heat[i,j]) > vmax_h*0.5 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    cb.set_label("Monthly return (%)", color=TEXT, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT)
ax.set_title("SPY Monthly Return Heatmap (%)")

fig.savefig(os.path.join(FIGS, "m49_fig1_tearsheet.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: DRAWDOWN ANALYSIS + BHB ATTRIBUTION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle("Module 49 -- Drawdown Analysis & BHB Attribution",
             fontsize=12, color=TEXT, y=1.01)

# 2A: Top-5 drawdown timeline
ax = axes[0, 0]
ax.plot(t_, (spy_eng.wealth - 1)*100, color=ACCENT, lw=1.0, alpha=0.7)
colors_dd = [RED, AMBER, VIOLET, TEAL, GREEN]
for (s, tr, e, depth, dur), col in zip(top5_dd, colors_dd):
    ax.axvspan(s, e, alpha=0.25, color=col,
               label=f"DD={depth:.2f} ({dur}d)")
ax.set_title("SPY Top-5 Drawdown Periods")
ax.set_xlabel("Day"); ax.set_ylabel("Cumulative return (%)")
ax.legend(fontsize=6); ax.grid(True)

# 2B: Drawdown depth vs duration scatter
ax = axes[0, 1]
if spy_eng.dd_episodes:
    depths = [e[3]*100 for e in spy_eng.dd_episodes]
    durs   = [e[4]     for e in spy_eng.dd_episodes]
    ax.scatter(durs, depths, color=RED, s=20, alpha=0.6)
    ax.set_xlabel("Duration (days)"); ax.set_ylabel("Depth (%)")
    ax.set_title("Drawdown: Depth vs Duration")
    ax.grid(True)

# 2C: BHB attribution bar
ax = axes[1, 0]
bhb_vals   = [r_alloc*100, r_select*100, r_interact*100, r_active*100]
bhb_labels = ["Allocation", "Selection", "Interaction", "Total Active"]
bhb_colors = [ACCENT, GREEN, AMBER, VIOLET]
bars = ax.bar(bhb_labels, bhb_vals, color=bhb_colors, edgecolor=DARK, width=0.5)
ax.axhline(0, color=GRID, lw=0.8)
for bar, v in zip(bars, bhb_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v + np.sign(v)*0.02,
            f"{v:.2f}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8, color=TEXT)
ax.set_title("BHB Attribution (sector overweights)")
ax.set_ylabel("Active return (%)"); ax.grid(True, axis="y")

# 2D: Sector weight comparison
ax = axes[1, 1]
x_sec = np.arange(n_sec)
width_s = 0.35
ax.bar(x_sec - width_s/2, w_bench*100, width_s, color=ACCENT,
       label="Benchmark (equal)", edgecolor=DARK, alpha=0.8)
ax.bar(x_sec + width_s/2, w_port*100, width_s, color=GREEN,
       label="Portfolio", edgecolor=DARK, alpha=0.8)
ax.set_xticks(x_sec)
ax.set_xticklabels(available_sectors, rotation=45, ha="right", fontsize=7)
ax.set_title("Sector Weights: Portfolio vs Benchmark")
ax.set_ylabel("Weight (%)"); ax.legend(fontsize=7); ax.grid(True, axis="y")

for ax in axes.flat:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m49_fig2_drawdown_attribution.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: DSR + ROLLING VOL + QQ PLOT
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("Module 49 -- Statistical Tests & Risk Metrics",
             fontsize=12, color=TEXT, y=1.01)

# 3A: DSR bar chart
ax = axes[0]
dsr_names = list(dsr_vals.keys())
dsr_v     = list(dsr_vals.values())
bar_col   = [GREEN if v > 0.95 else AMBER if v > 0.80 else RED for v in dsr_v]
bars = ax.bar(range(len(dsr_names)), dsr_v, color=bar_col, edgecolor=DARK, width=0.5)
ax.axhline(0.95, color=GREEN, lw=1.0, ls="--", label="DSR=0.95 threshold")
ax.axhline(0.80, color=AMBER, lw=1.0, ls="--", alpha=0.7, label="DSR=0.80")
for bar, v in zip(bars, dsr_v):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_xticks(range(len(dsr_names)))
ax.set_xticklabels([n.replace(" ","\\n") for n in dsr_names], fontsize=7)
ax.set_ylim(0, 1.1)
ax.set_title("Deflated Sharpe Ratio (n_trials=50)")
ax.set_ylabel("DSR"); ax.legend(fontsize=7); ax.grid(True, axis="y")

# 3B: Rolling vol
ax = axes[1]
ax.plot(t_, roll_vol*100, color=ACCENT, lw=0.8, alpha=0.8, label="SPY 21d vol")
ax.axhline(spy_eng.vol * np.sqrt(252) * 100, color=AMBER, lw=1.2, ls="--",
           label=f"Full-period vol={spy_eng.vol*np.sqrt(252)*100:.1f}%")
ax.fill_between(t_, roll_vol*100, 0, alpha=0.2, color=ACCENT)
ax.set_title("Rolling 21-Day Annualised Volatility (%)")
ax.set_xlabel("Day"); ax.set_ylabel("Vol (%)"); ax.legend(fontsize=7); ax.grid(True)

# 3C: QQ plot vs normal
ax = axes[2]
(osm, osr), (slope, intercept, r) = stats.probplot(spy_ret, dist="norm")
ax.scatter(osm, osr, color=ACCENT, s=4, alpha=0.5, label="SPY returns")
ax.plot(osm, slope*np.array(osm)+intercept, color=AMBER, lw=1.5,
        label=f"Normal line  R={r:.3f}")
ax.set_title(f"QQ Plot vs Normal\nJB stat={jb_stat:.0f}  p={jb_pval:.4f}")
ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
ax.legend(fontsize=7); ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m49_fig3_dsr_vol_qq.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
spy_s = spy_eng.summary()
print()
print("  MODULE 49 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Sharpe = mean(r-rf)/std(r) * sqrt(252)  -- symmetric risk")
print("  [2] Sortino = mean(r-rf)/sigma_d * sqrt(252) -- downside only")
print("  [3] Calmar = CAGR / MDD  -- return per unit of worst loss")
print("  [4] Omega > 1: gains above L exceed losses below L")
print("  [5] BHB: active_return = allocation + selection + interaction")
print("  [6] DSR corrects Sharpe for selection bias across multiple tests")
print(f"  [7] SPY: SR={spy_s['Sharpe']:.3f}  Sortino={spy_s['Sortino']:.3f}  "
      f"Calmar={spy_s['Calmar']:.3f}  MDD={spy_s['Max DD']:.3f}")
print(f"  [8] JB test: stat={jb_stat:.0f} p={jb_pval:.4f}  "
      f"fat tails confirmed (kurt={spy_s['Excess Kurt']:.2f})")
print(f"  NEXT: M50 -- Pairs Trading: Cointegration & Mean Reversion")
print()

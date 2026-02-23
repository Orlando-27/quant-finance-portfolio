#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 48 -- FACTOR IC, SIGNAL DECAY & ALPHALENS TEARSHEETS
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Factor analysis evaluates the predictive power of cross-sectional signals
(alpha factors) for future asset returns.  The core framework, popularised
by Quantopian's Alphalens library, decomposes factor evaluation into:

  1. Information Coefficient (IC)
  2. Factor Return (quantile spread)
  3. Turnover & Autocorrelation
  4. Signal Decay

INFORMATION COEFFICIENT (IC)
------------------------------
The IC measures the rank correlation between factor values and forward returns:

    IC_t = Spearman_rho(f_{i,t}, r_{i,t+h})

where f_{i,t} is the factor value for asset i at time t and r_{i,t+h} is
the h-period forward return.  Spearman rank correlation is used to be robust
to outliers and non-linear relationships.

Key benchmarks:
    |IC| > 0.02   : weak but usable signal
    |IC| > 0.05   : moderate, economically significant
    |IC| > 0.10   : strong (rare in practice)
    ICIR = mean(IC) / std(IC)  -- analogous to Sharpe of the signal

INFORMATION RATIO (IR) AND T-STATISTIC
----------------------------------------
    IR   = mean(IC) / std(IC) * sqrt(T)
    t    = mean(IC) / (std(IC) / sqrt(T))
    t > 2  =>  IC is statistically significant at 5% level

FACTOR RETURN: QUANTILE ANALYSIS
----------------------------------
Rank assets into Q quantiles by factor value at each rebalance date.
Compute the mean forward return of each quantile bucket:

    R_q(h) = mean_{i in Q_q, t} r_{i,t+h}

The spread R_Q - R_1 (top minus bottom quantile) is the factor return.
A monotone R_q vs q relationship validates the factor.

SIGNAL DECAY
------------
IC decays as the prediction horizon h increases:

    IC(h) = Spearman_rho(f_{i,t}, r_{i,t+h})   for h = 1, 2, ..., H

The decay profile reveals the optimal holding period.  Fast decay implies
short-horizon alpha; slow decay implies a structural (persistent) factor.

TURNOVER & FACTOR AUTOCORRELATION
-----------------------------------
Factor autocorrelation: AC_h = corr(f_{i,t}, f_{i,t-h})
High AC => slow-moving factor => low turnover => better net-of-cost returns.

Rank change turnover: fraction of assets that change quantile each period.

FACTOR ZOO
----------
We implement five canonical factors on a cross-section of US ETFs:
  1. Momentum (12-1 month): past return excluding last month
  2. Short-term reversal (1-month): past 1-month return (negative signal)
  3. Volatility (1-month realised vol): low-vol anomaly
  4. RSI mean-reversion: overbought/oversold signal
  5. Volume trend: price-volume divergence

REFERENCES
----------
[1] Fama, E. & French, K. (1993). "Common risk factors." JFE 33(1):3-56.
[2] Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners." JF 48.
[3] Ang, A. et al. (2006). "The Cross-Section of Volatility." JF 61(1).
[4] Hou, K., Xue, C. & Zhang, L. (2020). "Replicating Anomalies." RFS.
[5] Quantopian (2016). "Alphalens: A Performance Analysis Library."
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m48_alphalens/m48_alphalens.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m48")
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
# PRINT HEADER
# =============================================================================
print()
print("=" * 65)
print("  MODULE 48: FACTOR IC, SIGNAL DECAY & TEARSHEETS")
print("  IC | ICIR | Quantile Returns | Decay | Turnover | Zoo")
print("=" * 65)

# =============================================================================
# 1.  DATA: CROSS-SECTION OF US SECTOR ETFs
# =============================================================================
TICKERS = ["XLK","XLF","XLE","XLV","XLI","XLP","XLU","XLB","XLY","XLC",
           "SPY","QQQ","IWM","EFA","EEM","GLD","TLT","HYG","LQD","VNQ"]

raw_data = {}
for tk in TICKERS:
    try:
        d = yf.download(tk, start="2015-01-01", end="2023-12-31",
                        auto_adjust=True, progress=False)
        if len(d) > 500:
            raw_data[tk] = d["Close"].squeeze()
    except Exception:
        pass

tickers = list(raw_data.keys())
import pandas as pd
prices  = pd.DataFrame(raw_data).dropna()
ret_df  = np.log(prices / prices.shift(1)).dropna()
dates   = ret_df.index
T, N_assets = ret_df.shape

section(1, f"Cross-section: {N_assets} assets  {T} days  "
           f"[{dates[0].date()} -- {dates[-1].date()}]")

# =============================================================================
# 2.  FACTOR CONSTRUCTION (monthly rebalance)
# =============================================================================
# Resample to monthly frequency
prices_m = prices.resample("ME").last()
ret_m    = np.log(prices_m / prices_m.shift(1)).dropna()
T_m      = len(ret_m)
dates_m  = ret_m.index
P_arr    = prices_m.values           # (T_m+1, N_assets)
R_arr    = ret_m.values              # (T_m, N_assets)

def rank_normalise(x: np.ndarray) -> np.ndarray:
    """
    Cross-sectionally rank-normalise factor values to [-1, +1].
    Handles NaN by assigning rank 0.
    """
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        row = x[i]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        ranks = np.zeros(len(row))
        ranks[valid] = np.argsort(np.argsort(row[valid])).astype(float)
        n_v = valid.sum()
        ranks[valid] = (ranks[valid] / (n_v - 1)) * 2 - 1  # [-1, +1]
        ranks[~valid] = 0.0
        out[i] = ranks
    return out

# Factor 1: Momentum 12-1  (return from t-12 to t-1)
MOM_LONG  = 12
MOM_SHORT = 1
mom = np.full((T_m, N_assets), np.nan)
P_full = prices_m.values   # (T_m+1, N_assets) after resample+dropna shift
for t in range(MOM_LONG, T_m):
    mom[t] = P_arr[t] / P_arr[t - MOM_LONG + MOM_SHORT] - 1

# Factor 2: Short-term reversal (negative 1-month return)
rev = np.full((T_m, N_assets), np.nan)
for t in range(1, T_m):
    rev[t] = -(P_arr[t] / P_arr[t-1] - 1)

# Factor 3: Low Volatility (negative realised vol over 3 months)
VOL_WIN = 3
ret_daily = ret_df.values
# Map monthly dates to daily indices
vol_factor = np.full((T_m, N_assets), np.nan)
for t, dt in enumerate(dates_m):
    loc = dates.searchsorted(dt, side="right") - 1; mask = (dates <= dt) & (dates > dates[max(0, loc - 63)])
    window = ret_daily[mask]
    if len(window) > 5:
        vol_factor[t] = -window.std(axis=0) * np.sqrt(252)

# Factor 4: RSI mean-reversion (14-day RSI mapped to signal)
def rsi_series(price_series: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI for a single asset price series."""
    delta = np.diff(price_series)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = np.convolve(gain, np.ones(period)/period, "valid")
    avg_l = np.convolve(loss, np.ones(period)/period, "valid")
    rs    = avg_g / (avg_l + 1e-9)
    return 100 - 100 / (1 + rs)

rsi_monthly = np.full((T_m, N_assets), np.nan)
for j, tk in enumerate(tickers):
    px  = prices[tk].values
    rsi = rsi_series(px, 14)
    for t, dt in enumerate(dates_m):
        loc = dates.get_loc(dt) if dt in dates else -1
        if isinstance(loc, int) and loc >= 14:
            idx = loc - len(px) + len(rsi)
            if 0 <= idx < len(rsi):
                rsi_monthly[t, j] = 50 - rsi[idx]  # mean-reversion: buy oversold

# Factor 5: Price-to-52w-high (mean reversion from peak)
p52_factor = np.full((T_m, N_assets), np.nan)
for t in range(12, T_m):
    high52 = P_arr[max(0, t-12):t].max(axis=0)
    p52_factor[t] = -(P_arr[t] / (high52 + 1e-9) - 1)  # discount from peak

# Rank-normalise all factors
factors = {
    "Momentum(12-1)"  : rank_normalise(mom),
    "Reversal(1m)"    : rank_normalise(rev),
    "LowVol(3m)"      : rank_normalise(vol_factor),
    "RSI_MeanRev"     : rank_normalise(rsi_monthly),
    "P52W_Discount"   : rank_normalise(p52_factor),
}

section(2, f"Factors constructed: {list(factors.keys())}")

# =============================================================================
# 3.  IC CALCULATION
# =============================================================================
MAX_HORIZON = 6   # months forward

def compute_ic(factor: np.ndarray, returns: np.ndarray,
               horizon: int = 1) -> np.ndarray:
    """
    Monthly cross-sectional IC at given forward horizon.

    IC_t = Spearman_rho(f_{:,t}, r_{:,t+h})
    """
    T_, N_ = factor.shape
    ic_series = []
    for t in range(T_ - horizon):
        f_t = factor[t]
        r_t = returns[t+1:t+horizon+1].sum(axis=0) if horizon > 1 else returns[t+1]
        valid = ~(np.isnan(f_t) | np.isnan(r_t))
        if valid.sum() < 5:
            ic_series.append(np.nan)
            continue
        ic, _ = spearmanr(f_t[valid], r_t[valid])
        ic_series.append(ic)
    return np.array(ic_series)

ic_results = {}
for fname, fmat in factors.items():
    ic_h = []
    for h in range(1, MAX_HORIZON + 1):
        ic_s = compute_ic(fmat, R_arr, h)
        ic_h.append(ic_s)
    ic_results[fname] = ic_h   # list of H arrays

# IC summary statistics
print()
print(f"  {'Factor':20s}  {'IC(1m)':>8s}  {'ICIR':>8s}  {'t-stat':>8s}  "
      f"{'IC(3m)':>8s}  {'IC(6m)':>8s}")
print(f"  {'-'*70}")
ic_summary = {}
for fname, ic_h in ic_results.items():
    ic1    = ic_h[0][~np.isnan(ic_h[0])]
    ic3    = ic_h[2][~np.isnan(ic_h[2])]
    ic6    = ic_h[5][~np.isnan(ic_h[5])]
    mean1  = ic1.mean()
    icir   = mean1 / (ic1.std() + 1e-9)
    t_stat = mean1 / (ic1.std() / np.sqrt(len(ic1)) + 1e-9)
    print(f"  {fname:20s}  {mean1:+8.4f}  {icir:+8.4f}  {t_stat:+8.3f}  "
          f"{ic3.mean():+8.4f}  {ic6.mean():+8.4f}")
    ic_summary[fname] = {"mean": mean1, "icir": icir, "tstat": t_stat,
                         "ic3": ic3.mean(), "ic6": ic6.mean()}

section(3, "IC computed for all factors at horizons 1-6m")

# =============================================================================
# 4.  QUANTILE RETURN ANALYSIS
# =============================================================================
N_QUANTILES = 5
HORIZON_Q   = 1   # 1-month forward

def quantile_returns(factor: np.ndarray, returns: np.ndarray,
                     n_q: int = 5) -> np.ndarray:
    """
    Mean forward return by factor quantile, averaged across time.

    Returns array of shape (n_q,) -- one mean return per quantile.
    """
    T_, N_ = factor.shape
    q_rets  = [[] for _ in range(n_q)]
    for t in range(T_ - 1):
        f_t = factor[t]
        r_t = returns[t + 1]
        valid = ~(np.isnan(f_t) | np.isnan(r_t))
        if valid.sum() < n_q:
            continue
        fv = f_t[valid]; rv = r_t[valid]
        edges = np.percentile(fv, np.linspace(0, 100, n_q + 1))
        for q in range(n_q):
            mask_q = (fv >= edges[q]) & (fv <= edges[q+1])
            if mask_q.sum() > 0:
                q_rets[q].append(rv[mask_q].mean())
    return np.array([np.mean(qr) if qr else 0.0 for qr in q_rets])

qret_results = {}
for fname, fmat in factors.items():
    qret_results[fname] = quantile_returns(fmat, R_arr, N_QUANTILES)

section(4, f"Quantile returns computed ({N_QUANTILES} quantiles, h={HORIZON_Q}m)")

# =============================================================================
# 5.  SIGNAL DECAY PROFILE
# =============================================================================
decay_results = {}
for fname, ic_h in ic_results.items():
    decay = [np.nanmean(ic_h[h]) for h in range(MAX_HORIZON)]
    decay_results[fname] = decay

section(5, f"Signal decay profiles computed for h=1..{MAX_HORIZON}m")

# =============================================================================
# 6.  FACTOR AUTOCORRELATION & TURNOVER
# =============================================================================
def factor_autocorr(factor: np.ndarray, lag: int = 1) -> float:
    """Cross-sectional mean of time-series autocorrelation."""
    ac_list = []
    for j in range(factor.shape[1]):
        f_j = factor[:, j]
        valid = ~np.isnan(f_j)
        if valid.sum() < lag + 5:
            continue
        f_j  = f_j[valid]
        ac   = np.corrcoef(f_j[lag:], f_j[:-lag])[0, 1]
        if not np.isnan(ac):
            ac_list.append(ac)
    return float(np.mean(ac_list)) if ac_list else 0.0

def quantile_turnover(factor: np.ndarray, n_q: int = 5) -> float:
    """
    Mean fraction of assets changing quantile from t to t+1.
    High turnover => expensive factor.
    """
    T_, N_ = factor.shape
    turns = []
    for t in range(T_ - 1):
        f0 = factor[t];   f1 = factor[t+1]
        valid = ~(np.isnan(f0) | np.isnan(f1))
        if valid.sum() < n_q:
            continue
        fv0 = f0[valid]; fv1 = f1[valid]
        edges0 = np.percentile(fv0, np.linspace(0, 100, n_q+1))
        edges1 = np.percentile(fv1, np.linspace(0, 100, n_q+1))
        q0 = np.digitize(fv0, edges0[1:-1])
        q1 = np.digitize(fv1, edges1[1:-1])
        turns.append((q0 != q1).mean())
    return float(np.mean(turns)) if turns else 0.5

ac_turnover = {}
for fname, fmat in factors.items():
    ac  = factor_autocorr(fmat, lag=1)
    to  = quantile_turnover(fmat, N_QUANTILES)
    ac_turnover[fname] = {"ac": ac, "to": to}

section(6, "Factor autocorrelation and turnover computed")
for fname, d in ac_turnover.items():
    print(f"       {fname:20s}  AC={d['ac']:+.3f}  Turnover={d['to']:.3f}")

# =============================================================================
# 7.  COMBINED FACTOR (equal-weight IC-weighted combination)
# =============================================================================
# IC weights: sign(mean_IC) * |ICIR| / sum(|ICIR|)
icir_vals = np.array([ic_summary[f]["icir"] for f in factors])
ic_weights = np.sign(icir_vals) * np.abs(icir_vals) / (np.abs(icir_vals).sum() + 1e-9)

combined = np.zeros_like(list(factors.values())[0])
for w, fmat in zip(ic_weights, factors.values()):
    combined += w * np.nan_to_num(fmat)

combined_ic = compute_ic(combined, R_arr, 1)
combined_ic_mean = np.nanmean(combined_ic)
combined_icir    = combined_ic_mean / (np.nanstd(combined_ic) + 1e-9)

section(7, f"Combined factor  IC={combined_ic_mean:.4f}  ICIR={combined_icir:.3f}  "
           f"weights={dict(zip(factors.keys(), ic_weights.round(3)))}")

# =============================================================================
# 8.  CUMULATIVE IC P&L (IC-weighted long-short portfolio)
# =============================================================================
cum_factor_ret = {}
for fname, fmat in factors.items():
    port_rets = []
    for t in range(T_m - 1):
        f_t = np.nan_to_num(fmat[t])
        # Long top half, short bottom half -- IC-sign adjusted
        sign = np.sign(ic_summary[fname]["mean"])
        f_t  = f_t * sign
        pos  = np.where(f_t > 0, f_t, 0) - np.where(f_t < 0, -f_t, 0)
        pos /= (np.abs(pos).sum() + 1e-9)   # dollar-neutral
        port_rets.append((pos * R_arr[t+1]).sum())
    cum_factor_ret[fname] = np.cumprod(1 + np.array(port_rets)) - 1

section(8, "Cumulative IC-weighted L/S factor returns computed")

# =============================================================================
# FIGURE 1: IC TEARSHEET
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle("Module 48 -- Factor IC Tearsheet",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.38, hspace=0.45)

colors_f = [ACCENT, GREEN, AMBER, VIOLET, TEAL]

# 1A: Rolling IC (1m horizon) for all factors
ax = fig.add_subplot(gs[0, :])
t_m = np.arange(len(list(ic_results.values())[0][0]))
for (fname, ic_h), col in zip(ic_results.items(), colors_f):
    ic1 = ic_h[0]
    ic1_smooth = np.convolve(np.nan_to_num(ic1), np.ones(3)/3, "same")
    ax.plot(t_m, ic1_smooth, color=col, lw=1.0, alpha=0.8, label=fname)
ax.axhline(0, color=GRID, lw=0.8)
ax.axhline( 0.05, color=TEXT, lw=0.6, ls="--", alpha=0.4)
ax.axhline(-0.05, color=TEXT, lw=0.6, ls="--", alpha=0.4)
ax.set_title("Rolling Monthly IC (1m horizon, 3m smoothed)")
ax.set_xlabel("Month"); ax.set_ylabel("IC")
ax.legend(fontsize=7, ncol=5); ax.grid(True)

# 1B: IC distribution
ax = fig.add_subplot(gs[1, 0])
for (fname, ic_h), col in zip(ic_results.items(), colors_f):
    ic1 = ic_h[0][~np.isnan(ic_h[0])]
    ax.hist(ic1, bins=20, alpha=0.5, color=col, density=True, label=fname)
ax.axvline(0, color=GRID, lw=0.8)
ax.set_title("IC Distribution by Factor"); ax.set_xlabel("IC"); ax.set_ylabel("Density")
ax.legend(fontsize=6); ax.grid(True)

# 1C: ICIR bar chart
ax = fig.add_subplot(gs[1, 1])
names_f = list(ic_summary.keys())
icirs_v = [ic_summary[f]["icir"] for f in names_f]
tstats  = [ic_summary[f]["tstat"] for f in names_f]
bar_col = [GREEN if v > 0 else RED for v in icirs_v]
bars    = ax.bar(range(len(names_f)), icirs_v, color=bar_col, edgecolor=DARK)
ax.axhline(0,  color=GRID, lw=0.8)
ax.axhline( 0.5, color=AMBER, lw=0.8, ls="--", alpha=0.7, label="ICIR=0.5")
ax.axhline(-0.5, color=AMBER, lw=0.8, ls="--", alpha=0.7)
for bar, v in zip(bars, icirs_v):
    ax.text(bar.get_x()+bar.get_width()/2, v + np.sign(v)*0.01,
            f"{v:.2f}", ha="center", va="bottom" if v>0 else "top",
            fontsize=7, color=TEXT)
ax.set_xticks(range(len(names_f)))
ax.set_xticklabels([n.replace("(","\\n(") for n in names_f], fontsize=7)
ax.set_title("ICIR by Factor"); ax.set_ylabel("ICIR"); ax.legend(fontsize=7)
ax.grid(True, axis="y")

# 1D: Signal decay
ax = fig.add_subplot(gs[2, 0])
horizons = np.arange(1, MAX_HORIZON + 1)
for (fname, decay), col in zip(decay_results.items(), colors_f):
    ax.plot(horizons, decay, color=col, lw=1.5, marker="o", ms=4, label=fname)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Signal Decay: IC vs Forward Horizon")
ax.set_xlabel("Forward horizon (months)"); ax.set_ylabel("Mean IC")
ax.legend(fontsize=7); ax.grid(True)

# 1E: Combined factor IC
ax = fig.add_subplot(gs[2, 1])
cic_smooth = np.convolve(np.nan_to_num(combined_ic), np.ones(3)/3, "same")
ax.fill_between(t_m, cic_smooth, 0, where=cic_smooth > 0, alpha=0.5, color=GREEN)
ax.fill_between(t_m, cic_smooth, 0, where=cic_smooth < 0, alpha=0.5, color=RED)
ax.axhline(combined_ic_mean, color=AMBER, lw=1.2, ls="--",
           label=f"mean IC={combined_ic_mean:.4f}  ICIR={combined_icir:.3f}")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Combined Factor IC (IC-weighted)")
ax.set_xlabel("Month"); ax.set_ylabel("IC")
ax.legend(fontsize=7); ax.grid(True)

fig.savefig(os.path.join(FIGS, "m48_fig1_ic_tearsheet.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: QUANTILE RETURNS + TURNOVER + L/S PERFORMANCE
# =============================================================================
fig = plt.figure(figsize=(16, 10), facecolor=DARK)
fig.suptitle("Module 48 -- Quantile Returns, Turnover & L/S Performance",
             fontsize=12, color=TEXT, y=0.99)
gs2 = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)

# 2A-C: Quantile return bar for top 3 factors by |ICIR|
top3 = sorted(ic_summary.keys(), key=lambda f: abs(ic_summary[f]["icir"]), reverse=True)[:3]
for col_idx, fname in enumerate(top3):
    ax = fig.add_subplot(gs2[0, col_idx])
    qr = qret_results[fname] * 100   # to %
    bar_colors = [RED] * 2 + [AMBER] + [GREEN] * 2
    bars = ax.bar(range(1, N_QUANTILES+1), qr, color=bar_colors, edgecolor=DARK)
    for bar, v in zip(bars, qr):
        ax.text(bar.get_x()+bar.get_width()/2, v + np.sign(v)*0.001,
                f"{v:.2f}%", ha="center",
                va="bottom" if v > 0 else "top", fontsize=7, color=TEXT)
    ax.axhline(0, color=GRID, lw=0.8)
    spread = qr[-1] - qr[0]
    ax.set_title(f"{fname}\nSpread={spread:.2f}%/m")
    ax.set_xlabel("Quantile"); ax.set_ylabel("Mean monthly return (%)")
    ax.grid(True, axis="y")

# 2D: Autocorrelation & Turnover scatter
ax = fig.add_subplot(gs2[1, 0])
for (fname, d), col in zip(ac_turnover.items(), colors_f):
    ax.scatter(d["to"], d["ac"], color=col, s=80, zorder=4, label=fname)
    ax.annotate(fname.split("(")[0], (d["to"], d["ac"]),
                fontsize=6, color=TEXT, xytext=(4, 4),
                textcoords="offset points")
ax.set_xlabel("Turnover (fraction)"); ax.set_ylabel("Factor AC (lag-1)")
ax.set_title("Turnover vs Autocorrelation")
ax.legend(fontsize=6); ax.grid(True)

# 2E: Cumulative L/S factor returns
ax = fig.add_subplot(gs2[1, 1:])
t_ls = np.arange(len(list(cum_factor_ret.values())[0]))
for (fname, cr), col in zip(cum_factor_ret.items(), colors_f):
    sr = np.mean(np.diff(np.log(cr + 1 + 1e-9))) / \
         (np.std(np.diff(np.log(cr + 1 + 1e-9))) + 1e-9) * np.sqrt(12)
    ax.plot(t_ls, cr * 100, color=col, lw=1.2,
            label=f"{fname}  SR={sr:.2f}")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Cumulative L/S Factor Return (%)")
ax.set_xlabel("Month"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=6); ax.grid(True)

fig.savefig(os.path.join(FIGS, "m48_fig2_quantile_turnover_ls.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: SUMMARY HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
ax.set_facecolor(PANEL)
fig.suptitle("Module 48 -- Factor Summary Heatmap",
             fontsize=12, color=TEXT)

metrics  = ["IC(1m)", "ICIR", "t-stat", "IC(3m)", "IC(6m)", "AC", "Turnover"]
fn_list  = list(factors.keys())
heat_mat = np.zeros((len(fn_list), len(metrics)))
for i, fname in enumerate(fn_list):
    s = ic_summary[fname]
    d = ac_turnover[fname]
    heat_mat[i] = [s["mean"], s["icir"], s["tstat"],
                   s["ic3"], s["ic6"], d["ac"], d["to"]]

# Normalise each metric for colour scale
heat_norm = heat_mat.copy()
for j in range(heat_mat.shape[1]):
    col_j = heat_mat[:, j]
    rng   = col_j.max() - col_j.min()
    if rng > 1e-9:
        heat_norm[:, j] = (col_j - col_j.min()) / rng
    # Invert turnover (lower is better)
    if metrics[j] == "Turnover":
        heat_norm[:, j] = 1 - heat_norm[:, j]

im = ax.imshow(heat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=9)
ax.set_yticks(range(len(fn_list))); ax.set_yticklabels(fn_list, fontsize=9)

for i in range(len(fn_list)):
    for j in range(len(metrics)):
        v = heat_mat[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8, color="black")

cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.set_label("Normalised score (green=better)", color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)
ax.set_title("Factor Quality Heatmap (normalised per metric)")

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m48_fig3_summary_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
best_factor = max(ic_summary, key=lambda f: abs(ic_summary[f]["icir"]))
print()
print("  MODULE 48 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] IC = Spearman_rho(factor_t, return_{t+h}) -- rank correlation")
print("  [2] ICIR = mean(IC)/std(IC) -- Sharpe ratio of the signal")
print("  [3] t-stat = mean(IC)/(std(IC)/sqrt(T)) > 2 => significant")
print("  [4] Quantile spread: monotone R_q vs q validates factor")
print("  [5] Signal decay: IC(h) vs h reveals optimal holding period")
print("  [6] Factor AC measures persistence; high AC => low turnover")
print(f"  [7] Best factor: {best_factor}  "
      f"ICIR={ic_summary[best_factor]['icir']:.3f}  "
      f"t={ic_summary[best_factor]['tstat']:.2f}")
print(f"  [8] Combined factor ICIR={combined_icir:.3f}  "
      f"IC={combined_ic_mean:.4f}")
print(f"  NEXT: M49 -- Sharpe, Sortino, Drawdowns & Performance Attribution")
print()

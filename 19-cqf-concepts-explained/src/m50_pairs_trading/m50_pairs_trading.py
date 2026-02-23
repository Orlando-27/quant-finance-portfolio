#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 50 -- PAIRS TRADING: COINTEGRATION & MEAN REVERSION
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Pairs trading exploits the long-run equilibrium relationship between two
economically linked assets.  When the spread deviates from equilibrium, the
strategy bets on its reversion.

COINTEGRATION
-------------
Two I(1) time series P_A(t) and P_B(t) are cointegrated if there exists a
vector beta = (1, -b) such that:

    S(t) = P_A(t) - b * P_B(t)  ~  I(0)   (stationary spread)

The cointegrating vector (1, -b) is estimated by OLS:
    P_A(t) = alpha + b * P_B(t) + epsilon(t)

ENGLE-GRANGER TWO-STEP PROCEDURE
----------------------------------
Step 1: Estimate b by OLS regression of P_A on P_B.
Step 2: Apply ADF test to residuals epsilon(t) = P_A - alpha - b*P_B.
        Reject unit root => series are cointegrated.

ADF TEST STATISTIC
------------------
    H_0: epsilon has a unit root (non-stationary, random walk)
    H_1: epsilon is stationary (mean-reverting)

ADF stat = t-statistic on rho in:
    Delta(epsilon_t) = rho * epsilon_{t-1} + sum phi_k Delta(epsilon_{t-k}) + u_t

Reject H_0 (cointegration confirmed) if ADF stat < critical value:
    1%: -3.43    5%: -2.86    10%: -2.57

ORNSTEIN-UHLENBECK SPREAD MODEL
---------------------------------
A stationary spread is modelled as an OU process:

    dS = kappa * (mu - S) * dt + sigma * dW

Parameters estimated by discrete-time regression:
    S_t - S_{t-1} = a + b * S_{t-1} + epsilon_t
    kappa = -log(1 + b) / dt  (mean reversion speed)
    mu    = -a / b             (long-run mean)
    sigma = std(epsilon) / sqrt(dt)

Half-life of mean reversion:
    t_{1/2} = log(2) / kappa

TRADING SIGNAL: Z-SCORE
-----------------------
    z_t = (S_t - mu_S) / sigma_S

    Entry long  spread: z_t < -entry_z   (spread below mean)
    Entry short spread: z_t > +entry_z   (spread above mean)
    Exit:               |z_t| < exit_z

DOLLAR-NEUTRAL POSITION
------------------------
Long beta shares of asset B for every 1 share of asset A:
    P&L = (S_t+1 - S_t) * signal_t
        = Delta(P_A) - b * Delta(P_B)  per unit notional

REFERENCES
----------
[1] Engle, R. & Granger, C. (1987). "Co-integration and Error Correction."
    Econometrica 55(2):251-276.
[2] Gatev, E., Goetzmann, W. & Rouwenhorst, K. (2006). "Pairs Trading."
    RFS 19(3):797-827.
[3] Elliott, R., van der Hoek, J. & Malcolm, W. (2005). "Pairs Trading."
    Quantitative Finance 5(3):271-276.
[4] Avellaneda, M. & Lee, J. (2010). "Statistical Arbitrage in the US
    Equities Market." Quantitative Finance 10(7):761-782.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m50_pairs_trading/m50_pairs_trading.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m50")
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
# STATISTICAL TOOLS (numpy-only)
# =============================================================================
def ols(y: np.ndarray, x: np.ndarray):
    """
    OLS regression y = alpha + beta * x.
    Returns (alpha, beta, residuals, R2).
    """
    X   = np.column_stack([np.ones(len(x)), x])
    b   = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    ss_res = (res**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2  = 1 - ss_res / (ss_tot + 1e-9)
    return b[0], b[1], res, r2

def adf_test(series: np.ndarray, max_lags: int = 1) -> tuple:
    """
    Augmented Dickey-Fuller test (numpy implementation).

    H_0: unit root (non-stationary)
    H_1: stationary

    Returns (adf_stat, n_lags_used)
    ADF critical values (approx, T>100):
        1%: -3.43   5%: -2.86   10%: -2.57
    """
    y    = series.copy()
    dy   = np.diff(y)
    n    = len(dy)

    # Determine lag by minimising AIC
    best_aic = np.inf
    best_lag = 0
    for lag in range(0, max_lags + 1):
        if lag == 0:
            X = y[:-1].reshape(-1, 1)
            X = np.column_stack([X, np.ones(len(X))])
            Y = dy
        else:
            lpad = lag
            X_lev = y[lpad:-1].reshape(-1, 1)
            X_lag = np.column_stack([np.diff(y)[k:n-lpad+k]
                                     for k in range(lag)])
            X = np.column_stack([X_lev, X_lag, np.ones(len(X_lev))])
            Y = dy[lpad:]
        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid  = Y - X @ coeffs
            aic    = len(Y) * np.log((resid**2).mean() + 1e-12) + 2 * X.shape[1]
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        except Exception:
            continue

    # Final regression with best lag
    lag = best_lag
    if lag == 0:
        X = y[:-1].reshape(-1, 1)
        X = np.column_stack([X, np.ones(len(X))])
        Y = dy
    else:
        lpad = lag
        X_lev = y[lpad:-1].reshape(-1, 1)
        X_lag = np.column_stack([np.diff(y)[k:n-lpad+k]
                                 for k in range(lag)])
        X = np.column_stack([X_lev, X_lag, np.ones(len(X_lev))])
        Y = dy[lpad:]

    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid  = Y - X @ coeffs
    rho    = coeffs[0]
    sigma2 = (resid**2).sum() / (len(Y) - X.shape[1])
    # Standard error of rho via OLS formula
    xtx_inv = np.linalg.pinv(X.T @ X)
    se_rho  = np.sqrt(sigma2 * xtx_inv[0, 0])
    adf_stat = rho / (se_rho + 1e-9)
    return float(adf_stat), lag

def ou_params(spread: np.ndarray, dt: float = 1.0) -> dict:
    """
    Estimate Ornstein-Uhlenbeck parameters from discrete spread series.

    Model: S_t - S_{t-1} = a + b * S_{t-1} + epsilon
    kappa = -log(1+b)/dt   (mean reversion speed, annualised)
    mu    = -a/b            (long-run mean)
    sigma = std(epsilon)/sqrt(dt)
    half_life = log(2)/kappa  (days to revert halfway)
    """
    ds  = np.diff(spread)
    s_l = spread[:-1]
    a, b, eps, _ = ols(ds, s_l)
    if b >= 0:
        # Non-stationary: return degenerate params
        return {"kappa": 0.0, "mu": spread.mean(),
                "sigma": spread.std(), "half_life": np.inf}
    kappa    = -np.log(1 + b) / dt
    mu       = -a / b
    sigma    = eps.std() / np.sqrt(dt)
    half_life= np.log(2) / (kappa + 1e-9)
    return {"kappa": kappa, "mu": mu, "sigma": sigma,
            "half_life": half_life, "a": a, "b": b, "eps": eps}

# =============================================================================
# PRINT HEADER + DATA
# =============================================================================
print()
print("=" * 65)
print("  MODULE 50: PAIRS TRADING")
print("  Cointegration | ADF | OU | Z-Score | Backtest | Screening")
print("=" * 65)

# Pairs universe: sector ETFs within same sector
UNIVERSE = {
    "Energy"    : ["XOM","CVX","COP","SLB"],
    "Financials": ["JPM","BAC","GS","MS"],
    "Tech"      : ["MSFT","AAPL","NVDA","AMD"],
    "Retail"    : ["WMT","TGT","COST","KR"],
}

raw = {}
for sector, tickers in UNIVERSE.items():
    for tk in tickers:
        try:
            d = yf.download(tk, start="2018-01-01", end="2023-12-31",
                            auto_adjust=True, progress=False)
            px = d["Close"].squeeze().dropna()
            if len(px) > 500:
                raw[tk] = px
        except Exception:
            pass

import pandas as pd
prices_df = pd.DataFrame(raw).dropna()
tickers   = list(prices_df.columns)
log_p     = np.log(prices_df.values)
dates     = prices_df.index
N, M      = log_p.shape

section(1, f"Universe: {M} assets  {N} days  [{dates[0].date()} -- {dates[-1].date()}]")

# =============================================================================
# 2.  PAIR SCREENING: ENGLE-GRANGER COINTEGRATION
# =============================================================================
ADF_CRIT_5PCT = -2.86

candidates = []
for i, j in combinations(range(M), 2):
    tkA, tkB = tickers[i], tickers[j]
    pA, pB   = log_p[:, i], log_p[:, j]
    alpha, beta, resid, r2 = ols(pA, pB)
    adf_stat, n_lags = adf_test(resid, max_lags=2)
    ou = ou_params(resid)
    cointegrated = adf_stat < ADF_CRIT_5PCT and ou["half_life"] < 90
    candidates.append({
        "pair"      : (tkA, tkB),
        "beta"      : beta,
        "alpha"     : alpha,
        "adf"       : adf_stat,
        "half_life" : ou["half_life"],
        "kappa"     : ou["kappa"],
        "r2"        : r2,
        "coint"     : cointegrated,
        "resid"     : resid,
    })

coint_pairs = [c for c in candidates if c["coint"]]
coint_pairs.sort(key=lambda x: x["adf"])   # most cointegrated first

section(2, f"Screened {len(candidates)} pairs  "
           f"cointegrated: {len(coint_pairs)}  "
           f"(ADF<{ADF_CRIT_5PCT}  half-life<90d)")
print()
print(f"  {'Pair':14s}  {'ADF':>8s}  {'Half-life':>10s}  {'Beta':>8s}  {'R2':>6s}")
print(f"  {'-'*55}")
for c in coint_pairs[:8]:
    hl = c["half_life"]
    hl_str = f"{hl:.1f}d" if hl < 365 else "inf"
    print(f"  {str(c['pair']):14s}  {c['adf']:+8.3f}  {hl_str:>10s}  "
          f"{c['beta']:+8.4f}  {c['r2']:6.4f}")
print()

# =============================================================================
# 3.  BEST PAIR DEEP DIVE
# =============================================================================
if coint_pairs:
    best = coint_pairs[0]
else:
    best = candidates[0]

tkA, tkB = best["pair"]
iA, iB   = tickers.index(tkA), tickers.index(tkB)
pA, pB   = log_p[:, iA], log_p[:, iB]
beta     = best["beta"]
alpha_ols= best["alpha"]
spread   = pA - alpha_ols - beta * pB    # cointegrating residual

ou = ou_params(spread)
section(3, f"Best pair: {tkA}/{tkB}  ADF={best['adf']:.3f}  "
           f"beta={beta:.4f}  half-life={ou['half_life']:.1f}d  "
           f"kappa={ou['kappa']:.4f}")

# =============================================================================
# 4.  OU SIMULATION (theoretical vs empirical)
# =============================================================================
def simulate_ou(kappa, mu, sigma, s0, T, dt=1.0, seed=42):
    """
    Simulate OU process via Euler-Maruyama scheme:
        S_{t+1} = S_t + kappa*(mu - S_t)*dt + sigma*sqrt(dt)*Z
    """
    rng = np.random.default_rng(seed)
    s   = np.zeros(T)
    s[0]= s0
    for t in range(1, T):
        s[t] = s[t-1] + kappa*(mu - s[t-1])*dt + sigma*np.sqrt(dt)*rng.normal()
    return s

sim_spread = simulate_ou(ou["kappa"], ou["mu"], ou["sigma"],
                          spread[0], N)

section(4, f"OU simulation: kappa={ou['kappa']:.4f}  mu={ou['mu']:.4f}  "
           f"sigma={ou['sigma']:.4f}")

# =============================================================================
# 5.  Z-SCORE SIGNAL CONSTRUCTION
# =============================================================================
# Rolling normalisation window
ROLL_WIN  = 60
ENTRY_Z   = 2.0
EXIT_Z    = 0.5

z_score = np.zeros(N)
for t in range(ROLL_WIN, N):
    w       = spread[t-ROLL_WIN:t]
    z_score[t] = (spread[t] - w.mean()) / (w.std() + 1e-9)

# Signal: +1 long spread, -1 short spread, 0 flat
signal = np.zeros(N)
pos    = 0
for t in range(ROLL_WIN, N):
    z = z_score[t]
    if pos == 0:
        if z < -ENTRY_Z: pos =  1
        if z >  ENTRY_Z: pos = -1
    elif pos == 1:
        if z > -EXIT_Z:  pos =  0
    elif pos == -1:
        if z <  EXIT_Z:  pos =  0
    signal[t] = pos

n_long  = (signal == 1).sum()
n_short = (signal == -1).sum()

section(5, f"Z-score signal  entry={ENTRY_Z}  exit={EXIT_Z}  "
           f"long={n_long}d  short={n_short}d  "
           f"flat={(signal==0).sum()}d")

# =============================================================================
# 6.  BACKTEST
# =============================================================================
ret_pA = np.diff(pA, prepend=pA[0])
ret_pB = np.diff(pB, prepend=pB[0])

# Spread return = ret_A - beta * ret_B  (dollar-neutral)
spread_ret  = ret_pA - beta * ret_pB
strat_ret   = np.zeros(N)
strat_ret[1:] = signal[:-1] * spread_ret[1:]   # lag signal

# Performance
cum_strat = np.exp(np.cumsum(strat_ret)) - 1
cum_spy   = np.exp(np.cumsum(
    np.log(prices_df["SPY"].dropna() /
           prices_df["SPY"].dropna().shift(1)).fillna(0).values[:N]
    if "SPY" in prices_df.columns else ret_pA
)) - 1

sr_strat = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(252)
peak     = np.maximum.accumulate(1 + cum_strat)
mdd      = ((peak - (1 + cum_strat)) / peak).max()
hit      = (strat_ret[strat_ret != 0] > 0).mean()

section(6, f"Backtest  Sharpe={sr_strat:.3f}  MDD={mdd:.3f}  "
           f"HitRate={hit:.3f}  CAGR={np.exp(strat_ret.mean()*252)-1:.3f}")

# =============================================================================
# 7.  ROLLING COINTEGRATION STABILITY
# =============================================================================
WIN_COINT = 252
roll_adf  = np.full(N, np.nan)
roll_hl   = np.full(N, np.nan)
for t in range(WIN_COINT, N, 5):   # step=5 for speed
    pA_w = pA[t-WIN_COINT:t]
    pB_w = pB[t-WIN_COINT:t]
    _, _, res_w, _ = ols(pA_w, pB_w)
    adf_w, _ = adf_test(res_w, max_lags=1)
    ou_w     = ou_params(res_w)
    roll_adf[t] = adf_w
    roll_hl[t]  = min(ou_w["half_life"], 365)

section(7, f"Rolling cointegration (1y window, step=5d) computed")

# =============================================================================
# 8.  PAIR CORRELATION HEATMAP (screening matrix)
# =============================================================================
adf_matrix = np.full((M, M), np.nan)
for c in candidates:
    i = tickers.index(c["pair"][0])
    j = tickers.index(c["pair"][1])
    adf_matrix[i, j] = c["adf"]
    adf_matrix[j, i] = c["adf"]

section(8, f"ADF matrix computed for {M}x{M} universe")

# =============================================================================
# FIGURE 1: COINTEGRATION OVERVIEW
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle(f"Module 50 -- Pairs Trading: {tkA}/{tkB} Cointegration Analysis",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.45)
t_ = np.arange(N)

# 1A: Normalised price series
ax = fig.add_subplot(gs[0, :])
pA_norm = (np.exp(pA) / np.exp(pA[0])) * 100
pB_norm = (np.exp(pB) / np.exp(pB[0])) * 100
ax.plot(t_, pA_norm, color=ACCENT, lw=1.2, label=tkA)
ax.plot(t_, pB_norm, color=GREEN,  lw=1.2, label=tkB)
ax.set_title(f"Normalised Price Series (base=100)  beta={beta:.4f}")
ax.set_xlabel("Day"); ax.set_ylabel("Price (base 100)")
ax.legend(fontsize=8); ax.grid(True)

# 1B: Spread + OU simulation
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_, spread,     color=ACCENT, lw=0.8, alpha=0.7, label="Empirical spread")
ax.plot(t_, sim_spread, color=AMBER,  lw=0.8, alpha=0.6, label="OU simulation")
ax.axhline(ou["mu"], color=GREEN, lw=1.2, ls="--", label=f"mu={ou['mu']:.4f}")
ax.axhline(ou["mu"] + 2*spread.std(), color=RED, lw=0.8, ls=":", alpha=0.7)
ax.axhline(ou["mu"] - 2*spread.std(), color=RED, lw=0.8, ls=":", alpha=0.7)
ax.set_title(f"Spread: Empirical vs OU  half-life={ou['half_life']:.1f}d")
ax.set_xlabel("Day"); ax.set_ylabel("log spread")
ax.legend(fontsize=7); ax.grid(True)

# 1C: Z-score + signals
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_, z_score, color=ACCENT, lw=0.7, alpha=0.8, label="Z-score")
ax.axhline( ENTRY_Z, color=RED,   lw=1.0, ls="--", label=f"+{ENTRY_Z} entry")
ax.axhline(-ENTRY_Z, color=GREEN, lw=1.0, ls="--", label=f"-{ENTRY_Z} entry")
ax.axhline( EXIT_Z,  color=AMBER, lw=0.8, ls=":",  label=f"+/-{EXIT_Z} exit")
ax.axhline(-EXIT_Z,  color=AMBER, lw=0.8, ls=":",  alpha=0.7)
ax.fill_between(t_, z_score, 0, where=signal==1,  alpha=0.25, color=GREEN)
ax.fill_between(t_, z_score, 0, where=signal==-1, alpha=0.25, color=RED)
ax.set_title("Z-Score & Entry/Exit Signals")
ax.set_xlabel("Day"); ax.set_ylabel("Z-score")
ax.legend(fontsize=6); ax.grid(True)

# 1D: Strategy cumulative return
ax = fig.add_subplot(gs[2, 0])
ax.plot(t_, cum_strat*100, color=ACCENT, lw=1.5,
        label=f"Pairs  SR={sr_strat:.2f}  MDD={mdd:.2f}")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Pairs Strategy Cumulative Return (%)")
ax.set_xlabel("Day"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1E: Rolling cointegration stability
ax = fig.add_subplot(gs[2, 1])
valid = ~np.isnan(roll_adf)
ax.plot(t_[valid], roll_adf[valid], color=ACCENT, lw=1.0,
        label="Rolling ADF (1y)")
ax.axhline(ADF_CRIT_5PCT, color=RED,   lw=1.2, ls="--",
           label=f"5% critical ({ADF_CRIT_5PCT})")
ax.axhline(-3.43,          color=AMBER, lw=0.8, ls="--",
           label="1% critical (-3.43)")
ax.fill_between(t_[valid], roll_adf[valid], ADF_CRIT_5PCT,
                where=roll_adf[valid] < ADF_CRIT_5PCT,
                alpha=0.3, color=GREEN, label="Cointegrated region")
ax.set_title("Rolling Cointegration Stability (1y window)")
ax.set_xlabel("Day"); ax.set_ylabel("ADF statistic")
ax.legend(fontsize=6); ax.grid(True)

fig.savefig(os.path.join(FIGS, "m50_fig1_cointegration_overview.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: ADF SCREENING MATRIX + OU PARAMETER SPACE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle("Module 50 -- Pair Screening: ADF Matrix & OU Parameter Space",
             fontsize=12, color=TEXT, y=1.01)

ax = axes[0]
im = ax.imshow(adf_matrix, cmap="RdYlGn_r", aspect="auto",
               vmin=-5, vmax=0)
ax.set_xticks(range(M)); ax.set_xticklabels(tickers, rotation=45,
                                             ha="right", fontsize=7)
ax.set_yticks(range(M)); ax.set_yticklabels(tickers, fontsize=7)
for i in range(M):
    for j in range(M):
        if not np.isnan(adf_matrix[i, j]):
            coint_flag = "*" if adf_matrix[i, j] < ADF_CRIT_5PCT else ""
            ax.text(j, i, f"{adf_matrix[i,j]:.1f}{coint_flag}",
                    ha="center", va="center", fontsize=6,
                    color="white" if adf_matrix[i,j] < -3 else TEXT)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("ADF stat (* = coint. 5%)", color=TEXT, fontsize=7)
cb.ax.yaxis.set_tick_params(color=TEXT)
ax.set_title("Engle-Granger ADF Screening Matrix")

ax = axes[1]
for c in candidates:
    hl  = min(c["half_life"], 200)
    col = GREEN if c["coint"] else RED
    ax.scatter(abs(c["adf"]), hl, color=col, s=30, alpha=0.7)
    if c["coint"]:
        ax.annotate(f"{c['pair'][0]}/{c['pair'][1]}",
                    (abs(c["adf"]), hl), fontsize=6, color=TEXT,
                    xytext=(3, 3), textcoords="offset points")
ax.axhline(90, color=AMBER, lw=1.0, ls="--", label="Half-life 90d threshold")
ax.axvline(abs(ADF_CRIT_5PCT), color=RED, lw=1.0, ls="--",
           label=f"|ADF|={abs(ADF_CRIT_5PCT)} (5% crit)")
ax.scatter([], [], color=GREEN, s=30, label="Cointegrated")
ax.scatter([], [], color=RED,   s=30, label="Not cointegrated")
ax.set_xlabel("|ADF statistic|"); ax.set_ylabel("Half-life (days, capped 200d)")
ax.set_title("OU Parameter Space: |ADF| vs Half-life")
ax.legend(fontsize=7); ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m50_fig2_screening_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: SPREAD DISTRIBUTION + RESIDUAL DIAGNOSTICS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle(f"Module 50 -- Spread Diagnostics: {tkA}/{tkB}",
             fontsize=12, color=TEXT, y=1.01)

# 3A: Spread histogram vs normal
ax = axes[0, 0]
ax.hist(spread, bins=50, color=ACCENT, alpha=0.8, density=True, edgecolor="none")
xs = np.linspace(spread.min(), spread.max(), 200)
from scipy.stats import norm as sp_norm
ax.plot(xs, sp_norm.pdf(xs, spread.mean(), spread.std()),
        color=AMBER, lw=2.0, label="Normal fit")
ax.axvline(spread.mean(), color=GREEN, lw=1.2, ls="--",
           label=f"mu={spread.mean():.4f}")
ax.set_title("Spread Distribution"); ax.set_xlabel("Spread"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

# 3B: ACF of spread (manual)
ax = axes[0, 1]
max_lag = 40
acf = [1.0]
s_dm = spread - spread.mean()
var  = (s_dm**2).mean()
for lag in range(1, max_lag+1):
    acf.append((s_dm[lag:] * s_dm[:-lag]).mean() / (var + 1e-9))
acf = np.array(acf)
ci  = 1.96 / np.sqrt(N)
ax.bar(range(max_lag+1), acf, color=ACCENT, edgecolor=DARK, alpha=0.8)
ax.axhline( ci, color=RED, lw=0.8, ls="--", label="95% CI")
ax.axhline(-ci, color=RED, lw=0.8, ls="--")
ax.axhline(0,   color=GRID, lw=0.8)
ax.set_title("Spread Autocorrelation Function")
ax.set_xlabel("Lag"); ax.set_ylabel("ACF"); ax.legend(fontsize=7); ax.grid(True)

# 3C: OLS regression scatter P_A vs beta*P_B
ax = axes[1, 0]
pB_hat = alpha_ols + beta * pB
ax.scatter(pB, pA, s=3, alpha=0.3, color=ACCENT, label="Observed")
ax.plot(pB, pB_hat, color=AMBER, lw=1.5, label=f"OLS fit  R2={best['r2']:.4f}")
ax.set_xlabel(f"log({tkB})"); ax.set_ylabel(f"log({tkA})")
ax.set_title(f"OLS Regression: log({tkA}) = a + b*log({tkB})")
ax.legend(fontsize=7); ax.grid(True)

# 3D: OU mean reversion path illustration
ax = axes[1, 1]
n_paths = 5
for k in range(n_paths):
    sim_k = simulate_ou(ou["kappa"], ou["mu"], ou["sigma"],
                         spread.mean() + k * spread.std() * 0.5, 200, seed=k)
    ax.plot(sim_k, lw=0.8, alpha=0.7)
ax.axhline(ou["mu"], color=AMBER, lw=1.5, ls="--",
           label=f"Long-run mean mu={ou['mu']:.4f}")
ax.axhline(ou["mu"] + ou["sigma"]*2, color=RED,   lw=0.8, ls=":", alpha=0.7)
ax.axhline(ou["mu"] - ou["sigma"]*2, color=GREEN, lw=0.8, ls=":", alpha=0.7)
ax.set_title(f"OU Mean Reversion Paths  t_half={ou['half_life']:.1f}d")
ax.set_xlabel("Day"); ax.set_ylabel("Spread"); ax.legend(fontsize=7); ax.grid(True)

for ax in axes.flat:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m50_fig3_spread_diagnostics.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("  MODULE 50 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Cointegration: I(1) series share a common stochastic trend")
print("  [2] Engle-Granger: OLS residuals -> ADF test for stationarity")
print(f"  [3] ADF<{ADF_CRIT_5PCT} (5% crit) => reject unit root => cointegrated")
print("  [4] OU process: dS = kappa*(mu-S)*dt + sigma*dW")
print("  [5] Half-life = log(2)/kappa -- optimal holding period proxy")
print("  [6] Z-score entry/exit avoids over-trading in mean-reversion")
print(f"  [7] Best pair {tkA}/{tkB}: ADF={best['adf']:.3f}  "
      f"half-life={ou['half_life']:.1f}d  SR={sr_strat:.3f}")
print(f"  [8] Rolling coint. stability reveals regime breaks in the pair")
print(f"  NEXT: M51 -- Regime Detection: HMM & Markov Switching Models")
print()

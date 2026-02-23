#!/usr/bin/env python3
"""
M15 — Black-Scholes Formula & Greeks
=====================================
Module 3 of 9 | CQF Concepts Explained

Theory
------
Under GBM and no-arbitrage, the Black-Scholes PDE is:

    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

Closed-form solution for a European call (put via put-call parity):

    C = S*N(d1) - K*e^{-rT}*N(d2)
    P = K*e^{-rT}*N(-d2) - S*N(-d1)

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Greeks (sensitivities):
    Delta  = dV/dS          : hedge ratio
    Gamma  = d2V/dS2        : delta hedging cost
    Vega   = dV/d(sigma)    : volatility sensitivity
    Theta  = dV/dt          : time decay
    Rho    = dV/dr          : interest rate sensitivity

Delta-Gamma-Theta relationship (BS PDE rewritten):
    Theta + 0.5*sigma^2*S^2*Gamma + r*S*Delta - r*V = 0

Put-Call Parity:
    C - P = S - K*e^{-rT}

Author : Jose O. Bobadilla | CQF
Project: 19-cqf-concepts-explained
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK   = "#0a0a0a"
PANEL  = "#111111"
WHITE  = "#e8e8e8"
GREY   = "#444444"
BLUE   = "#4a9eff"
GREEN  = "#2ecc71"
YELLOW = "#f1c40f"
RED    = "#e74c3c"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"
TEAL   = "#1abc9c"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": GREY,   "axes.labelcolor": WHITE,
    "xtick.color": WHITE,     "ytick.color": WHITE,
    "text.color": WHITE,      "grid.color": GREY,
    "grid.alpha": 0.35,       "grid.linestyle": "--",
    "font.family": "monospace",
})

OUT_DIR = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs/figures"
)
os.makedirs(OUT_DIR, exist_ok=True)
WATERMARK = "Jose O. Bobadilla | CQF"

def watermark(ax):
    ax.text(0.98, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=6, color=GREY, ha="right", va="bottom",
            style="italic", alpha=0.8)

# ---------------------------------------------------------------------------
# Black-Scholes engine
# ---------------------------------------------------------------------------
def bs_d1d2(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def bs_call(S, K, T, r, sigma):
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt="call"):
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if opt=="call" else norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(np.maximum(T, 1e-10)))

def bs_vega(S, K, T, r, sigma):
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(np.maximum(T, 1e-10)) / 100  # per 1% sigma move

def bs_theta(S, K, T, r, sigma, opt="call"):
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-10)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if opt == "call":
        return (term1 - r*K*np.exp(-r*T)*norm.cdf(d2))  / 365  # per calendar day
    else:
        return (term1 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

def bs_rho(S, K, T, r, sigma, opt="call"):
    _, d2 = bs_d1d2(S, K, T, r, sigma)
    if opt == "call":
        return K*T*np.exp(-r*T)*norm.cdf(d2)  / 100   # per 1% rate move
    else:
        return -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20

C0 = bs_call(S0, K, T, r, sigma)
P0 = bs_put(S0, K, T, r, sigma)
print(f"[M15] ATM Call = {C0:.6f}  |  ATM Put = {P0:.6f}")
print(f"      Put-call parity check: C-P = {C0-P0:.6f}  "
      f"S-K*exp(-rT) = {S0 - K*np.exp(-r*T):.6f}")


# ===========================================================================
# FIGURE 1 — Price surface: call/put vs Spot and Time-to-Expiry
# ===========================================================================
print("[M15] Figure 1: Price surface ...")
t0 = time.perf_counter()

S_grid = np.linspace(60, 140, 80)
T_grid = np.linspace(0.02, 2.0, 80)
SS, TT = np.meshgrid(S_grid, T_grid)

CC = bs_call(SS, K, TT, r, sigma)
PP = bs_put(SS, K, TT, r, sigma)

fig = plt.figure(figsize=(14, 6), facecolor=DARK)
fig.suptitle(
    f"M15 — Black-Scholes Price Surface\n"
    f"K={K}, r={r:.0%}, sigma={sigma:.0%}",
    color=WHITE, fontsize=11
)

for idx, (ZZ, title, col) in enumerate([
    (CC, "European Call Price C(S,T)", "YlOrRd"),
    (PP, "European Put Price P(S,T)",  "YlGnBu"),
]):
    ax = fig.add_subplot(1, 2, idx+1, projection="3d")
    ax.set_facecolor(PANEL)
    surf = ax.plot_surface(SS, TT, ZZ, cmap=col, alpha=0.85, linewidth=0)
    ax.set_xlabel("Spot S", labelpad=6, color=WHITE)
    ax.set_ylabel("Time to Expiry T", labelpad=6, color=WHITE)
    ax.set_zlabel("Price", labelpad=6, color=WHITE)
    ax.set_title(title, color=WHITE, fontsize=9, pad=8)
    ax.tick_params(colors=WHITE, labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GREY)
    ax.yaxis.pane.set_edgecolor(GREY)
    ax.zaxis.pane.set_edgecolor(GREY)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m15_01_price_surface.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Greeks vs Spot (all 5 Greeks, call and put)
# ===========================================================================
print("[M15] Figure 2: Greeks vs spot ...")
t0 = time.perf_counter()

S_range = np.linspace(60, 140, 300)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
fig.suptitle(
    f"M15 — Black-Scholes Greeks vs Spot\n"
    f"K={K}, T={T}Y, r={r:.0%}, sigma={sigma:.0%}",
    color=WHITE, fontsize=11
)

# Delta
ax = axes[0, 0]
ax.plot(S_range, bs_delta(S_range, K, T, r, sigma, "call"), color=BLUE,  lw=2.5,
        label="Call delta")
ax.plot(S_range, bs_delta(S_range, K, T, r, sigma, "put"),  color=GREEN, lw=2.5,
        label="Put delta")
ax.axhline(0,   color=GREY, lw=0.8)
ax.axhline(1,   color=GREY, lw=0.8, linestyle=":")
ax.axhline(-1,  color=GREY, lw=0.8, linestyle=":")
ax.axvline(K,   color=WHITE, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Delta")
ax.set_title("Delta = dV/dS\n(call: 0->1, put: -1->0)", color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Gamma
ax = axes[0, 1]
ax.plot(S_range, bs_gamma(S_range, K, T, r, sigma), color=YELLOW, lw=2.5,
        label="Gamma (call = put)")
ax.axvline(K, color=WHITE, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Gamma")
ax.set_title("Gamma = d2V/dS2\nPeaks ATM, identical for call and put",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Vega
ax = axes[0, 2]
ax.plot(S_range, bs_vega(S_range, K, T, r, sigma), color=ORANGE, lw=2.5,
        label="Vega (per 1% sigma, call = put)")
ax.axvline(K, color=WHITE, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Vega (per 1% sigma)")
ax.set_title("Vega = dV/d(sigma) / 100\nMax at ATM, identical for call and put",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Theta
ax = axes[1, 0]
ax.plot(S_range, bs_theta(S_range, K, T, r, sigma, "call"), color=BLUE,  lw=2.5,
        label="Call theta (per day)")
ax.plot(S_range, bs_theta(S_range, K, T, r, sigma, "put"),  color=GREEN, lw=2.5,
        label="Put theta (per day)")
ax.axhline(0, color=GREY, lw=0.8)
ax.axvline(K, color=WHITE, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Theta (per calendar day)")
ax.set_title("Theta = dV/dt / 365\nTime decay (negative = lose value daily)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Rho
ax = axes[1, 1]
ax.plot(S_range, bs_rho(S_range, K, T, r, sigma, "call"), color=BLUE,  lw=2.5,
        label="Call rho (per 1% rate)")
ax.plot(S_range, bs_rho(S_range, K, T, r, sigma, "put"),  color=GREEN, lw=2.5,
        label="Put rho (per 1% rate)")
ax.axhline(0, color=GREY, lw=0.8)
ax.axvline(K, color=WHITE, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Rho (per 1% rate)")
ax.set_title("Rho = dV/dr / 100\nCall positive, put negative",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# BS PDE verification: Theta + 0.5*sig^2*S^2*Gamma + r*S*Delta - r*V = 0
ax = axes[1, 2]
pde_call = (bs_theta(S_range, K, T, r, sigma, "call") * 365
            + 0.5*sigma**2*S_range**2 * bs_gamma(S_range, K, T, r, sigma)
            + r*S_range * bs_delta(S_range, K, T, r, sigma, "call")
            - r * bs_call(S_range, K, T, r, sigma))
ax.plot(S_range, pde_call, color=RED, lw=2, label="BS PDE residual (call)")
ax.axhline(0, color=YELLOW, lw=2, linestyle="--", label="Zero (exact)")
ax.set_xlabel("Spot S"); ax.set_ylabel("PDE residual")
ax.set_title("BS PDE Verification\n"
             "Theta + 0.5*sig^2*S^2*Gamma + r*S*Delta - r*V = 0",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m15_02_greeks.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Greeks vs Time-to-Expiry + Put-Call Parity
# ===========================================================================
print("[M15] Figure 3: Greeks vs time and put-call parity ...")
t0 = time.perf_counter()

T_range = np.linspace(0.01, 2.0, 300)
S_itm, S_atm, S_otm = 110.0, 100.0, 90.0
styles = [(S_itm, BLUE, "ITM S=110"),
          (S_atm, YELLOW, "ATM S=100"),
          (S_otm, GREEN, "OTM S=90")]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
fig.suptitle(
    "M15 — Greeks vs Time-to-Expiry and Put-Call Parity Verification\n"
    f"K={K}, r={r:.0%}, sigma={sigma:.0%}",
    color=WHITE, fontsize=11
)

# Delta vs T
ax = axes[0, 0]
for s, col, lbl in styles:
    ax.plot(T_range, bs_delta(s, K, T_range, r, sigma, "call"),
            color=col, lw=2, label=f"Call {lbl}")
ax.set_xlabel("Time to Expiry T"); ax.set_ylabel("Call Delta")
ax.set_title("Call Delta vs T\n(converges to 0 or 1 as T->0)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Gamma vs T
ax = axes[0, 1]
for s, col, lbl in styles:
    ax.plot(T_range, bs_gamma(s, K, T_range, r, sigma),
            color=col, lw=2, label=lbl)
ax.set_xlabel("Time to Expiry T"); ax.set_ylabel("Gamma")
ax.set_title("Gamma vs T\n(ATM gamma spikes as T->0: pin risk)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Vega vs T
ax = axes[0, 2]
for s, col, lbl in styles:
    ax.plot(T_range, bs_vega(s, K, T_range, r, sigma),
            color=col, lw=2, label=lbl)
ax.set_xlabel("Time to Expiry T"); ax.set_ylabel("Vega (per 1% sigma)")
ax.set_title("Vega vs T\n(long-dated options most vega sensitive)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Theta vs T
ax = axes[1, 0]
for s, col, lbl in styles:
    ax.plot(T_range, bs_theta(s, K, T_range, r, sigma, "call"),
            color=col, lw=2, label=f"Call {lbl}")
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlabel("Time to Expiry T"); ax.set_ylabel("Theta (per day)")
ax.set_title("Call Theta vs T\n(ATM theta most negative near expiry)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Put-Call Parity verification across S and T
ax = axes[1, 1]
S_pcp = np.linspace(60, 140, 100)
for t_val, col, lbl in [(0.25, BLUE, "T=0.25Y"),
                         (0.50, GREEN, "T=0.50Y"),
                         (1.00, YELLOW, "T=1.00Y"),
                         (2.00, ORANGE, "T=2.00Y")]:
    C  = bs_call(S_pcp, K, t_val, r, sigma)
    P  = bs_put(S_pcp, K, t_val, r, sigma)
    lhs = C - P
    rhs = S_pcp - K*np.exp(-r*t_val)
    ax.plot(S_pcp, lhs - rhs, color=col, lw=2, label=lbl)
ax.axhline(0, color=WHITE, lw=2, linestyle="--", label="Zero (exact parity)")
ax.set_xlabel("Spot S"); ax.set_ylabel("C - P - (S - Ke^{-rT})")
ax.set_title("Put-Call Parity Residual\nC - P = S - K*e^{-rT}  (must be 0)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Summary table
ax = axes[1, 2]
ax.axis("off")
rows = [
    ["Greek",    "Formula",                       "Call",    "Put"],
    ["Delta",    "dV/dS",                         "N(d1)",   "N(d1)-1"],
    ["Gamma",    "d2V/dS2",                       "N'(d1)/(S*sig*sqrt(T))", "same"],
    ["Vega",     "dV/d(sig)",                     "S*N'(d1)*sqrt(T)",       "same"],
    ["Theta",    "dV/dt",                         "negative", "neg/pos"],
    ["Rho",      "dV/dr",                         "positive", "negative"],
    ["",         "",                              "",         ""],
    ["ATM Call", f"C={C0:.4f}",                  f"Delta={bs_delta(S0,K,T,r,sigma,'call'):.4f}",
                 f"Gamma={bs_gamma(S0,K,T,r,sigma):.4f}"],
    ["ATM Put",  f"P={P0:.4f}",                  f"Delta={bs_delta(S0,K,T,r,sigma,'put'):.4f}",
                 f"Vega={bs_vega(S0,K,T,r,sigma):.4f}"],
    ["PCP check", f"C-P={C0-P0:.4f}",            f"S-Ke^(-rT)={S0-K*np.exp(-r*T):.4f}", "OK"],
]
colors_row = [[PANEL]*4] + [
    [DARK if i%2==0 else PANEL]*4 for i in range(len(rows)-1)
]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=colors_row)
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.0, 1.45)
for (r_i, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=YELLOW if r_i==0 else WHITE,
                        weight="bold" if r_i==0 else "normal")
ax.set_title("Greeks Summary Table", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m15_03_greeks_time_pcp.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 15 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] BS formula: C=S*N(d1)-K*e^{-rT}*N(d2)")
print("  [2] d1=(ln(S/K)+(r+sig^2/2)*T)/(sig*sqrt(T))")
print("  [3] Delta: call in [0,1], put in [-1,0]")
print("  [4] Gamma = Vega peak at ATM; call theta always negative")
print("  [5] BS PDE: Theta+0.5*sig^2*S^2*Gamma+r*S*Delta-r*V=0")
print("  [6] Put-Call Parity: C-P=S-K*e^{-rT}")
print(f"  ATM Call={C0:.6f}  Put={P0:.6f}  "
      f"Delta(call)={bs_delta(S0,K,T,r,sigma,'call'):.4f}  "
      f"Gamma={bs_gamma(S0,K,T,r,sigma):.4f}")
print("=" * 65)

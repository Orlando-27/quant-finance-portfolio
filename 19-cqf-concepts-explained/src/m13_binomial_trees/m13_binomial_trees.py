#!/usr/bin/env python3
"""
M13 — Binomial Trees: Cox-Ross-Rubinstein (CRR) Model
======================================================
Module 3 of 9 | CQF Concepts Explained

Theory
------
The CRR binomial tree discretizes the stock price process as follows.
Over each time step dt = T/N, the stock either:
    moves up:   S -> S * u,   with risk-neutral probability p
    moves down: S -> S * d,   with probability 1 - p

CRR parameters (match GBM moments):
    u = exp(sigma * sqrt(dt))
    d = 1 / u  = exp(-sigma * sqrt(dt))
    p = (exp(r*dt) - d) / (u - d)         (risk-neutral probability)

The no-arbitrage condition requires: d < exp(r*dt) < u.

European option pricing (backward induction):
    At expiry (step N): V_{N,j} = payoff(S_0 * u^j * d^{N-j})
    At each earlier node: V_{i,j} = exp(-r*dt) * [p*V_{i+1,j+1} + (1-p)*V_{i+1,j}]

American option pricing:
    At each node, compare continuation value with immediate exercise:
    V_{i,j} = max(payoff(S_{i,j}), exp(-r*dt) * [p*V_{i+1,j+1} + (1-p)*V_{i+1,j}])

Early exercise premium:
    American put price - European put price >= 0  (always)
    American call on non-dividend paying stock = European call (no early exercise)

Black-Scholes limit:
    As N -> inf, the CRR binomial tree price converges to the Black-Scholes price.
    Convergence rate: O(1/N) with oscillations due to odd/even N effects.

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
# Core CRR engine
# ---------------------------------------------------------------------------
def crr_price(S, K, T, r, sigma, N, option_type="call", style="european"):
    """
    CRR binomial tree pricing.

    Parameters
    ----------
    S, K, T, r, sigma : standard option parameters
    N                  : number of time steps
    option_type        : 'call' or 'put'
    style              : 'european' or 'american'

    Returns
    -------
    price : float
    delta : float  (at root node)
    gamma : float  (at root node)
    """
    dt  = T / N
    u   = np.exp(sigma * np.sqrt(dt))
    d   = 1.0 / u
    p   = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal stock prices: S * u^j * d^(N-j) for j=0..N
    j_arr = np.arange(N + 1)
    ST    = S * (u ** j_arr) * (d ** (N - j_arr))

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if style == "american":
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            intrinsic = (np.maximum(S_i - K, 0.0) if option_type == "call"
                         else np.maximum(K - S_i, 0.0))
            V = np.maximum(V, intrinsic)

    price = V[0]

    # Greeks at root (finite difference on tree values at step 1)
    # Rebuild one step to get V_u and V_d
    j1   = np.arange(N)
    ST1  = S * (u ** j1) * (d ** (N - 1 - j1))
    if option_type == "call":
        V1 = np.maximum(ST1 - K, 0.0)
    else:
        V1 = np.maximum(K - ST1, 0.0)
    for i in range(N - 2, -1, -1):
        V1 = disc * (p * V1[1:] + (1 - p) * V1[:-1])
        if style == "american":
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            intr = (np.maximum(S_i - K, 0.0) if option_type == "call"
                    else np.maximum(K - S_i, 0.0))
            V1 = np.maximum(V1, intr)
    if len(V1) < 2:
        return price, np.nan, np.nan
    V_u, V_d = V1[1], V1[0]
    Su, Sd   = S * u, S * d
    delta    = (V_u - V_d) / (Su - Sd)
    gamma    = ((V_u - price) / (Su - S) - (price - V_d) / (S - Sd)) / (0.5*(Su - Sd))

    return price, delta, gamma


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes closed-form."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20
SEED  = 42

bs_call = bs_price(S0, K, T, r, sigma, "call")
bs_put  = bs_price(S0, K, T, r, sigma, "put")
print(f"[M13] BS call = {bs_call:.6f}  |  BS put = {bs_put:.6f}")


# ===========================================================================
# FIGURE 1 — Tree diagram (small N=6) + price lattice visualization
# ===========================================================================
print("[M13] Figure 1: Tree diagram and price lattice ...")
t0 = time.perf_counter()

N_small = 6
dt_s = T / N_small
u_s  = np.exp(sigma * np.sqrt(dt_s))
d_s  = 1.0 / u_s
p_s  = (np.exp(r * dt_s) - d_s) / (u_s - d_s)
disc_s = np.exp(-r * dt_s)

# Build full price lattice
S_lat = np.full((N_small+1, N_small+1), np.nan)
for i in range(N_small+1):
    for j in range(i+1):
        S_lat[j, i] = S0 * (u_s**j) * (d_s**(i-j))

# Build call option value lattice (European)
V_lat = np.full((N_small+1, N_small+1), np.nan)
for j in range(N_small+1):
    V_lat[j, N_small] = max(S_lat[j, N_small] - K, 0)
for i in range(N_small-1, -1, -1):
    for j in range(i+1):
        V_lat[j, i] = disc_s*(p_s*V_lat[j+1,i+1] + (1-p_s)*V_lat[j,i+1])

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle(
    f"M13 — CRR Binomial Tree (N={N_small} steps)\n"
    f"S={S0}, K={K}, T={T}Y, r={r:.0%}, sigma={sigma:.0%}  |  "
    f"u={u_s:.4f}, d={d_s:.4f}, p={p_s:.4f}",
    color=WHITE, fontsize=10
)

# Stock price lattice
ax = axes[0]
for i in range(N_small+1):
    for j in range(i+1):
        s_val = S_lat[j, i]
        circ = plt.Circle((i, j - i/2), 0.18, color=BLUE, alpha=0.8, zorder=3)
        ax.add_patch(circ)
        ax.text(i, j - i/2, f"{s_val:.1f}", ha="center", va="center",
                fontsize=6, color=WHITE, fontweight="bold", zorder=4)
        if i < N_small:
            ax.plot([i, i+1], [j-i/2, (j+1)-(i+1)/2], color=GREEN,  lw=0.8, alpha=0.6)
            ax.plot([i, i+1], [j-i/2,  j   -(i+1)/2], color=ORANGE, lw=0.8, alpha=0.6)
ax.set_xlim(-0.4, N_small+0.4); ax.set_ylim(-N_small/2-0.5, N_small/2+0.5)
ax.set_xlabel("Time step"); ax.set_yticks([])
ax.set_title(f"Stock Price Lattice S_{{i,j}}", color=WHITE, fontsize=9)
ax.set_facecolor(PANEL); watermark(ax)
# Time axis labels
for i in range(N_small+1):
    ax.text(i, -N_small/2-0.4, f"t={i*dt_s:.2f}",
            ha="center", va="top", fontsize=6, color=GREY)

# Option value lattice
ax = axes[1]
vmax = V_lat[~np.isnan(V_lat)].max()
for i in range(N_small+1):
    for j in range(i+1):
        v_val = V_lat[j, i]
        intensity = v_val / (vmax + 1e-8)
        col = plt.cm.YlOrRd(0.2 + 0.78*intensity)
        circ = plt.Circle((i, j - i/2), 0.18, color=col, alpha=0.9, zorder=3)
        ax.add_patch(circ)
        ax.text(i, j - i/2, f"{v_val:.2f}", ha="center", va="center",
                fontsize=6, color="black", fontweight="bold", zorder=4)
        if i < N_small:
            ax.plot([i, i+1], [j-i/2, (j+1)-(i+1)/2], color=GREY, lw=0.8, alpha=0.5)
            ax.plot([i, i+1], [j-i/2,  j   -(i+1)/2], color=GREY, lw=0.8, alpha=0.5)
ax.set_xlim(-0.4, N_small+0.4); ax.set_ylim(-N_small/2-0.5, N_small/2+0.5)
ax.set_xlabel("Time step"); ax.set_yticks([])
ax.set_title(f"European Call Value Lattice V_{{i,j}}\nRoot = {V_lat[0,0]:.4f}  (BS = {bs_call:.4f})",
             color=WHITE, fontsize=9)
ax.set_facecolor(PANEL); watermark(ax)
for i in range(N_small+1):
    ax.text(i, -N_small/2-0.4, f"t={i*dt_s:.2f}",
            ha="center", va="top", fontsize=6, color=GREY)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m13_01_tree_diagram.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Convergence to Black-Scholes
# ===========================================================================
print("[M13] Figure 2: Convergence to Black-Scholes ...")
t0 = time.perf_counter()

N_vals = list(range(1, 201))
crr_calls = [crr_price(S0, K, T, r, sigma, n, "call", "european")[0] for n in N_vals]
crr_puts  = [crr_price(S0, K, T, r, sigma, n, "put",  "european")[0] for n in N_vals]
crr_am_p  = [crr_price(S0, K, T, r, sigma, n, "put",  "american")[0] for n in N_vals]

crr_calls = np.array(crr_calls)
crr_puts  = np.array(crr_puts)
crr_am_p  = np.array(crr_am_p)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("M13 — CRR Tree Convergence to Black-Scholes",
             color=WHITE, fontsize=11)

# (0) Call convergence
ax = axes[0]
ax.plot(N_vals, crr_calls, color=BLUE, lw=1.2, alpha=0.8, label="CRR European Call")
ax.axhline(bs_call, color=YELLOW, lw=2, linestyle="--",
           label=f"BS exact = {bs_call:.6f}")
ax.set_xlabel("N (steps)"); ax.set_ylabel("Option Price")
ax.set_title("European Call: CRR vs Black-Scholes\n(odd/even oscillation damped with N)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (1) Put convergence: European vs American
ax = axes[1]
ax.plot(N_vals, crr_puts,  color=GREEN,  lw=1.2, alpha=0.8, label="CRR European Put")
ax.plot(N_vals, crr_am_p,  color=ORANGE, lw=1.2, alpha=0.8, label="CRR American Put")
ax.axhline(bs_put, color=YELLOW, lw=2, linestyle="--",
           label=f"BS European = {bs_put:.6f}")
ax.set_xlabel("N (steps)"); ax.set_ylabel("Option Price")
ax.set_title("Put: European vs American\nAmerican >= European (early exercise premium)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (2) Absolute error log-scale
ax = axes[2]
err_call = np.abs(crr_calls - bs_call)
err_put  = np.abs(crr_puts  - bs_put)
ax.semilogy(N_vals, err_call, color=BLUE,  lw=1.2, alpha=0.8, label="|CRR call - BS|")
ax.semilogy(N_vals, err_put,  color=GREEN, lw=1.2, alpha=0.8, label="|CRR put - BS|")
# Smooth O(1/N) reference
N_ref = np.linspace(5, 200, 300)
ref   = err_call[4] * 5 / N_ref
ax.semilogy(N_ref, ref, "--", color=GREY, lw=1.5, label="O(1/N) reference")
ax.set_xlabel("N (steps)"); ax.set_ylabel("|Error|")
ax.set_title("Absolute Pricing Error vs N\nOscillates around O(1/N) trend",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m13_02_convergence.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Greeks and American Early Exercise Boundary
# ===========================================================================
print("[M13] Figure 3: Greeks and early exercise boundary ...")
t0 = time.perf_counter()

N_greek = 200

# Delta and Gamma vs spot price
S_range = np.linspace(60, 140, 81)
crr_delta_call = []
crr_delta_put  = []
crr_gamma_call = []
bs_delta_call  = []
bs_delta_put   = []

for s in S_range:
    _, dc, gc = crr_price(s, K, T, r, sigma, N_greek, "call", "european")
    _, dp, gp = crr_price(s, K, T, r, sigma, N_greek, "put",  "european")
    crr_delta_call.append(dc)
    crr_delta_put.append(dp)
    crr_gamma_call.append(gc)
    bs_delta_call.append(bs_delta(s, K, T, r, sigma, "call"))
    bs_delta_put.append(bs_delta(s, K, T, r, sigma, "put"))

# Early exercise boundary for American put: critical S* where exercise is optimal
# Scan S from low to high, find where American put intrinsic >= continuation
boundary_S = []
T_vals = np.linspace(0.02, T, 40)
for t_rem in T_vals:
    lo, hi = 1.0, K
    for _ in range(40):
        mid = 0.5*(lo + hi)
        am_val, _, _ = crr_price(mid, K, t_rem, r, sigma, 100, "put", "american")
        intrinsic = max(K - mid, 0)
        if am_val <= intrinsic + 1e-6:
            hi = mid
        else:
            lo = mid
    boundary_S.append(hi)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(f"M13 — CRR Greeks (N={N_greek}) and American Early Exercise Boundary",
             color=WHITE, fontsize=10)

# (0) Delta vs spot
ax = axes[0]
ax.plot(S_range, crr_delta_call, color=BLUE,   lw=2, label="CRR delta (call)")
ax.plot(S_range, crr_delta_put,  color=GREEN,  lw=2, label="CRR delta (put)")
ax.plot(S_range, bs_delta_call,  color=YELLOW, lw=1.5, linestyle="--",
        label="BS delta (call)")
ax.plot(S_range, [bs_delta(s, K, T, r, sigma, "put") for s in S_range],
        color=ORANGE, lw=1.5, linestyle="--", label="BS delta (put)")
ax.axvline(K, color=WHITE, lw=1, linestyle=":", alpha=0.6, label=f"ATM K={K}")
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlabel("Spot S"); ax.set_ylabel("Delta")
ax.set_title("Delta vs Spot\nCRR vs Black-Scholes", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Gamma vs spot
ax = axes[1]
ax.plot(S_range, crr_gamma_call, color=BLUE, lw=2, label="CRR gamma (call=put)")
ax.axvline(K, color=WHITE, lw=1, linestyle=":", alpha=0.6, label=f"ATM K={K}")
# BS gamma
bs_gamma = []
for s in S_range:
    d1 = (np.log(s/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    bs_gamma.append(norm.pdf(d1) / (s * sigma * np.sqrt(T)))
ax.plot(S_range, bs_gamma, color=YELLOW, lw=1.5, linestyle="--", label="BS gamma")
ax.set_xlabel("Spot S"); ax.set_ylabel("Gamma")
ax.set_title("Gamma vs Spot\nPeaks at ATM, zero deep ITM/OTM",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Early exercise boundary
ax = axes[2]
time_to_expiry = np.array(T_vals)
ax.plot(time_to_expiry, boundary_S, color=RED, lw=2.5,
        label="Critical S* (exercise if S < S*)")
ax.axhline(K, color=WHITE, lw=1.5, linestyle="--",
           label=f"Strike K = {K}")
ax.fill_between(time_to_expiry, boundary_S, K,
                color=RED, alpha=0.12, label="Early exercise region")
ax.fill_between(time_to_expiry, 0, boundary_S,
                color=GREEN, alpha=0.08, label="Hold region")
ax.set_xlabel("Time to Expiry T (years)")
ax.set_ylabel("Critical Stock Price S*")
ax.set_title("American Put Early Exercise Boundary\n"
             "Exercise immediately if S < S*(T)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m13_03_greeks_boundary.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
N_ref2 = 500
am_put_500, _, _ = crr_price(S0, K, T, r, sigma, N_ref2, "put", "american")
print("=" * 65)
print("  MODULE 13 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] CRR: u=exp(sig*sqrt(dt)), d=1/u, p=(exp(r*dt)-d)/(u-d)")
print("  [2] Backward induction: risk-neutral discounting")
print("  [3] American put: early exercise when S < S*(T)")
print(f"  [4] BS call = {bs_call:.6f}  |  BS put = {bs_put:.6f}")
print(f"  [5] CRR N=200 call error = {abs(crr_calls[-1]-bs_call):.6f}")
print(f"  [6] American put N=500 = {am_put_500:.6f}  "
      f"(premium = {am_put_500-bs_put:.6f})")
print("=" * 65)

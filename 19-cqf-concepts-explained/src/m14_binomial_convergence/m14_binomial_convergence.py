#!/usr/bin/env python3
"""
M14 — Binomial Tree Convergence & Acceleration Techniques
==========================================================
Module 3 of 9 | CQF Concepts Explained

Theory
------
The CRR tree converges to Black-Scholes at rate O(1/N), but with
pronounced odd/even oscillations. Several acceleration techniques
eliminate the oscillations and achieve O(1/N^2) or better:

1. Richardson Extrapolation (RE):
   Use two tree prices V(N) and V(2N):
       V_RE = 2*V(2N) - V(N)
   Cancels the leading O(1/N) error term => O(1/N^2) convergence.

2. Control Variate (CV):
   Price an analytically tractable "nearby" option (e.g. European)
   alongside the target (e.g. American) using the SAME tree:
       V_CV = V_tree_target + (V_BS_control - V_tree_control)
   Reduces variance/bias because tree errors are correlated.

3. Smooth Binomial (SB) / Broadie-Detemple:
   Choose N so the strike falls exactly on a node, or average
   over N and N+1 to cancel odd/even oscillation:
       V_SB = 0.5 * (V(N) + V(N+1))

4. Binomial Black-Scholes (BBS):
   Replace terminal payoffs with the BS continuation value at
   the penultimate step (one step before expiry). Eliminates
   the discretization of the terminal condition.

5. Leisen-Reimer (LR) tree:
   Uses Peizer-Pratt inversion to match BS d1/d2 exactly at
   each step. Achieves O(1/N^2) without extrapolation.

Author : Jose O. Bobadilla | CQF
Project: 19-cqf-concepts-explained
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Pricing engines
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def crr_european(S, K, T, r, sigma, N, opt="call"):
    """Standard CRR European option."""
    dt   = T / N
    u    = np.exp(sigma*np.sqrt(dt))
    d    = 1.0/u
    p    = (np.exp(r*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    j    = np.arange(N+1)
    ST   = S * u**j * d**(N-j)
    V    = np.maximum(ST - K, 0) if opt=="call" else np.maximum(K - ST, 0)
    for _ in range(N):
        V = disc * (p*V[1:] + (1-p)*V[:-1])
    return V[0]

def crr_american_put(S, K, T, r, sigma, N):
    """Standard CRR American put."""
    dt   = T / N
    u    = np.exp(sigma*np.sqrt(dt))
    d    = 1.0/u
    p    = (np.exp(r*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    j    = np.arange(N+1)
    ST   = S * u**j * d**(N-j)
    V    = np.maximum(K - ST, 0.0)
    for i in range(N-1, -1, -1):
        V = disc * (p*V[1:] + (1-p)*V[:-1])
        Si = S * u**np.arange(i+1) * d**(i-np.arange(i+1))
        V  = np.maximum(V, K - Si)
    return V[0]

def richardson(S, K, T, r, sigma, N, fn):
    """Richardson extrapolation: 2*V(2N) - V(N)."""
    return 2*fn(S, K, T, r, sigma, 2*N) - fn(S, K, T, r, sigma, N)

def smooth_binomial(S, K, T, r, sigma, N, fn):
    """Average V(N) and V(N+1) to cancel odd/even oscillation."""
    return 0.5*(fn(S, K, T, r, sigma, N) + fn(S, K, T, r, sigma, N+1))

def bbs_call(S, K, T, r, sigma, N):
    """
    Binomial Black-Scholes: replace terminal payoffs with
    BS value at penultimate step (N-1 -> expiry).
    """
    dt   = T / N
    u    = np.exp(sigma*np.sqrt(dt))
    d    = 1.0/u
    p    = (np.exp(r*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    # Penultimate stock prices
    j      = np.arange(N)
    S_pen  = S * u**j * d**(N-1-j)
    # Use BS for remaining dt instead of terminal payoff
    V = np.array([bs_call(s, K, dt, r, sigma) for s in S_pen])
    for _ in range(N-1):
        V = disc * (p*V[1:] + (1-p)*V[:-1])
    return V[0]

def lr_call(S, K, T, r, sigma, N):
    """
    Leisen-Reimer tree: Peizer-Pratt inversion for d1, d2.
    Achieves O(1/N^2) convergence.
    """
    def pp2(z, n):
        # Peizer-Pratt method 2 inversion
        return 0.5 + np.sign(z)*np.sqrt(
            0.25 - 0.25*np.exp(-(z/(n+1/3+0.1/(n+1)))**2 * (n+1/6))
        )
    dt   = T / N
    d1   = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2   = d1 - sigma*np.sqrt(T)
    p_   = pp2(d2, N)
    p1   = pp2(d1, N)
    u    = np.exp(r*dt) * p1 / p_
    d    = (np.exp(r*dt) - p_*u) / (1 - p_)
    disc = np.exp(-r*dt)
    j    = np.arange(N+1)
    ST   = S * u**j * d**(N-j)
    V    = np.maximum(ST - K, 0.0)
    for _ in range(N):
        V = disc * (p_*V[1:] + (1-p_)*V[:-1])
    return V[0]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20

BS_CALL = bs_call(S0, K, T, r, sigma)
BS_PUT  = bs_put(S0, K, T, r, sigma)
print(f"[M14] BS call = {BS_CALL:.8f}  |  BS put = {BS_PUT:.8f}")

N_vals = list(range(5, 201, 2))   # odd N to avoid even/odd ambiguity in some methods


# ===========================================================================
# FIGURE 1 — European Call: raw CRR vs acceleration methods
# ===========================================================================
print("[M14] Figure 1: European call convergence acceleration ...")
t0 = time.perf_counter()

crr_raw = np.array([crr_european(S0, K, T, r, sigma, n) for n in N_vals])
crr_re  = np.array([richardson(S0, K, T, r, sigma, n,
                                lambda *a: crr_european(*a)) for n in N_vals])
crr_sb  = np.array([smooth_binomial(S0, K, T, r, sigma, n,
                                     lambda *a: crr_european(*a)) for n in N_vals])
crr_bbs = np.array([bbs_call(S0, K, T, r, sigma, n) for n in N_vals])
crr_lr  = np.array([lr_call(S0, K, T, r, sigma, n)  for n in N_vals])

err_raw = np.abs(crr_raw - BS_CALL)
err_re  = np.abs(crr_re  - BS_CALL)
err_sb  = np.abs(crr_sb  - BS_CALL)
err_bbs = np.abs(crr_bbs - BS_CALL)
err_lr  = np.abs(crr_lr  - BS_CALL)
N_arr   = np.array(N_vals, float)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle(
    "M14 — European Call Convergence: CRR vs Acceleration Methods\n"
    f"S={S0}, K={K}, T={T}Y, r={r:.0%}, sigma={sigma:.0%}  |  BS={BS_CALL:.6f}",
    color=WHITE, fontsize=10
)

# (0) Price convergence
ax = axes[0]
ax.plot(N_vals, crr_raw, color=BLUE,   lw=1.2, alpha=0.8, label="CRR raw")
ax.plot(N_vals, crr_re,  color=GREEN,  lw=1.5, label="Richardson extrap.")
ax.plot(N_vals, crr_sb,  color=ORANGE, lw=1.5, label="Smooth (avg N, N+1)")
ax.plot(N_vals, crr_bbs, color=PURPLE, lw=1.5, label="BBS (BS terminal)")
ax.plot(N_vals, crr_lr,  color=YELLOW, lw=1.5, label="Leisen-Reimer")
ax.axhline(BS_CALL, color=RED, lw=2, linestyle="--", label=f"BS exact={BS_CALL:.4f}")
ax.set_xlabel("N (steps)"); ax.set_ylabel("Call Price")
ax.set_title("Price vs N\n(accelerated methods flatten quickly)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Error log-log
ax = axes[1]
# Fit slopes (use N > 20 to avoid startup noise)
mask = N_arr > 20
for err, col, lbl in [
    (err_raw, BLUE,   "CRR raw"),
    (err_re,  GREEN,  "Richardson"),
    (err_sb,  ORANGE, "Smooth"),
    (err_bbs, PURPLE, "BBS"),
    (err_lr,  YELLOW, "Leisen-Reimer"),
]:
    safe = err[mask] + 1e-12
    slope = np.polyfit(np.log(N_arr[mask]), np.log(safe), 1)[0]
    ax.loglog(N_arr, err + 1e-12, lw=1.5, color=col,
              label=f"{lbl}  (slope={slope:.2f})")

ref1 = N_arr**(-1.0) * (err_raw[5] / N_arr[5]**(-1.0))
ref2 = N_arr**(-2.0) * (err_lr[5]  / N_arr[5]**(-2.0))
ax.loglog(N_arr, ref1, "--", color=GREY,  lw=1.2, label="O(N^{-1})")
ax.loglog(N_arr, ref2, ":",  color=WHITE, lw=1.2, label="O(N^{-2})")
ax.set_xlabel("N"); ax.set_ylabel("|Error|")
ax.set_title("Absolute Error log-log\nAccelerated: O(N^{-2}) vs CRR O(N^{-1})",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m14_01_european_convergence.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — American Put: control variate + Richardson
# ===========================================================================
print("[M14] Figure 2: American put acceleration ...")
t0 = time.perf_counter()

# True American put reference (very large N)
AM_REF = crr_american_put(S0, K, T, r, sigma, 2000)
print(f"      American put reference (N=2000): {AM_REF:.8f}")

N_am = list(range(5, 151, 2))
N_am_arr = np.array(N_am, float)

crr_am_raw = np.array([crr_american_put(S0, K, T, r, sigma, n) for n in N_am])

# Richardson for American put
crr_am_re  = np.array([
    2*crr_american_put(S0, K, T, r, sigma, 2*n) - crr_american_put(S0, K, T, r, sigma, n)
    for n in N_am
])

# Control variate: American put + (BS European put - CRR European put)
crr_am_cv  = np.array([
    crr_american_put(S0, K, T, r, sigma, n)
    + (BS_PUT - crr_european(S0, K, T, r, sigma, n, "put"))
    for n in N_am
])

# Smooth binomial for American put
crr_am_sb  = np.array([
    0.5*(crr_american_put(S0, K, T, r, sigma, n)
       + crr_american_put(S0, K, T, r, sigma, n+1))
    for n in N_am
])

err_am_raw = np.abs(crr_am_raw - AM_REF)
err_am_re  = np.abs(crr_am_re  - AM_REF)
err_am_cv  = np.abs(crr_am_cv  - AM_REF)
err_am_sb  = np.abs(crr_am_sb  - AM_REF)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle(
    f"M14 — American Put Convergence & Acceleration\n"
    f"Reference price (N=2000) = {AM_REF:.6f}  |  European put BS = {BS_PUT:.6f}  "
    f"(early exercise premium = {AM_REF-BS_PUT:.6f})",
    color=WHITE, fontsize=10
)

ax = axes[0]
ax.plot(N_am, crr_am_raw, color=BLUE,   lw=1.2, alpha=0.8, label="CRR raw")
ax.plot(N_am, crr_am_re,  color=GREEN,  lw=1.5, label="Richardson extrap.")
ax.plot(N_am, crr_am_cv,  color=ORANGE, lw=1.5, label="Control variate")
ax.plot(N_am, crr_am_sb,  color=YELLOW, lw=1.5, label="Smooth (avg N, N+1)")
ax.axhline(AM_REF, color=RED, lw=2, linestyle="--",
           label=f"Reference={AM_REF:.4f}")
ax.set_xlabel("N"); ax.set_ylabel("American Put Price")
ax.set_title("Price vs N (American Put)", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

ax = axes[1]
mask2 = N_am_arr > 15
for err, col, lbl in [
    (err_am_raw, BLUE,   "CRR raw"),
    (err_am_re,  GREEN,  "Richardson"),
    (err_am_cv,  ORANGE, "Control variate"),
    (err_am_sb,  YELLOW, "Smooth"),
]:
    safe = err[mask2] + 1e-12
    slope = np.polyfit(np.log(N_am_arr[mask2]), np.log(safe), 1)[0]
    ax.loglog(N_am_arr, err + 1e-12, lw=1.5, color=col,
              label=f"{lbl}  (slope={slope:.2f})")
ref1b = N_am_arr**(-1.0) * (err_am_raw[5] / N_am_arr[5]**(-1.0))
ax.loglog(N_am_arr, ref1b, "--", color=GREY, lw=1.2, label="O(N^{-1})")
ax.set_xlabel("N"); ax.set_ylabel("|Error vs reference|")
ax.set_title("Absolute Error log-log\n(American put, reference N=2000)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m14_02_american_acceleration.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Moneyness sensitivity + speed vs accuracy tradeoff
# ===========================================================================
print("[M14] Figure 3: Moneyness sensitivity and speed/accuracy tradeoff ...")
t0 = time.perf_counter()

# Error vs moneyness (K varies) for fixed N=50
N_fix  = 50
K_range = np.linspace(70, 130, 31)
err_mono_raw = []
err_mono_lr  = []
err_mono_re  = []

for k in K_range:
    bs_c  = bs_call(S0, k, T, r, sigma)
    c_raw = crr_european(S0, k, T, r, sigma, N_fix)
    c_lr  = lr_call(S0, k, T, r, sigma, N_fix)
    c_re  = richardson(S0, k, T, r, sigma, N_fix,
                       lambda *a: crr_european(*a))
    err_mono_raw.append(abs(c_raw - bs_c))
    err_mono_lr.append(abs(c_lr  - bs_c))
    err_mono_re.append(abs(c_re  - bs_c))

# Speed vs accuracy: time per price vs N
N_speed = [10, 25, 50, 100, 200, 500, 1000]
times_crr = []
errors_crr = []
for n in N_speed:
    t_s = time.perf_counter()
    for _ in range(20):
        p = crr_european(S0, K, T, r, sigma, n)
    times_crr.append((time.perf_counter()-t_s)/20 * 1000)
    errors_crr.append(abs(p - BS_CALL))

# LR speed (same N but O(N^2) accuracy)
times_lr  = []
errors_lr = []
for n in N_speed:
    t_s = time.perf_counter()
    for _ in range(20):
        p = lr_call(S0, K, T, r, sigma, n)
    times_lr.append((time.perf_counter()-t_s)/20 * 1000)
    errors_lr.append(abs(p - BS_CALL))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("M14 — Moneyness Sensitivity and Speed/Accuracy Tradeoff",
             color=WHITE, fontsize=11)

# (0) Error vs moneyness
ax = axes[0]
ax.semilogy(K_range, err_mono_raw, "o-", color=BLUE,   lw=1.5, ms=5,
            label=f"CRR raw  (N={N_fix})")
ax.semilogy(K_range, err_mono_re,  "s-", color=GREEN,  lw=1.5, ms=5,
            label=f"Richardson (N={N_fix})")
ax.semilogy(K_range, err_mono_lr,  "^-", color=YELLOW, lw=1.5, ms=5,
            label=f"Leisen-Reimer (N={N_fix})")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.7, label=f"ATM S={S0}")
ax.set_xlabel("Strike K"); ax.set_ylabel("|Error|  (log scale)")
ax.set_title(f"Pricing Error vs Moneyness  (N={N_fix})\n"
             "LR and RE dominate across all strikes",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Time per call (ms) vs N
ax = axes[1]
ax.loglog(N_speed, times_crr, "o-", color=BLUE,   lw=2, ms=6, label="CRR raw")
ax.loglog(N_speed, times_lr,  "^-", color=YELLOW, lw=2, ms=6, label="Leisen-Reimer")
ref_n2 = np.array(N_speed, float)**2 * (times_crr[0]/N_speed[0]**2)
ax.loglog(N_speed, ref_n2, "--", color=GREY, lw=1.2, label="O(N^2) expected")
ax.set_xlabel("N"); ax.set_ylabel("Time per call (ms)")
ax.set_title("Computation Time vs N\n(O(N^2) cost from backward induction)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Error vs computation time (Pareto frontier)
ax = axes[2]
ax.loglog(times_crr, [e+1e-12 for e in errors_crr], "o-", color=BLUE,   lw=2, ms=6,
          label="CRR raw")
ax.loglog(times_lr,  [e+1e-12 for e in errors_lr],  "^-", color=YELLOW, lw=2, ms=6,
          label="Leisen-Reimer")
for n, tc, ec in zip(N_speed[::2], times_crr[::2], errors_crr[::2]):
    ax.annotate(f"N={n}", (tc, ec+1e-12), textcoords="offset points",
                xytext=(4, 4), fontsize=6, color=BLUE)
for n, tl, el in zip(N_speed[::2], times_lr[::2], errors_lr[::2]):
    ax.annotate(f"N={n}", (tl, el+1e-12), textcoords="offset points",
                xytext=(4, -10), fontsize=6, color=YELLOW)
ax.set_xlabel("Computation time (ms)")
ax.set_ylabel("|Error|")
ax.set_title("Error vs Computation Time\n(LR dominates: same time, less error)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m14_03_speed_accuracy.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 14 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] CRR raw: O(N^{-1}) with odd/even oscillation")
print("  [2] Richardson: 2*V(2N)-V(N) => O(N^{-2})")
print("  [3] Smooth: avg(V(N),V(N+1)) cancels oscillation")
print("  [4] BBS: BS terminal => removes terminal discretization error")
print("  [5] Leisen-Reimer: Peizer-Pratt => O(N^{-2}) directly")
print("  [6] Control variate: uses BS European as correlated benchmark")
print(f"  [7] LR N=50 error  = {abs(lr_call(S0,K,T,r,sigma,50)-BS_CALL):.8f}")
print(f"      CRR N=50 error = {abs(crr_european(S0,K,T,r,sigma,50)-BS_CALL):.8f}")
print("=" * 65)

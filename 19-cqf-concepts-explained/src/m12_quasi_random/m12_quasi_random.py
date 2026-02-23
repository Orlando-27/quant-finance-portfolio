#!/usr/bin/env python3
"""
M12 — Quasi-Random Sequences: Sobol and Halton
===============================================
Module 2 of 9 | CQF Concepts Explained

Theory
------
Standard Monte Carlo (MC) uses pseudo-random numbers:
    Error ~ O(N^{-1/2})  — convergence is slow and stochastic.

Quasi-Monte Carlo (QMC) uses low-discrepancy sequences (LDS) that
fill the unit hypercube more uniformly than random samples:
    Error ~ O((log N)^d / N)  — faster convergence in low dimensions.

Discrepancy measures how far a point set deviates from perfect
uniformity. Low discrepancy => faster convergence of integrals.

Halton Sequence (1960):
    Built from Van der Corput sequences in coprime bases b_1, b_2, ...
    x_n^(b) = sum_{k} d_k * b^{-(k+1)}   where n = sum d_k * b^k
    Simple to implement; suffers from correlation in high dimensions (d > 10).

Sobol Sequence (1967):
    Uses base-2 construction with direction numbers chosen to minimize
    discrepancy. Better uniformity than Halton in higher dimensions.
    Requires initialization from precomputed direction numbers.

Box-Muller transform (converts uniform [0,1] to N(0,1)):
    Z_1 = sqrt(-2 ln U_1) * cos(2*pi*U_2)
    Z_2 = sqrt(-2 ln U_1) * sin(2*pi*U_2)

Application — European call pricing:
    C = e^{-rT} * E^Q[max(S_T - K, 0)]
    Exact (Black-Scholes): C_BS
    QMC converges to C_BS faster than MC with same N.

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
from scipy.stats.qmc import Halton, Sobol

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
# Helpers
# ---------------------------------------------------------------------------
def van_der_corput(n, base):
    """Van der Corput sequence in given base, length n."""
    seq = np.zeros(n)
    for i in range(n):
        num, denom = i + 1, 1
        while num > 0:
            denom *= base
            num, remainder = divmod(num, base)
            seq[i] += remainder / denom
    return seq

def halton_2d(n):
    """2D Halton sequence using bases 2 and 3."""
    return np.column_stack([van_der_corput(n, 2),
                            van_der_corput(n, 3)])

def uniform_to_normal(u):
    """Box-Muller transform: (N,2) uniform -> (N,2) standard normal."""
    u1, u2 = np.clip(u[:, 0], 1e-12, 1-1e-12), u[:, 1]
    r = np.sqrt(-2 * np.log(u1))
    return np.column_stack([r * np.cos(2*np.pi*u2),
                            r * np.sin(2*np.pi*u2)])

def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def mc_call_price(S, K, T, r, sigma, Z):
    """Monte Carlo call price given standard normal draws Z (1-D)."""
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0.0)
    return np.exp(-r*T) * payoff.mean()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 105.0
T     = 1.0
r     = 0.05
sigma = 0.20
SEED  = 42

C_exact = bs_call(S0, K, T, r, sigma)
print(f"[M12] Black-Scholes exact call price: {C_exact:.6f}")


# ===========================================================================
# FIGURE 1 — Point sets in 2D: MC vs Halton vs Sobol
# ===========================================================================
print("[M12] Figure 1: 2D point set comparison ...")
t0 = time.perf_counter()

N_pts = 1024
rng   = np.random.default_rng(SEED)

pts_mc     = rng.uniform(0, 1, (N_pts, 2))
pts_halton = halton_2d(N_pts)
sobol_eng  = Sobol(d=2, scramble=True, seed=SEED)
pts_sobol  = sobol_eng.random(N_pts)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
fig.suptitle(
    f"M12 — Point Sets in [0,1]^2  (N={N_pts})\n"
    "MC (pseudo-random) vs Halton vs Sobol (low-discrepancy)",
    color=WHITE, fontsize=11
)

datasets = [
    (pts_mc,     BLUE,   "Pseudo-Random MC\n(uneven clusters and gaps)"),
    (pts_halton, GREEN,  "Halton Sequence (bases 2,3)\n(structured, uniform coverage)"),
    (pts_sobol,  YELLOW, "Sobol Sequence (scrambled)\n(best uniformity, base-2)"),
]

for ax, (pts, col, title) in zip(axes, datasets):
    ax.scatter(pts[:, 0], pts[:, 1], s=3, color=col, alpha=0.6,
               rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2")
    ax.set_title(title, color=WHITE, fontsize=9)
    ax.set_aspect("equal"); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m12_01_point_sets.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Convergence: MC vs QMC for European Call Pricing
# ===========================================================================
print("[M12] Figure 2: Convergence study ...")
t0 = time.perf_counter()

# Use powers of 2 for fair Sobol comparison
N_vals = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
N_REPS = 30    # MC repetitions to estimate variance

errors_mc     = []   # mean absolute error over N_REPS runs
errors_mc_std = []
errors_halton = []
errors_sobol  = []

for N in N_vals:
    # --- MC: N_REPS independent runs ---
    mc_prices = []
    for rep in range(N_REPS):
        Z = np.random.default_rng(SEED + rep).standard_normal(N)
        mc_prices.append(mc_call_price(S0, K, T, r, sigma, Z))
    mc_prices = np.array(mc_prices)
    errors_mc.append(np.abs(mc_prices - C_exact).mean())
    errors_mc_std.append(np.abs(mc_prices - C_exact).std())

    # --- Halton ---
    u_h  = halton_2d(N)
    Z_h  = uniform_to_normal(u_h)[:, 0]
    errors_halton.append(abs(mc_call_price(S0, K, T, r, sigma, Z_h) - C_exact))

    # --- Sobol (scrambled) ---
    eng  = Sobol(d=2, scramble=True, seed=SEED)
    u_s  = eng.random(N)
    Z_s  = uniform_to_normal(u_s)[:, 0]
    errors_sobol.append(abs(mc_call_price(S0, K, T, r, sigma, Z_s) - C_exact))

errors_mc     = np.array(errors_mc)
errors_mc_std = np.array(errors_mc_std)
errors_halton = np.array(errors_halton)
errors_sobol  = np.array(errors_sobol)
N_arr         = np.array(N_vals, dtype=float)

# Fit slopes in log-log
slope_mc  = np.polyfit(np.log(N_arr), np.log(errors_mc    + 1e-12), 1)[0]
slope_h   = np.polyfit(np.log(N_arr), np.log(errors_halton+ 1e-12), 1)[0]
slope_s   = np.polyfit(np.log(N_arr), np.log(errors_sobol + 1e-12), 1)[0]

# Reference lines
ref_half = N_arr**(-0.5) * (errors_mc[0] / N_arr[0]**(-0.5))
ref_one  = N_arr**(-1.0) * (errors_sobol[0] / N_arr[0]**(-1.0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("M12 — Convergence: MC vs QMC for European Call Pricing\n"
             f"S={S0}, K={K}, T={T}Y, r={r:.0%}, sigma={sigma:.0%}  |  "
             f"Exact C_BS = {C_exact:.6f}",
             color=WHITE, fontsize=10)

# (0) Log-log convergence
ax = axes[0]
ax.loglog(N_arr, errors_mc,     "o-", color=BLUE,   lw=2, ms=6,
          label=f"MC pseudo-random  (slope={slope_mc:.2f})")
ax.fill_between(N_arr,
                errors_mc - errors_mc_std,
                errors_mc + errors_mc_std,
                color=BLUE, alpha=0.15, label="MC +/- 1 std")
ax.loglog(N_arr, errors_halton,  "s-", color=GREEN,  lw=2, ms=6,
          label=f"QMC Halton       (slope={slope_h:.2f})")
ax.loglog(N_arr, errors_sobol,   "^-", color=YELLOW, lw=2, ms=6,
          label=f"QMC Sobol        (slope={slope_s:.2f})")
ax.loglog(N_arr, ref_half, "--", color=BLUE,   alpha=0.5, lw=1.2,
          label=r"$O(N^{-1/2})$ reference")
ax.loglog(N_arr, ref_one,  "--", color=YELLOW, alpha=0.5, lw=1.2,
          label=r"$O(N^{-1})$ reference")
ax.set_xlabel("Number of Samples N"); ax.set_ylabel("|Estimated - Exact|")
ax.set_title("Absolute Error vs N  (log-log)\n"
             "QMC achieves ~O(N^{-1}) vs MC O(N^{-1/2})",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Efficiency ratio: MC error / QMC error
ax = axes[1]
ratio_h = errors_mc / (errors_halton + 1e-12)
ratio_s = errors_mc / (errors_sobol  + 1e-12)
ax.semilogx(N_arr, ratio_h, "s-", color=GREEN,  lw=2, ms=6,
            label="MC error / Halton error")
ax.semilogx(N_arr, ratio_s, "^-", color=YELLOW, lw=2, ms=6,
            label="MC error / Sobol error")
ax.axhline(1, color=WHITE, lw=1, linestyle=":", alpha=0.6,
           label="Ratio = 1  (equal accuracy)")
ax.fill_between(N_arr, 1, np.maximum(ratio_s, 1),
                color=YELLOW, alpha=0.10, label="QMC advantage region")
ax.set_xlabel("Number of Samples N")
ax.set_ylabel("Efficiency ratio  (>1 = QMC better)")
ax.set_title("QMC Efficiency Gain over MC\n"
             "Ratio = |error_MC| / |error_QMC|",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m12_02_convergence.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Uniformity: empirical CDF gap and discrepancy
# ===========================================================================
print("[M12] Figure 3: Uniformity and discrepancy ...")
t0 = time.perf_counter()

N_unif = 2048
rng_u  = np.random.default_rng(SEED + 5)
mc_1d  = rng_u.uniform(0, 1, N_unif)
hal_1d = van_der_corput(N_unif, 2)
sob_eng = Sobol(d=1, scramble=True, seed=SEED)
sob_1d  = sob_eng.random(N_unif).flatten()

# Star discrepancy (1D: max |empirical CDF - uniform CDF|)
def star_discrepancy_1d(pts):
    n = len(pts)
    s = np.sort(pts)
    emp_cdf = np.arange(1, n+1) / n
    uni_cdf = s
    return max(np.abs(emp_cdf - uni_cdf).max(),
               np.abs(np.arange(n)/n - uni_cdf).max())

d_mc  = star_discrepancy_1d(mc_1d)
d_hal = star_discrepancy_1d(hal_1d)
d_sob = star_discrepancy_1d(sob_1d)

# Discrepancy vs N (growing sequence)
N_disc = np.array([64, 128, 256, 512, 1024, 2048])
disc_mc  = []; disc_hal = []; disc_sob = []
for nd in N_disc:
    rng_nd = np.random.default_rng(SEED + nd)
    disc_mc.append(star_discrepancy_1d(rng_nd.uniform(0,1,nd)))
    disc_hal.append(star_discrepancy_1d(van_der_corput(nd, 2)))
    se = Sobol(d=1, scramble=True, seed=SEED)
    disc_sob.append(star_discrepancy_1d(se.random(nd).flatten()))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(f"M12 — Uniformity and Star Discrepancy  (N={N_unif})",
             color=WHITE, fontsize=11)

# (0) 1D ECDF comparison
ax = axes[0]
x_ref = np.linspace(0, 1, 500)
for pts, col, lbl, disc in [
    (mc_1d,  BLUE,   "MC pseudo-random", d_mc),
    (hal_1d, GREEN,  "Halton (base 2)",  d_hal),
    (sob_1d, YELLOW, "Sobol scrambled",  d_sob),
]:
    s = np.sort(pts)
    ax.step(s, np.arange(1, N_unif+1)/N_unif, color=col, lw=1.5,
            label=f"{lbl}  D*={disc:.4f}", where="post")
ax.plot(x_ref, x_ref, color=WHITE, lw=2, linestyle="--",
        label="Uniform CDF (ideal)")
ax.set_xlabel("x"); ax.set_ylabel("Empirical CDF")
ax.set_title("Empirical CDF vs Uniform\n(gap = discrepancy)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Histogram of 1D samples
ax = axes[1]
bins = np.linspace(0, 1, 33)
ax.hist(mc_1d,  bins=bins, density=True, color=BLUE,   alpha=0.50,
        edgecolor="none", label=f"MC  D*={d_mc:.4f}")
ax.hist(hal_1d, bins=bins, density=True, color=GREEN,  alpha=0.50,
        edgecolor="none", label=f"Halton  D*={d_hal:.4f}")
ax.hist(sob_1d, bins=bins, density=True, color=YELLOW, alpha=0.50,
        edgecolor="none", label=f"Sobol  D*={d_sob:.4f}")
ax.axhline(1.0, color=WHITE, lw=2, linestyle=":", label="Ideal uniform density")
ax.set_xlabel("x"); ax.set_ylabel("Density")
ax.set_title("Histogram of 1D Samples\n(LDS fills bins more evenly)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Discrepancy vs N
ax = axes[2]
ax.loglog(N_disc, disc_mc,  "o-", color=BLUE,   lw=2, ms=6, label="MC pseudo-random")
ax.loglog(N_disc, disc_hal, "s-", color=GREEN,  lw=2, ms=6, label="Halton (base 2)")
ax.loglog(N_disc, disc_sob, "^-", color=YELLOW, lw=2, ms=6, label="Sobol scrambled")
ref_half_d = np.array(N_disc, float)**(-0.5) * (disc_mc[0] / N_disc[0]**(-0.5))
ref_one_d  = np.array(N_disc, float)**(-1.0) * (disc_hal[0] / N_disc[0]**(-1.0))
ax.loglog(N_disc, ref_half_d, "--", color=BLUE,   alpha=0.5, lw=1.2,
          label=r"$O(N^{-1/2})$")
ax.loglog(N_disc, ref_one_d,  "--", color=GREEN,  alpha=0.5, lw=1.2,
          label=r"$O(N^{-1})$")
ax.set_xlabel("N"); ax.set_ylabel("Star Discrepancy D*")
ax.set_title("1D Star Discrepancy vs N\n"
             "LDS achieves O(N^{-1}) vs MC O(N^{-1/2})",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m12_03_discrepancy.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 12 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] MC error: O(N^{-1/2}) -- dimension-independent but slow")
print("  [2] QMC error: O((log N)^d / N) -- faster in low dimensions")
print("  [3] Halton: Van der Corput in coprime bases 2,3,5,...")
print("  [4] Sobol: base-2, direction numbers, best uniformity")
print("  [5] Box-Muller: uniform -> N(0,1) via LDS for QMC pricing")
print(f"  [6] BS exact = {C_exact:.6f}")
print(f"      Sobol N=16384 error = {errors_sobol[-1]:.8f}")
print(f"      MC    N=16384 error = {errors_mc[-1]:.8f}")
print("=" * 65)

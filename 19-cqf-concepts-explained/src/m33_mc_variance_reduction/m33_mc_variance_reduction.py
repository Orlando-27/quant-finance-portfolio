#!/usr/bin/env python3
"""
M33 — Monte Carlo for Options: Variance Reduction Techniques
=============================================================
Module 33 | CQF Concepts Explained
Group 7   | Derivatives Pricing

Theory
------
Standard Monte Carlo (Plain MC)
---------------------------------
Under risk-neutral measure Q, the terminal asset price is:
    S_T = S_0 * exp((r - q - sigma^2/2)*T + sigma*sqrt(T)*Z)
    Z ~ N(0,1)

Option price:
    C = e^{-rT} * E^Q[max(S_T - K, 0)]
    C_MC = e^{-rT} * (1/N) * sum_{i=1}^N max(S_T^i - K, 0)

Standard error: SE = sigma_payoff / sqrt(N)
Convergence rate: O(1/sqrt(N)) -- need 100x more paths for 10x accuracy.

Variance Reduction Techniques
-------------------------------

1. Antithetic Variates
-----------------------
For each Z_i ~ N(0,1), also simulate -Z_i:
    S_T^+ = S_0 * exp(mu*T + sigma*sqrt(T)* Z_i)
    S_T^- = S_0 * exp(mu*T + sigma*sqrt(T)*(-Z_i))
Estimator: C_AV = e^{-rT} * (1/(2N)) * sum [f(S_T^+) + f(S_T^-)]

Variance reduction: Var(AV) = Var(MC)/2 * (1 + rho(f^+, f^-))
For monotone payoffs (rho < 0): significant variance reduction.

2. Control Variates
--------------------
Use a correlated variable X with known expectation E[X]:
    C_CV = C_MC + c*(X_bar - E[X])
Optimal c* = -Cov(f, X) / Var(X)
Variance reduction factor: 1 - rho_{f,X}^2

For options: use the geometric average Asian as control for
arithmetic average Asian (geometric has closed-form).
For European: use the asset price S_T itself (E[S_T] = S_0*e^{rT}).

3. Importance Sampling
-----------------------
Change measure to concentrate simulations in the important region.
For OTM options: shift the mean of Z toward the exercise region.
    Z' = Z + mu_IS    (drift toward K)
Radon-Nikodym derivative:
    dP/dQ = exp(-mu_IS*Z - mu_IS^2/2)
    C_IS = e^{-rT} * E^P[f(S_T)*dP/dQ]

4. Quasi-Monte Carlo (Sobol Sequences)
----------------------------------------
Replace pseudo-random Z_i with low-discrepancy Sobol sequence.
Sobol sequences fill space more uniformly than pseudo-random.
Convergence rate: O((log N)^d / N) vs O(1/sqrt(N)) for MC.
In practice: QMC often achieves O(1/N) for smooth integrands.

Stratified Sampling
--------------------
Divide [0,1] into N equal strata, sample one point per stratum:
    U_i = (i - U_rand) / N,  i = 1,...,N
Eliminates clumping, reduces variance for monotone functions.

References
----------
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
  Springer. (The definitive reference — Chapter 4 on variance reduction)
- Boyle, P. (1977). Options: a Monte Carlo approach. Journal of
  Financial Economics, 4(3), 323-338. (First MC option pricing paper)
- Niederreiter, H. (1992). Random Number Generation and Quasi-Monte
  Carlo Methods. SIAM.
- Bratley, P., Fox, B.L. (1988). Algorithm 659: implementing Sobol's
  quasirandom sequence generator. ACM TOMS, 14(1), 88-100.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a";  PANEL  = "#111111"; GRID   = "#1e1e1e"
WHITE  = "#e8e8e8";  BLUE   = "#4a9eff"; GREEN  = "#00d4aa"
ORANGE = "#ff8c42";  RED    = "#ff4757"; PURPLE = "#a855f7"
YELLOW = "#ffd700";  CYAN   = "#00bcd4"

WATERMARK = "Jose O. Bobadilla | CQF"
OUT_DIR   = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": DARK,   "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,   "axes.labelcolor":  WHITE,
    "axes.titlecolor":  WHITE,  "xtick.color":      WHITE,
    "ytick.color":      WHITE,  "text.color":       WHITE,
    "grid.color":       GRID,   "grid.linewidth":   0.6,
    "legend.facecolor": PANEL,  "legend.edgecolor": GRID,
    "font.family":      "monospace",
    "axes.spines.top":  False,  "axes.spines.right": False,
})

def watermark(ax):
    ax.text(0.99, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=WHITE, alpha=0.35, ha="right", va="bottom",
            fontstyle="italic")

# ============================================================
# SECTION 1 — CORE PARAMETERS AND BS REFERENCE
# ============================================================

S0 = 100.0; K = 100.0; r = 0.05; T = 1.0; sigma = 0.20; q = 0.0

def bs_call(S, K, r, T, sigma, q=0.0):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

BS_PRICE = bs_call(S0, K, r, T, sigma, q)
MU_LOG   = (r - q - 0.5*sigma**2)*T   # log-return drift
SIGMA_LOG = sigma * np.sqrt(T)

print(f"[M33] BS reference price: {BS_PRICE:.6f}")

# ============================================================
# SECTION 2 — PRICER FUNCTIONS
# ============================================================

def mc_standard(N, seed=0):
    """Plain Monte Carlo European call."""
    rng  = np.random.default_rng(seed)
    Z    = rng.standard_normal(N)
    S_T  = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
    pay  = np.maximum(S_T - K, 0)
    disc = np.exp(-r * T)
    return disc * pay.mean(), disc * pay.std() / np.sqrt(N)

def mc_antithetic(N, seed=0):
    """Antithetic variates: use Z and -Z pairs."""
    rng  = np.random.default_rng(seed)
    Z    = rng.standard_normal(N)         # N draws
    S_p  = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
    S_m  = S0 * np.exp(MU_LOG - SIGMA_LOG * Z)
    pay  = 0.5 * (np.maximum(S_p-K, 0) + np.maximum(S_m-K, 0))
    disc = np.exp(-r * T)
    return disc * pay.mean(), disc * pay.std() / np.sqrt(N)

def mc_control_variate(N, seed=0):
    """
    Control variate: use S_T (asset price) as control.
    E^Q[S_T] = S_0 * exp((r-q)*T) = S_0*e^{rT} (for q=0).
    c* = -Cov(payoff, S_T) / Var(S_T)
    """
    rng    = np.random.default_rng(seed)
    Z      = rng.standard_normal(N)
    S_T    = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
    pay    = np.maximum(S_T - K, 0)
    E_S_T  = S0 * np.exp(r * T)          # known expectation
    # Optimal c
    cov_pS = np.cov(pay, S_T)[0, 1]
    var_S  = np.var(S_T)
    c_star = -cov_pS / var_S
    pay_cv = pay + c_star * (S_T - E_S_T)
    disc   = np.exp(-r * T)
    return disc * pay_cv.mean(), disc * pay_cv.std() / np.sqrt(N)

def mc_importance_sampling(N, seed=0):
    """
    Importance sampling: shift mean toward K.
    Optimal shift: mu_IS = ln(K/S_0*exp(-MU_LOG)) / SIGMA_LOG
    (shifts Z distribution so E[S_T] ~ K).
    """
    rng    = np.random.default_rng(seed)
    # Optimal drift toward the strike
    mu_IS  = (np.log(K/S0) - MU_LOG) / SIGMA_LOG
    Z      = rng.standard_normal(N)
    Z_IS   = Z + mu_IS
    S_T    = S0 * np.exp(MU_LOG + SIGMA_LOG * Z_IS)
    pay    = np.maximum(S_T - K, 0)
    # Likelihood ratio (Radon-Nikodym)
    LR     = np.exp(-mu_IS * Z - 0.5 * mu_IS**2)
    pay_IS = pay * LR
    disc   = np.exp(-r * T)
    return disc * pay_IS.mean(), disc * pay_IS.std() / np.sqrt(N)

def sobol_1d(N):
    """
    Simple 1D Sobol-like low-discrepancy sequence (Van der Corput base 2).
    For production use scipy.stats.qmc.Sobol.
    """
    seq = np.zeros(N)
    for i in range(N):
        n, bit, f = i + 1, 0, 0.5
        while n:
            if n & 1:
                seq[i] += f
            n >>= 1
            f *= 0.5
    return seq

def mc_qmc(N, seed=0):
    """Quasi-Monte Carlo using Van der Corput sequence -> standard normal."""
    U    = sobol_1d(N)
    U    = np.clip(U, 1e-6, 1-1e-6)
    Z    = norm.ppf(U)
    S_T  = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
    pay  = np.maximum(S_T - K, 0)
    disc = np.exp(-r * T)
    return disc * pay.mean(), disc * pay.std() / np.sqrt(N)

def mc_stratified(N, seed=0):
    """Stratified sampling: N equal strata on [0,1]."""
    rng  = np.random.default_rng(seed)
    i_arr = np.arange(N)
    U    = (i_arr + rng.uniform(0, 1, N)) / N
    U    = np.clip(U, 1e-6, 1-1e-6)
    Z    = norm.ppf(U)
    S_T  = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
    pay  = np.maximum(S_T - K, 0)
    disc = np.exp(-r * T)
    return disc * pay.mean(), disc * pay.std() / np.sqrt(N)

# ============================================================
# SECTION 3 — DIAGNOSTICS
# ============================================================

N_BASE = 10_000
methods = [
    ("Standard MC",         mc_standard),
    ("Antithetic",          mc_antithetic),
    ("Control Variate",     mc_control_variate),
    ("Importance Sampling", mc_importance_sampling),
    ("QMC (Van der Corput)",mc_qmc),
    ("Stratified",          mc_stratified),
]

print(f"\n[M33] Pricing comparison (N={N_BASE:,}, BS={BS_PRICE:.6f})")
print(f"{'Method':25s}  {'Price':>9s}  {'Std Err':>9s}  {'Error':>9s}  {'Var Ratio':>9s}")
print("-" * 70)

base_se = None
for name, fn in methods:
    price, se = fn(N_BASE)
    err = abs(price - BS_PRICE)
    if name == "Standard MC":
        base_se = se
    vr = (base_se / se)**2 if base_se and se > 0 else 1.0
    print(f"{name:25s}  {price:9.6f}  {se:9.6f}  {err:9.6f}  {vr:9.2f}x")

# ============================================================
# FIGURE 1 — Convergence comparison
# ============================================================
t0 = time.perf_counter()
print("\n[M33] Figure 1: Convergence of MC methods ...")

N_arr = np.logspace(1, 5, 40).astype(int)
# Run each method multiple seeds for confidence bands
N_SEEDS = 10
results_all = {nm: [] for nm, _ in methods[:4]}

for N_ in N_arr:
    for nm, fn in methods[:4]:
        prices = [fn(N_, seed=s)[0] for s in range(N_SEEDS)]
        results_all[nm].append(prices)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M33 — Monte Carlo Variance Reduction\n"
             "Convergence to Black-Scholes | Standard Error Comparison",
             color=WHITE, fontsize=12, fontweight="bold")

colors_m = [BLUE, GREEN, ORANGE, RED]

# (a) Price convergence (mean across seeds)
ax = axes[0]
ax.axhline(BS_PRICE, color=WHITE, lw=2, linestyle="--",
           label=f"BS price = {BS_PRICE:.4f}")
for (nm, _), col in zip(methods[:4], colors_m):
    means = [np.mean(results_all[nm][i]) for i in range(len(N_arr))]
    stds  = [np.std(results_all[nm][i])  for i in range(len(N_arr))]
    ax.semilogx(N_arr, means, color=col, lw=2, label=nm)
    ax.fill_between(N_arr,
                    np.array(means)-np.array(stds),
                    np.array(means)+np.array(stds),
                    color=col, alpha=0.12)
ax.set_xlabel("N (log scale)"); ax.set_ylabel("Estimated price")
ax.set_title("Price Convergence (10 seeds, ± 1 std)\nAll converge to BS",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) Standard error vs N (log-log)
ax = axes[1]
for (nm, fn), col in zip(methods, colors_m + [PURPLE, YELLOW]):
    se_arr = [fn(n)[1] for n in N_arr]
    ax.loglog(N_arr, se_arr, color=col, lw=2, label=nm)
# Reference lines
N_ref = np.array([N_arr[0], N_arr[-1]], dtype=float)
ax.loglog(N_ref, BS_PRICE*0.5/np.sqrt(N_ref), color=WHITE,
          lw=1.5, linestyle=":", alpha=0.6, label="O(1/sqrt(N))")
ax.set_xlabel("N (log scale)"); ax.set_ylabel("Std Error (log scale)")
ax.set_title("Standard Error vs N (log-log)\nSlope=-0.5 => O(1/sqrt(N))",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (c) Variance reduction ratios at N=10,000
ax = axes[2]
vr_vals = []
se_std  = mc_standard(10000)[1]
for nm, fn in methods:
    se_ = fn(10000)[1]
    vr_vals.append((se_std / se_)**2 if se_ > 1e-10 else 1.0)

bar_cols = [BLUE, GREEN, ORANGE, RED, PURPLE, YELLOW]
bars = ax.bar([nm.replace(" ", "\n") for nm, _ in methods],
              vr_vals, color=bar_cols, alpha=0.85)
ax.axhline(1.0, color=WHITE, lw=1.5, linestyle="--",
           label="Baseline (Standard MC)")
for bar_, v_ in zip(bars, vr_vals):
    ax.text(bar_.get_x()+bar_.get_width()/2, v_+0.1,
            f"{v_:.1f}x", ha="center", fontsize=8, color=WHITE)
ax.set_ylabel("Variance reduction ratio (vs Standard MC)")
ax.set_title("Variance Reduction at N=10,000\n"
             "Ratio > 1: fewer paths needed for same accuracy",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m33_01_convergence.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Antithetic and control variate mechanics
# ============================================================
t0 = time.perf_counter()
print("[M33] Figure 2: Antithetic and control variate mechanics ...")

rng   = np.random.default_rng(42)
N_VIZ = 5000
Z     = rng.standard_normal(N_VIZ)
S_p   = S0 * np.exp(MU_LOG + SIGMA_LOG * Z)
S_m   = S0 * np.exp(MU_LOG - SIGMA_LOG * Z)
pay_p = np.maximum(S_p - K, 0)
pay_m = np.maximum(S_m - K, 0)
pay_av = 0.5 * (pay_p + pay_m)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M33 — Antithetic Variates and Control Variates\n"
             "Mechanics of Variance Reduction",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Antithetic: scatter of f(Z) vs f(-Z)
ax = axes[0]
ax.scatter(pay_p[:300], pay_m[:300], s=5, alpha=0.4, color=BLUE)
ax.set_xlabel("Payoff f(Z)"); ax.set_ylabel("Payoff f(-Z)")
corr_av = np.corrcoef(pay_p, pay_m)[0,1]
ax.set_title(f"Antithetic: f(Z) vs f(-Z)\n"
             f"Corr = {corr_av:.4f}  (negative => variance reduced)",
             color=WHITE, fontsize=9)
ax.grid(True); watermark(ax)

# (b) Payoff distributions: Standard vs Antithetic
ax = axes[1]
disc = np.exp(-r*T)
ax.hist(disc*pay_p, bins=50, density=True, alpha=0.45,
        color=BLUE, label=f"Standard MC (std={disc*pay_p.std():.4f})")
ax.hist(disc*pay_av, bins=50, density=True, alpha=0.45,
        color=GREEN, label=f"Antithetic (std={disc*pay_av.std():.4f})")
ax.axvline(BS_PRICE, color=ORANGE, lw=2, linestyle="--",
           label=f"BS = {BS_PRICE:.4f}")
ax.set_xlabel("Discounted payoff"); ax.set_ylabel("Density")
ax.set_title("Payoff Distribution Comparison\n"
             "Antithetic has lower variance => narrower histogram",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Control variate: payoff vs S_T scatter with regression
ax = axes[2]
E_S_T = S0 * np.exp(r*T)
cov_pS = np.cov(pay_p, S_p)[0,1]
var_S  = np.var(S_p)
c_star = -cov_pS / var_S
pay_cv_pts = pay_p + c_star * (S_p - E_S_T)

ax.scatter(S_p[:500], pay_p[:500], s=5, alpha=0.3,
           color=BLUE, label="Raw payoff f(S_T)")
ax.scatter(S_p[:500], pay_cv_pts[:500], s=5, alpha=0.3,
           color=GREEN, label="After CV adjustment")
S_line = np.linspace(S_p.min(), S_p.max(), 200)
ax.plot(S_line, np.maximum(S_line-K, 0), color=ORANGE, lw=2,
        label="Payoff function")
ax.set_xlabel("Terminal price S_T"); ax.set_ylabel("Discounted payoff")
corr_cv = np.corrcoef(pay_p, S_p)[0,1]
ax.set_title(f"Control Variate: Using S_T as Control\n"
             f"Corr(payoff, S_T)={corr_cv:.4f}  "
             f"c*={c_star:.4f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m33_02_antithetic_cv.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Importance sampling and QMC vs MC
# ============================================================
t0 = time.perf_counter()
print("[M33] Figure 3: Importance sampling and QMC uniformity ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M33 — Importance Sampling | QMC vs MC\n"
             "OTM option pricing and low-discrepancy sequences",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) IS for OTM call (K=130, sigma=0.20, T=1)
K_otm   = 130.0
BS_OTM  = bs_call(S0, K_otm, r, T, sigma, q)
MU_OTM  = (r - q - 0.5*sigma**2)*T
SIG_OTM = sigma*np.sqrt(T)
mu_IS   = (np.log(K_otm/S0) - MU_OTM) / SIG_OTM

N_IS = 10_000
rng_is = np.random.default_rng(0)
Z_is   = rng_is.standard_normal(N_IS)

# Standard MC for OTM
S_std  = S0*np.exp(MU_OTM + SIG_OTM*Z_is)
pay_std = np.maximum(S_std - K_otm, 0)
price_std_otm = np.exp(-r*T)*pay_std.mean()

# IS for OTM
Z_is2  = rng_is.standard_normal(N_IS)
Z_shifted = Z_is2 + mu_IS
S_IS   = S0*np.exp(MU_OTM + SIG_OTM*Z_shifted)
pay_IS_ = np.maximum(S_IS - K_otm, 0)
LR     = np.exp(-mu_IS*Z_is2 - 0.5*mu_IS**2)
price_IS_otm = np.exp(-r*T)*(pay_IS_*LR).mean()

ax = axes[0]
# Show the two distributions
x_z = np.linspace(-4, 6, 300)
ax.plot(x_z, norm.pdf(x_z), color=BLUE, lw=2.5, label="Standard N(0,1)")
ax.plot(x_z, norm.pdf(x_z, mu_IS, 1), color=GREEN, lw=2.5,
        label=f"IS N({mu_IS:.2f},1)")
ax.axvline((np.log(K_otm/S0)-MU_OTM)/SIG_OTM, color=RED, lw=2,
           linestyle="--", label=f"Exercise boundary (K={K_otm})")
ax.fill_between(x_z[x_z>=(np.log(K_otm/S0)-MU_OTM)/SIG_OTM],
                norm.pdf(x_z[x_z>=(np.log(K_otm/S0)-MU_OTM)/SIG_OTM]),
                color=RED, alpha=0.20, label="Exercise region")
ax.set_xlabel("Z"); ax.set_ylabel("Density")
ax.set_title(f"Importance Sampling: OTM Call (K={K_otm})\n"
             f"Shift mean to concentrate paths near K\n"
             f"BS={BS_OTM:.6f}  MC={price_std_otm:.4f}  IS={price_IS_otm:.6f}",
             color=WHITE, fontsize=8)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) QMC vs MC: 2D uniformity (Van der Corput in 2D)
ax = axes[1]
N_pts = 256
rng_u = np.random.default_rng(0)
MC_pts = rng_u.uniform(0, 1, (N_pts, 2))

def vdc(N, base=2):
    seq = np.zeros(N)
    for i in range(N):
        n, bit, f = i+1, 0, 0.5
        while n:
            if n & 1: seq[i] += f
            n >>= 1; f *= 0.5
    return seq

def vdc_base3(N):
    seq = np.zeros(N)
    for i in range(N):
        n, f = i+1, 1/3
        while n:
            seq[i] += (n % 3) * f
            n //= 3; f /= 3
    return seq

QMC_pts = np.column_stack([vdc(N_pts, 2), vdc_base3(N_pts)])
ax.scatter(MC_pts[:,0], MC_pts[:,1], s=5, alpha=0.5, color=BLUE,
           label=f"Pseudo-random ({N_pts} pts)")
ax.scatter(QMC_pts[:,0], QMC_pts[:,1], s=5, alpha=0.5, color=GREEN,
           label=f"QMC VdC ({N_pts} pts)")
ax.set_xlabel("U1"); ax.set_ylabel("U2")
ax.set_title("QMC vs Pseudo-Random: 2D Uniformity\n"
             "QMC fills space more uniformly (lower discrepancy)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Running estimates: all methods for OTM call
ax = axes[2]
N_run = 5000
estimates = {nm: [] for nm in ["Standard", "Antithetic", "IS", "Stratified"]}
for n_ in range(1, N_run+1, 50):
    estimates["Standard"].append(mc_standard(n_, seed=1)[0])
    estimates["Antithetic"].append(mc_antithetic(n_, seed=1)[0])
    estimates["IS"].append(mc_importance_sampling(n_, seed=1)[0])
    estimates["Stratified"].append(mc_stratified(n_, seed=1)[0])

x_run = range(1, N_run+1, 50)
for nm, col in zip(["Standard","Antithetic","IS","Stratified"],
                    [BLUE, GREEN, RED, PURPLE]):
    ax.plot(list(x_run), estimates[nm], color=col, lw=1.8, alpha=0.85,
            label=nm)
ax.axhline(BS_PRICE, color=WHITE, lw=2, linestyle="--",
           label=f"BS = {BS_PRICE:.4f}")
ax.set_xlabel("N paths"); ax.set_ylabel("Price estimate")
ax.set_title("Running MC Estimate — All Methods\nATM European Call",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m33_03_is_qmc.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
se_std_10k = mc_standard(10000)[1]
se_av_10k  = mc_antithetic(10000)[1]
se_cv_10k  = mc_control_variate(10000)[1]
se_is_10k  = mc_importance_sampling(10000)[1]

print()
print("=" * 65)
print("  MODULE 33 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] MC: C = e^{-rT}*E[max(S_T-K,0)]  error O(1/sqrt(N))")
print("  [2] Antithetic: use Z and -Z  Var reduced by (1+rho)/2")
print("  [3] Control variate: c*=-Cov(f,X)/Var(X)  VR=1-rho^2")
print("  [4] IS: shift distribution toward exercise region")
print("  [5] QMC: Sobol/VdC sequences  O((logN)^d/N) convergence")
print("  [6] Stratified: divide [0,1]^d into strata, 1 sample each")
print(f"  BS reference:     {BS_PRICE:.6f}")
print(f"  Standard MC SE:   {se_std_10k:.6f}  (N=10,000)")
print(f"  Antithetic SE:    {se_av_10k:.6f}  VR={(se_std_10k/se_av_10k)**2:.1f}x")
print(f"  Control variate SE:{se_cv_10k:.6f}  VR={(se_std_10k/se_cv_10k)**2:.1f}x")
print(f"  Imp. sampling SE: {se_is_10k:.6f}  VR={(se_std_10k/se_is_10k)**2:.1f}x")
print("=" * 65)

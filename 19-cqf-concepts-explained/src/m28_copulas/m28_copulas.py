#!/usr/bin/env python3
"""
M28 — Copulas: Dependency Modeling Beyond Correlation
======================================================
Module 28 | CQF Concepts Explained
Group 5   | Risk Management

Theory
------
Linear correlation is an insufficient measure of dependence:
- It is only invariant under linear transformations.
- It cannot distinguish tail dependence from central dependence.
- The same correlation can produce dramatically different joint
  tail behavior depending on the copula used.

Sklar's Theorem (1959)
------------------------
Every multivariate joint CDF F with marginals F_1,...,F_n satisfies:
    F(x_1,...,x_n) = C(F_1(x_1), ..., F_n(x_n))

for some copula C: [0,1]^n -> [0,1], which is itself a joint CDF
on uniform [0,1] marginals. The copula C is unique if the marginals
are continuous. Conversely, for any marginals F_i and copula C,
F is a valid joint distribution.

Key implication: copulas separate the marginal distributions
from the dependence structure. We can fit marginals independently,
then choose a copula to model joint tail behavior.

Gaussian Copula
---------------
    C_Gauss(u, v; rho) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)

No tail dependence: lambda_U = lambda_L = 0.
Crisis implication: Gaussian copula underestimates joint extreme
losses (CDO mispricing pre-2008).

Student-t Copula
-----------------
    C_t(u, v; rho, nu) = T_{2,nu}(T_nu^{-1}(u), T_nu^{-1}(v); rho)

Symmetric tail dependence:
    lambda_U = lambda_L = 2*t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))

As nu -> inf: C_t -> C_Gauss (tail dependence vanishes).
As nu -> 1:   tail dependence -> max (Cauchy copula).

Archimedean Copulas
--------------------
Defined by a generator phi: C(u,v) = phi^{-1}(phi(u) + phi(v))

Clayton: phi(t) = t^{-theta} - 1   (theta > 0)
    C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
    Lower tail dependence: lambda_L = 2^{-1/theta}, lambda_U = 0

Gumbel: phi(t) = (-ln t)^theta     (theta >= 1)
    C(u,v) = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta})
    Upper tail dependence: lambda_U = 2 - 2^{1/theta}, lambda_L = 0

Frank: phi(t) = -ln((exp(-theta*t)-1)/(exp(-theta)-1))
    No tail dependence: lambda_U = lambda_L = 0 (symmetric like Gaussian)

Kendall's Tau and Copula Parameters
-------------------------------------
Clayton: tau = theta / (theta + 2)   => theta = 2*tau/(1-tau)
Gumbel:  tau = 1 - 1/theta           => theta = 1/(1-tau)
Frank:   tau = 1 - 4/theta*(1/theta * integral_0^theta t/(e^t-1) dt - 1)
         (numerical inversion required)

Tail Dependence Coefficient
-----------------------------
    lambda_U = lim_{u->1} P(V > u | U > u)  (upper tail)
    lambda_L = lim_{u->0} P(V < u | U < u)  (lower tail)

References
----------
- Sklar, A. (1959). Fonctions de répartition à n dimensions et
  leurs marges. Publications de l'Institut de Statistique de
  l'Université de Paris, 8, 229-231.
- Nelsen, R.B. (2006). An Introduction to Copulas. 2nd ed. Springer.
- Li, D.X. (2000). On default correlation: a copula function approach.
  Journal of Fixed Income, 9(4), 43-54. (CDO pricing, Gaussian copula)
- McNeil, A.J., Frey, R., Embrechts, P. (2015). Quantitative Risk
  Management. Princeton University Press. Chapter 7.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t, kendalltau
from scipy.special import ndtri

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a";  PANEL  = "#111111"; GRID   = "#1e1e1e"
WHITE  = "#e8e8e8";  BLUE   = "#4a9eff"; GREEN  = "#00d4aa"
ORANGE = "#ff8c42";  RED    = "#ff4757"; PURPLE = "#a855f7"
YELLOW = "#ffd700"

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
# SECTION 1 — COPULA SAMPLERS
# ============================================================

def sample_gaussian_copula(rho, n, seed=42):
    """Sample (U,V) from bivariate Gaussian copula C_Gauss(rho)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1, rho], [rho, 1]])
    L   = np.linalg.cholesky(cov)
    Z   = rng.standard_normal((n, 2)) @ L.T
    return norm.cdf(Z[:, 0]), norm.cdf(Z[:, 1])

def sample_t_copula(rho, nu, n, seed=42):
    """Sample (U,V) from bivariate Student-t copula C_t(rho, nu)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1, rho], [rho, 1]])
    L   = np.linalg.cholesky(cov)
    Z   = rng.standard_normal((n, 2)) @ L.T
    chi2_rv = rng.chisquare(nu, size=n)
    W   = Z / np.sqrt(chi2_rv[:, np.newaxis] / nu)
    return student_t.cdf(W[:, 0], nu), student_t.cdf(W[:, 1], nu)

def sample_clayton_copula(theta, n, seed=42):
    """Sample from Clayton copula using Marshall-Olkin algorithm."""
    rng = np.random.default_rng(seed)
    # V ~ Gamma(1/theta, 1); U1, U2 ~ Uniform
    V  = rng.gamma(1/theta, 1, size=n)
    E1 = rng.exponential(1, size=n)
    E2 = rng.exponential(1, size=n)
    U  = (1 + E1/V)**(-1/theta)
    V2 = (1 + E2/V)**(-1/theta)
    return np.clip(U, 1e-8, 1-1e-8), np.clip(V2, 1e-8, 1-1e-8)

def sample_gumbel_copula(theta, n, seed=42):
    """Sample from Gumbel copula using stable distribution method."""
    rng = np.random.default_rng(seed)
    # Stable(1/theta, 1) via Chambers-Mallows-Stuck
    alpha_s = 1/theta
    U0 = rng.uniform(0, np.pi, size=n)
    E0 = rng.exponential(1, size=n)
    if alpha_s == 1:
        S = np.sin(U0) / np.cos(U0) * (np.pi/2 - U0)
        S = (S + (np.pi/2 - U0) * np.tan(U0)) * 2/np.pi
    else:
        S = (np.sin(alpha_s*(U0 - np.pi/2 + np.pi/(2*alpha_s))) /
             (np.cos(U0)**( 1/alpha_s)) *
             (np.cos(U0 - alpha_s*(U0 - np.pi/2 + np.pi/(2*alpha_s))) /
              E0)**((1-alpha_s)/alpha_s))
    S = np.maximum(S, 1e-8)
    E1 = rng.exponential(1, size=n)
    E2 = rng.exponential(1, size=n)
    U  = np.exp(-(E1/S)**(1/theta))
    V  = np.exp(-(E2/S)**(1/theta))
    return np.clip(U, 1e-8, 1-1e-8), np.clip(V, 1e-8, 1-1e-8)

def sample_frank_copula(theta, n, seed=42):
    """Sample from Frank copula via conditional distribution inversion."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(1e-6, 1-1e-6, size=n)
    W = rng.uniform(1e-6, 1-1e-6, size=n)
    # Conditional: V = -1/theta * ln(1 + W*(exp(-theta)-1)/
    #                                   (W*(exp(-theta*U)-1) - exp(-theta*U)))
    et  = np.exp(-theta)
    etU = np.exp(-theta * U)
    num = W * (et - 1)
    den = W * (etU - 1) - etU
    # Avoid division by zero
    den = np.where(np.abs(den) < 1e-10, 1e-10, den)
    V   = -np.log(1 + num/den) / theta
    return np.clip(U, 1e-8, 1-1e-8), np.clip(V, 1e-8, 1-1e-8)

# ============================================================
# SECTION 2 — TAIL DEPENDENCE
# ============================================================

def tail_dep_t(rho, nu):
    """Upper=Lower tail dependence for Student-t copula."""
    return 2 * student_t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)

def tail_dep_clayton_lower(theta):
    return 2**(-1/theta)

def tail_dep_gumbel_upper(theta):
    return 2 - 2**(1/theta)

def empirical_tail_dep(U, V, threshold=0.05):
    """Empirical lower tail dependence: P(V<q | U<q)."""
    q = threshold
    mask = U < q
    return np.mean(V[mask] < q) if mask.sum() > 0 else np.nan

# ============================================================
# PARAMETERS AND DIAGNOSTICS
# ============================================================
RHO   = 0.70
NU    = 4
N_SMP = 3000

# Kendall's tau -> Clayton parameter
tau_target = 0.50
theta_clayton = 2*tau_target / (1-tau_target)
theta_gumbel  = 1 / (1-tau_target)
theta_frank   = 5.736  # numerical; tau~0.50 for Frank(5.736)

print(f"[M28] Copula parameters (target Kendall tau ~ {tau_target:.2f}):")
print(f"      Gaussian rho = {RHO:.2f}")
print(f"      Student-t: rho={RHO:.2f}, nu={NU}")
print(f"      Clayton theta = {theta_clayton:.4f}  (tau={tau_target:.2f})")
print(f"      Gumbel  theta = {theta_gumbel:.4f}   (tau={tau_target:.2f})")
print(f"      Frank   theta = {theta_frank:.4f}   (tau~{tau_target:.2f})")

print(f"\n[M28] Tail dependence coefficients:")
print(f"      Gaussian: lambda_U = lambda_L = 0 (no tail dep)")
print(f"      t-copula: lambda_U = lambda_L = {tail_dep_t(RHO, NU):.4f}")
print(f"      Clayton:  lambda_L = {tail_dep_clayton_lower(theta_clayton):.4f}  "
      f"lambda_U = 0")
print(f"      Gumbel:   lambda_U = {tail_dep_gumbel_upper(theta_gumbel):.4f}  "
      f"lambda_L = 0")
print(f"      Frank:    lambda_U = lambda_L = 0 (no tail dep)")

# Sample all copulas
U_gauss, V_gauss = sample_gaussian_copula(RHO, N_SMP)
U_t,     V_t     = sample_t_copula(RHO, NU, N_SMP, seed=43)
U_clay,  V_clay  = sample_clayton_copula(theta_clayton, N_SMP, seed=44)
U_gumb,  V_gumb  = sample_gumbel_copula(theta_gumbel,  N_SMP, seed=45)
U_frank, V_frank = sample_frank_copula(theta_frank,    N_SMP, seed=46)

# ============================================================
# FIGURE 1 — Copula scatter plots (uniform margins)
# ============================================================
t0 = time.perf_counter()
print("\n[M28] Figure 1: Copula scatter plots (uniform margins) ...")

copula_samples = [
    ("Gaussian\n(rho=0.70, no tail dep)",      U_gauss, V_gauss, BLUE),
    ("Student-t\n(rho=0.70, nu=4, sym. tail)", U_t,     V_t,     GREEN),
    ("Clayton\n(lower tail dep)",               U_clay,  V_clay,  ORANGE),
    ("Gumbel\n(upper tail dep)",                U_gumb,  V_gumb,  PURPLE),
    ("Frank\n(symmetric, no tail dep)",         U_frank, V_frank, YELLOW),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor(DARK)
fig.suptitle("M28 — Copulas: Dependence Structure on Uniform Margins\n"
             "Same Kendall tau ~ 0.50, different tail behavior",
             color=WHITE, fontsize=11, fontweight="bold")

for ax, (label, U_, V_, col) in zip(axes, copula_samples):
    ax.scatter(U_, V_, s=3, alpha=0.3, color=col)
    # Highlight lower-left and upper-right corners
    q = 0.10
    mask_ll = (U_ < q) & (V_ < q)
    mask_ur = (U_ > 1-q) & (V_ > 1-q)
    ax.scatter(U_[mask_ll], V_[mask_ll], s=15, color=RED, alpha=0.9,
               label=f"LL: {mask_ll.sum()}")
    ax.scatter(U_[mask_ur], V_[mask_ur], s=15, color=GREEN, alpha=0.9,
               label=f"UR: {mask_ur.sum()}")
    ax.set_title(label, color=WHITE, fontsize=8)
    ax.set_xlabel("U"); ax.set_ylabel("V")
    ax.legend(fontsize=6); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m28_01_copula_scatter.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Tail dependence: quantile exceedance curves
# ============================================================
t0 = time.perf_counter()
print("[M28] Figure 2: Tail dependence and Kendall's tau ...")

q_arr = np.linspace(0.01, 0.25, 60)

def emp_lower_tail(U_, V_, q_):
    mask = U_ < q_
    return np.mean(V_[mask] < q_) if mask.sum() > 5 else np.nan

def emp_upper_tail(U_, V_, q_):
    mask = U_ > (1-q_)
    return np.mean(V_[mask] > (1-q_)) if mask.sum() > 5 else np.nan

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M28 — Tail Dependence Analysis\n"
             "Empirical tail dependence functions",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Lower tail dependence
ax = axes[0]
for label, U_, V_, col in copula_samples:
    td_lower = [emp_lower_tail(U_, V_, q) for q in q_arr]
    ax.plot(q_arr, td_lower, color=col, lw=2,
            label=label.split("\n")[0])
ax.axhline(tail_dep_t(RHO, NU), color=GREEN,  lw=1.5, linestyle=":",
           alpha=0.6, label=f"t-copula lambda_L={tail_dep_t(RHO,NU):.3f}")
ax.axhline(tail_dep_clayton_lower(theta_clayton), color=ORANGE, lw=1.5,
           linestyle=":", alpha=0.6,
           label=f"Clayton lambda_L={tail_dep_clayton_lower(theta_clayton):.3f}")
ax.set_xlabel("Threshold q"); ax.set_ylabel("P(V < q | U < q)")
ax.set_title("Empirical Lower Tail Dependence\n"
             "Clayton and t-copula cluster in lower-left",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) Upper tail dependence
ax = axes[1]
for label, U_, V_, col in copula_samples:
    td_upper = [emp_upper_tail(U_, V_, q) for q in q_arr]
    ax.plot(q_arr, td_upper, color=col, lw=2,
            label=label.split("\n")[0])
ax.axhline(tail_dep_gumbel_upper(theta_gumbel), color=PURPLE, lw=1.5,
           linestyle=":", alpha=0.6,
           label=f"Gumbel lambda_U={tail_dep_gumbel_upper(theta_gumbel):.3f}")
ax.set_xlabel("Threshold q"); ax.set_ylabel("P(V > 1-q | U > 1-q)")
ax.set_title("Empirical Upper Tail Dependence\n"
             "Gumbel and t-copula cluster in upper-right",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (c) Kendall's tau vs linear correlation comparison
ax = axes[2]
# For Gaussian copula: tau = (2/pi)*arcsin(rho)
rho_arr = np.linspace(-1, 1, 200)
tau_gauss = (2/np.pi)*np.arcsin(rho_arr)
ax.plot(rho_arr, tau_gauss, color=BLUE,  lw=2.5,
        label="Gaussian: tau = (2/pi)*arcsin(rho)")
ax.plot(rho_arr, rho_arr,   color=WHITE, lw=1.5, linestyle=":",
        alpha=0.5, label="tau = rho (identity)")
# Clayton: rho vs tau
tau_arr   = np.linspace(-0.99, 0.99, 200)
theta_cl  = np.where(tau_arr > 0, 2*tau_arr/(1-tau_arr), np.nan)
# Spearman rho for Clayton (approx numerical)
ax.scatter([tau_target], [RHO], color=YELLOW, s=80, zorder=5,
           label=f"Base case: tau={tau_target}, rho_lin={RHO}")
ax.set_xlabel("Linear correlation rho")
ax.set_ylabel("Kendall's tau")
ax.set_title("Kendall's Tau vs Linear Correlation\n"
             "Gaussian copula: tau = (2/pi)*arcsin(rho)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m28_02_tail_dependence.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Portfolio loss distribution under different copulas
# ============================================================
t0 = time.perf_counter()
print("[M28] Figure 3: Portfolio loss distribution under different copulas ...")

# Two-asset portfolio: equal weights, same t(5) marginals
# Transform uniform marginals to t(5) losses
NU_MARG = 5; VOL = 0.01
N_PORT  = 10000

def copula_to_losses(U_, V_, nu_m=NU_MARG, vol=VOL):
    """Map uniform margins to t(nu_m) losses scaled by vol."""
    L1 = -student_t.ppf(U_, nu_m) * vol
    L2 = -student_t.ppf(V_, nu_m) * vol
    return 0.5*L1 + 0.5*L2   # equal-weight portfolio loss

U_g2, V_g2 = sample_gaussian_copula(RHO, N_PORT, seed=10)
U_t2, V_t2 = sample_t_copula(RHO, NU, N_PORT, seed=11)
U_c2, V_c2 = sample_clayton_copula(theta_clayton, N_PORT, seed=12)
U_gu2,V_gu2= sample_gumbel_copula(theta_gumbel, N_PORT, seed=13)

L_gauss  = copula_to_losses(U_g2,  V_g2)
L_t      = copula_to_losses(U_t2,  V_t2)
L_clay   = copula_to_losses(U_c2,  V_c2)
L_gumb   = copula_to_losses(U_gu2, V_gu2)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M28 — Portfolio Loss Distribution Under Different Copulas\n"
             "Same marginals, same tau, different tail behavior",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Loss distributions
ax = axes[0]
bins = np.linspace(-0.05, 0.05, 80)
ax.hist(L_gauss, bins=bins, density=True, alpha=0.45, color=BLUE,
        label="Gaussian")
ax.hist(L_t,     bins=bins, density=True, alpha=0.45, color=GREEN,
        label="Student-t (nu=4)")
ax.hist(L_clay,  bins=bins, density=True, alpha=0.45, color=ORANGE,
        label="Clayton")
ax.hist(L_gumb,  bins=bins, density=True, alpha=0.45, color=PURPLE,
        label="Gumbel")
ax.set_xlabel("Portfolio loss"); ax.set_ylabel("Density")
ax.set_title("Portfolio P&L Distribution\n"
             "Same marginals and tau, different copulas",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) VaR comparison across confidence levels
ax = axes[1]
alphas_c = np.linspace(0.90, 0.999, 100)
for L_, lbl_, col_ in [(L_gauss,"Gaussian",BLUE),(L_t,"t (nu=4)",GREEN),
                        (L_clay,"Clayton",ORANGE),(L_gumb,"Gumbel",PURPLE)]:
    var_c = [np.percentile(L_, a*100) for a in alphas_c]
    ax.plot(alphas_c*100, np.array(var_c)*100, color=col_, lw=2, label=lbl_)
ax.set_xlabel("Confidence level (%)"); ax.set_ylabel("Portfolio VaR (%)")
ax.set_title("VaR vs Confidence Level\nt-copula => highest VaR at extreme alpha",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Tail loss comparison: scatter plot of joint extremes
ax = axes[2]
q_ext = 0.05
for (U_, V_), lbl_, col_ in [
    ((U_g2,V_g2),"Gaussian",BLUE), ((U_t2,V_t2),"t-copula",GREEN),
    ((U_c2,V_c2),"Clayton",ORANGE),((U_gu2,V_gu2),"Gumbel",PURPLE)]:
    mask = (U_ < q_ext) & (V_ < q_ext)
    ax.scatter([lbl_], [mask.sum()], s=200, color=col_, zorder=5)
    ax.text(lbl_, mask.sum()+2, str(mask.sum()), ha="center",
            fontsize=9, color=col_)
expected = N_PORT * q_ext**2
ax.axhline(expected, color=WHITE, lw=1.5, linestyle="--",
           label=f"Independence: {expected:.0f}")
ax.set_ylabel("Joint lower-tail events  (q=5%)")
ax.set_title(f"Joint Lower-Tail Counts (out of {N_PORT})\n"
             f"P(U<5%, V<5%): independence baseline={expected:.0f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m28_03_portfolio_copula.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 28 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Sklar: F(x,y) = C(F1(x), F2(y))  copula separates deps")
print("  [2] Gaussian copula: no tail dependence (CDO crisis lesson)")
print("  [3] t-copula: symmetric tail dep  lambda = f(rho, nu)")
print("  [4] Clayton: lower tail dep  lambda_L = 2^{-1/theta}")
print("  [5] Gumbel:  upper tail dep  lambda_U = 2 - 2^{1/theta}")
print("  [6] Same tau, same rho => very different tail behavior")
print(f"  t-copula (rho={RHO}, nu={NU}): "
      f"lambda = {tail_dep_t(RHO, NU):.4f}")
print(f"  Clayton  (theta={theta_clayton:.3f}): "
      f"lambda_L = {tail_dep_clayton_lower(theta_clayton):.4f}")
print(f"  Gumbel   (theta={theta_gumbel:.3f}): "
      f"lambda_U = {tail_dep_gumbel_upper(theta_gumbel):.4f}")
print("=" * 65)

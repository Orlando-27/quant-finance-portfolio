"""
M55 -- Copulas & Dependence Modelling
======================================
CQF Concepts Explained | Project 19 | Quantitative Finance Portfolio

Theory
------
A copula separates the marginal distributions from the dependence structure,
enabling flexible multivariate modelling beyond the Gaussian assumption.

Sklar's Theorem (1959)
----------------------
For any joint CDF F(x_1,...,x_d) with marginals F_1,...,F_d, there exists
a copula C: [0,1]^d -> [0,1] such that:

    F(x_1,...,x_d) = C(F_1(x_1), ..., F_d(x_d))

If F_i are continuous, C is unique. Conversely, for any marginals F_i and
copula C, F is a valid joint distribution.

Key Copula Families
-------------------
1. Gaussian Copula
   C_Ga(u,v; rho) = Phi_rho(Phi^{-1}(u), Phi^{-1}(v))
   Tail dependence: lambda_L = lambda_U = 0 (asymptotically independent)

2. Student-t Copula
   C_t(u,v; rho, nu) = t_{rho,nu}(t_nu^{-1}(u), t_nu^{-1}(v))
   Tail dependence: lambda_L = lambda_U = 2*t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))

3. Clayton Copula (lower tail dependence)
   C_Cl(u,v; theta) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
   lambda_L = 2^{-1/theta},  lambda_U = 0

4. Gumbel Copula (upper tail dependence)
   C_Gu(u,v; theta) = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta})
   lambda_L = 0,  lambda_U = 2 - 2^{1/theta}

Kendall's tau & Copula Parameter
---------------------------------
Clayton: tau = theta / (theta + 2)   => theta = 2*tau/(1-tau)
Gumbel:  tau = 1 - 1/theta           => theta = 1/(1-tau)
Gaussian: tau = (2/pi)*arcsin(rho)   => rho = sin(pi*tau/2)

Tail Dependence Coefficient
---------------------------
lambda_U = lim_{u->1} P(V > u | U > u) = lim_{u->1} (1-2u+C(u,u))/(1-u)
lambda_L = lim_{u->0} P(V < u | U < u) = lim_{u->0} C(u,u)/u

References
----------
Sklar (1959) "Fonctions de Repartition a n Dimensions et Leurs Marges"
Nelsen (2006) "An Introduction to Copulas", Springer
McNeil, Frey & Embrechts (2005) "Quantitative Risk Management", ch.5
Li (2000) "On Default Correlation: A Copula Function Approach", JFI 9(4)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gamma as gamma_fn

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
ACCENT = "#58a6ff"
GOLD   = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
TEAL   = "#39d353"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  TEXT,
})

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "m55_copulas")
os.makedirs(FIGS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print()
print("=" * 65)
print("  MODULE 55: COPULAS & DEPENDENCE MODELLING")
print("  Sklar | Gaussian | t | Clayton | Gumbel | Tail Dependence")
print("=" * 65)

N = 2000   # simulation size

# =============================================================================
# 1. COPULA SAMPLERS (from scratch)
# =============================================================================

def sample_gaussian_copula(rho: float, n: int) -> np.ndarray:
    """Sample (U,V) from Gaussian copula with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L   = np.linalg.cholesky(cov)
    Z   = (L @ np.random.randn(2, n)).T    # (n,2) correlated normals
    U   = stats.norm.cdf(Z[:, 0])
    V   = stats.norm.cdf(Z[:, 1])
    return np.column_stack([U, V])

def sample_t_copula(rho: float, nu: float, n: int) -> np.ndarray:
    """Sample (U,V) from Student-t copula with correlation rho, df nu."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L   = np.linalg.cholesky(cov)
    Z   = (L @ np.random.randn(2, n)).T
    W   = np.random.chisquare(nu, n)
    T   = Z / np.sqrt(W[:, np.newaxis] / nu)   # bivariate t
    U   = stats.t.cdf(T[:, 0], nu)
    V   = stats.t.cdf(T[:, 1], nu)
    return np.column_stack([U, V])

def sample_clayton_copula(theta: float, n: int) -> np.ndarray:
    """Sample (U,V) from Clayton copula -- conditional inversion method."""
    # U ~ Uniform[0,1]
    # V | U: C_{2|1}^{-1}(t|u) = u * (t^{-theta/(1+theta)} - 1 + u^{-theta})^{-1/theta}
    U = np.random.uniform(0, 1, n)
    T = np.random.uniform(0, 1, n)
    # Conditional CDF inverse for Clayton
    V = (T**(-theta / (1 + theta)) - 1 + U**(-theta))**(-1.0 / theta)
    V = np.clip(V, 1e-9, 1 - 1e-9)
    return np.column_stack([U, V])

def sample_gumbel_copula(theta: float, n: int) -> np.ndarray:
    """
    Sample from Gumbel copula using Marshall-Olkin algorithm.
    theta >= 1.  For theta=1: independence.
    """
    # Sample stable random variable S ~ Stable(1/theta, 1, ...)
    # via Chambers-Mallows-Stuck method
    alpha_s = 1.0 / theta
    phi     = (np.random.uniform(0, 1, n) - 0.5) * np.pi
    E       = np.random.exponential(1.0, n)
    num     = np.sin(alpha_s * (phi + np.pi / 2))
    denom   = np.cos(phi)**(1.0 / alpha_s)
    ratio   = (np.cos(phi - alpha_s * (phi + np.pi / 2)) / E)**(
                (1 - alpha_s) / alpha_s)
    S       = (num / denom) * ratio

    # Sample E_1, E_2 ~ Exp(1) independent of S
    E1 = np.random.exponential(1.0, n)
    E2 = np.random.exponential(1.0, n)
    U  = np.exp(-(E1 / S)**(1.0 / theta))
    V  = np.exp(-(E2 / S)**(1.0 / theta))
    U  = np.clip(U, 1e-9, 1 - 1e-9)
    V  = np.clip(V, 1e-9, 1 - 1e-9)
    return np.column_stack([U, V])

# =============================================================================
# 2. COPULA DENSITY FUNCTIONS (for MLE)
# =============================================================================

def log_density_gaussian(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """Log-density of Gaussian copula."""
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    r2 = rho**2
    return (-0.5 * np.log(1 - r2)
            - (r2 * (x**2 + y**2) - 2 * rho * x * y)
            / (2 * (1 - r2)))

def log_density_t(u: np.ndarray, v: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """Log-density of Student-t copula."""
    x  = stats.t.ppf(u, nu)
    y  = stats.t.ppf(v, nu)
    r2 = rho**2
    # bivariate t density / product of marginal t densities
    log_biv = (np.log(gamma_fn((nu + 2) / 2))
               - np.log(gamma_fn(nu / 2))
               - np.log(nu * np.pi)
               - 0.5 * np.log(1 - r2)
               - (nu + 2) / 2 * np.log(
                    1 + (x**2 + y**2 - 2 * rho * x * y)
                    / (nu * (1 - r2))))
    log_marg = (np.log(gamma_fn((nu + 1) / 2))
                - np.log(gamma_fn(nu / 2))
                - 0.5 * np.log(nu * np.pi)
                - (nu + 1) / 2 * np.log(1 + x**2 / nu)
                + np.log(gamma_fn((nu + 1) / 2))
                - np.log(gamma_fn(nu / 2))
                - 0.5 * np.log(nu * np.pi)
                - (nu + 1) / 2 * np.log(1 + y**2 / nu))
    return log_biv - log_marg

def log_density_clayton(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Log-density of Clayton copula."""
    if theta <= 0:
        return np.full(len(u), -np.inf)
    return (np.log(1 + theta)
            + (-1 - theta) * (np.log(u) + np.log(v))
            + (-1/theta - 2) * np.log(u**(-theta) + v**(-theta) - 1))

def log_density_gumbel(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Log-density of Gumbel copula."""
    if theta < 1:
        return np.full(len(u), -np.inf)
    a  = (-np.log(u))**theta
    b  = (-np.log(v))**theta
    ab = (a + b)**(1.0 / theta)
    log_C = -ab
    log_d = (log_C
             + (theta - 1) * (np.log(-np.log(u)) + np.log(-np.log(v)))
             - np.log(u) - np.log(v)
             + np.log(ab**2 + (theta - 1) * ab)
             - (2 * theta - 1) / theta * np.log(a + b))
    return log_d

# =============================================================================
# 3. GENERATE DATA FROM t-COPULA + STUDENT MARGINALS (simulate equity pair)
# =============================================================================
RHO_TRUE = 0.65
NU_TRUE  = 5.0
NU_MARG  = 4.0

uv_true = sample_t_copula(RHO_TRUE, NU_TRUE, N)
# Apply Student-t marginals (simulate heavy-tailed equity returns)
X = stats.t.ppf(uv_true[:, 0], NU_MARG) * 0.012
Y = stats.t.ppf(uv_true[:, 1], NU_MARG) * 0.015

# Empirical uniform scores (probability integral transform)
U = stats.rankdata(X) / (N + 1)
V = stats.rankdata(Y) / (N + 1)

print(f"  [01] Simulated equity pair: N={N}  rho_true={RHO_TRUE}  nu_cop={NU_TRUE}")
print(f"       Pearson r={np.corrcoef(X, Y)[0,1]:.4f}  "
      f"Kendall tau={stats.kendalltau(X, Y).statistic:.4f}  "
      f"Spearman rho={stats.spearmanr(X, Y).statistic:.4f}")

# =============================================================================
# 4. COPULA MLE CALIBRATION
# =============================================================================

def fit_gaussian(u, v):
    def neg_ll(rho):
        rho = float(np.clip(rho, -0.999, 0.999))
        ll  = log_density_gaussian(u, v, rho)
        return -np.sum(ll[np.isfinite(ll)])
    res = minimize_scalar(neg_ll, bounds=(-0.999, 0.999), method="bounded")
    return float(res.x), -res.fun

def fit_t(u, v, nu_grid=np.arange(2, 20, 1)):
    best_ll, best_rho, best_nu = -np.inf, 0.0, 4.0
    for nu in nu_grid:
        def neg_ll(rho):
            rho = float(np.clip(rho, -0.999, 0.999))
            ll  = log_density_t(u, v, rho, nu)
            return -np.sum(ll[np.isfinite(ll)])
        res = minimize_scalar(neg_ll, bounds=(-0.999, 0.999), method="bounded")
        ll  = -res.fun
        if ll > best_ll:
            best_ll, best_rho, best_nu = ll, float(res.x), float(nu)
    return best_rho, best_nu, best_ll

def fit_clayton(u, v):
    def neg_ll(theta):
        theta = float(np.clip(theta, 0.01, 50))
        ll    = log_density_clayton(u, v, theta)
        return -np.sum(ll[np.isfinite(ll)])
    res = minimize_scalar(neg_ll, bounds=(0.01, 50), method="bounded")
    return float(res.x), -res.fun

def fit_gumbel(u, v):
    def neg_ll(theta):
        theta = float(np.clip(theta, 1.001, 20))
        ll    = log_density_gumbel(u, v, theta)
        return -np.sum(ll[np.isfinite(ll)])
    res = minimize_scalar(neg_ll, bounds=(1.001, 20), method="bounded")
    return float(res.x), -res.fun

rho_ga, ll_ga          = fit_gaussian(U, V)
rho_t, nu_t, ll_t      = fit_t(U, V)
theta_cl, ll_cl        = fit_clayton(U, V)
theta_gu, ll_gu        = fit_gumbel(U, V)

# AIC = 2k - 2*ll (k=number of parameters)
aic = {"Gaussian": 2*1 - 2*ll_ga,
       "t-copula":  2*2 - 2*ll_t,
       "Clayton":  2*1 - 2*ll_cl,
       "Gumbel":   2*1 - 2*ll_gu}

print(f"  [02] MLE Calibration Results:")
print(f"       Gaussian: rho={rho_ga:.4f}  AIC={aic['Gaussian']:.1f}")
print(f"       t-copula: rho={rho_t:.4f}  nu={nu_t:.1f}  AIC={aic['t-copula']:.1f}")
print(f"       Clayton:  theta={theta_cl:.4f}  AIC={aic['Clayton']:.1f}")
print(f"       Gumbel:   theta={theta_gu:.4f}  AIC={aic['Gumbel']:.1f}")
best = min(aic, key=aic.get)
print(f"       Best fit (lowest AIC): {best}")

# =============================================================================
# 5. TAIL DEPENDENCE COEFFICIENTS
# =============================================================================

def tail_dep_t(rho, nu):
    """Analytical upper/lower tail dependence for t-copula."""
    lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)),
                           nu + 1)
    return float(lam)   # lambda_L = lambda_U for t-copula

def tail_dep_clayton(theta):
    return float(2**(-1.0 / theta)), 0.0   # (lambda_L, lambda_U)

def tail_dep_gumbel(theta):
    return 0.0, float(2 - 2**(1.0 / theta))   # (lambda_L, lambda_U)

def empirical_tail_dep(u, v, q=0.05):
    """Empirical tail dependence coefficients."""
    lam_L = np.mean((u < q) & (v < q)) / q
    lam_U = np.mean((u > 1-q) & (v > 1-q)) / q
    return float(lam_L), float(lam_U)

lam_t   = tail_dep_t(rho_t, nu_t)
lam_cl_L, lam_cl_U = tail_dep_clayton(theta_cl)
lam_gu_L, lam_gu_U = tail_dep_gumbel(theta_gu)
lam_emp_L, lam_emp_U = empirical_tail_dep(U, V)

print(f"  [03] Tail Dependence Coefficients:")
print(f"       {'Copula':<12} {'lambda_L':>10} {'lambda_U':>10}")
print(f"       {'Gaussian':<12} {0.0:>10.4f} {0.0:>10.4f}")
print(f"       {'t-copula':<12} {lam_t:>10.4f} {lam_t:>10.4f}")
print(f"       {'Clayton':<12} {lam_cl_L:>10.4f} {lam_cl_U:>10.4f}")
print(f"       {'Gumbel':<12} {lam_gu_L:>10.4f} {lam_gu_U:>10.4f}")
print(f"       {'Empirical':<12} {lam_emp_L:>10.4f} {lam_emp_U:>10.4f}")

# =============================================================================
# 6. PORTFOLIO LOSS SIMULATION UNDER DIFFERENT COPULAS
# =============================================================================
N_SIM = 10000
W     = np.array([0.5, 0.5])    # equal-weight portfolio

def portfolio_losses(uv: np.ndarray, nu_marg: float = NU_MARG,
                     sigma: tuple = (0.012, 0.015)) -> np.ndarray:
    """Convert copula samples to portfolio losses via t marginals."""
    r1 = stats.t.ppf(uv[:, 0], nu_marg) * sigma[0]
    r2 = stats.t.ppf(uv[:, 1], nu_marg) * sigma[1]
    return -(W[0] * r1 + W[1] * r2)    # losses positive

uv_ga = sample_gaussian_copula(rho_ga, N_SIM)
uv_tc = sample_t_copula(rho_t, nu_t, N_SIM)
uv_cl = sample_clayton_copula(theta_cl, N_SIM)
uv_gu = sample_gumbel_copula(theta_gu, N_SIM)

loss_ga = portfolio_losses(uv_ga)
loss_tc = portfolio_losses(uv_tc)
loss_cl = portfolio_losses(uv_cl)
loss_gu = portfolio_losses(uv_gu)

def risk(losses, p=0.99):
    var  = float(np.quantile(losses, p))
    cvar = float(losses[losses >= var].mean())
    return var, cvar

print(f"  [04] Portfolio Risk (N={N_SIM} simulations):")
print(f"       {'Copula':<12} {'99% VaR':>9} {'99% CVaR':>10} {'99.9% VaR':>11}")
for name, los in [("Gaussian", loss_ga), ("t-copula", loss_tc),
                  ("Clayton",  loss_cl), ("Gumbel",   loss_gu)]:
    v99, c99   = risk(los, 0.99)
    v999, _    = risk(los, 0.999)
    print(f"       {name:<12} {v99*100:>9.3f}% {c99*100:>9.3f}%  {v999*100:>10.3f}%")

# =============================================================================
# 7. FIGURE 1 -- Copula Scatter Plots (uniform scale)
# =============================================================================
n_plot = 1000   # subsample for scatter clarity
uv_samp = {
    "Gaussian\n(no tail dep)":       sample_gaussian_copula(rho_ga, n_plot),
    f"t-copula\n(sym tail dep)":     sample_t_copula(rho_t, nu_t, n_plot),
    "Clayton\n(lower tail dep)":     sample_clayton_copula(theta_cl, n_plot),
    "Gumbel\n(upper tail dep)":      sample_gumbel_copula(theta_gu, n_plot),
}
colors_cop = [ACCENT, RED, GREEN, GOLD]

fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=DARK)
fig.suptitle("M55 -- Copula Families: Uniform Scale & Return Scale",
             color=TEXT, fontsize=11)

for col_i, (name, uv) in enumerate(uv_samp.items()):
    # Row 0: uniform (u,v) space
    ax = axes[0, col_i]
    ax.scatter(uv[:, 0], uv[:, 1], s=3, color=colors_cop[col_i], alpha=0.4)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(PANEL)
    ax.grid(True)

    # Row 1: return scale via t(4) marginals
    ax = axes[1, col_i]
    r1 = stats.t.ppf(uv[:, 0], NU_MARG) * 0.012 * 100
    r2 = stats.t.ppf(uv[:, 1], NU_MARG) * 0.015 * 100
    ax.scatter(r1, r2, s=3, color=colors_cop[col_i], alpha=0.4)
    ax.set_title(f"Return Scale\n(t({NU_MARG:.0f}) marginals)", fontsize=8)
    ax.set_xlabel("Return X (%)")
    ax.set_ylabel("Return Y (%)")
    ax.set_facecolor(PANEL)
    ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m55_fig1_copula_scatter.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [05] Fig 1 saved: copula scatter (uniform & return scale)")

# =============================================================================
# 8. FIGURE 2 -- Copula Densities & Tail Dependence
# =============================================================================
fig = plt.figure(figsize=(15, 9), facecolor=DARK)
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M55 -- Copula Densities & Tail Dependence",
             color=TEXT, fontsize=11)

n_grid  = 50
u_g     = np.linspace(0.01, 0.99, n_grid)
uu, vv  = np.meshgrid(u_g, u_g)
u_flat  = uu.ravel()
v_flat  = vv.ravel()

dens_funcs = [
    ("Gaussian", lambda u, v: np.exp(log_density_gaussian(u, v, rho_ga)), ACCENT),
    ("t-copula", lambda u, v: np.exp(log_density_t(u, v, rho_t, nu_t)),   RED),
    ("Clayton",  lambda u, v: np.exp(log_density_clayton(u, v, theta_cl)), GREEN),
    ("Gumbel",   lambda u, v: np.exp(log_density_gumbel(u, v, theta_gu)),  GOLD),
]

for i, (name, dens_fn, col) in enumerate(dens_funcs):
    ax = fig.add_subplot(gs[0, i])
    d  = dens_fn(u_flat, v_flat).reshape(n_grid, n_grid)
    d  = np.clip(d, 0, np.percentile(d[np.isfinite(d)], 99))
    im = ax.contourf(uu, vv, d, levels=20, cmap="YlOrRd")
    ax.set_title(f"{name}\nDensity Contours")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_facecolor(PANEL)

# Lower row: tail conditional probability curves
ax = fig.add_subplot(gs[1, :2])
q_range = np.linspace(0.01, 0.25, 50)

# lambda_L(q): P(V < q | U < q) estimated empirically for each copula
for (name, uv_s, col) in [("Gaussian", uv_ga[:2000], ACCENT),
                            ("t-copula", uv_tc[:2000], RED),
                            ("Clayton",  uv_cl[:2000], GREEN),
                            ("Gumbel",   uv_gu[:2000], GOLD)]:
    lam_L_q = np.array([
        np.mean((uv_s[:, 0] < q) & (uv_s[:, 1] < q)) / (q + 1e-9)
        for q in q_range
    ])
    ax.plot(q_range, lam_L_q, color=col, lw=1.5, label=name)

ax.axhline(lam_cl_L, color=GREEN,  lw=0.8, ls="--",
           label=f"Clayton lambda_L={lam_cl_L:.3f}")
ax.set_title("Lower Tail Dependence lambda_L(q) = P(V<q | U<q)")
ax.set_xlabel("Quantile q")
ax.set_ylabel("Conditional Probability")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

ax = fig.add_subplot(gs[1, 2:])
q_upper = np.linspace(0.75, 0.99, 50)
for (name, uv_s, col) in [("Gaussian", uv_ga[:2000], ACCENT),
                            ("t-copula", uv_tc[:2000], RED),
                            ("Clayton",  uv_cl[:2000], GREEN),
                            ("Gumbel",   uv_gu[:2000], GOLD)]:
    lam_U_q = np.array([
        np.mean((uv_s[:, 0] > q) & (uv_s[:, 1] > q)) / (1 - q + 1e-9)
        for q in q_upper
    ])
    ax.plot(q_upper, lam_U_q, color=col, lw=1.5, label=name)

ax.axhline(lam_gu_U, color=GOLD, lw=0.8, ls="--",
           label=f"Gumbel lambda_U={lam_gu_U:.3f}")
ax.set_title("Upper Tail Dependence lambda_U(q) = P(V>q | U>q)")
ax.set_xlabel("Quantile q")
ax.set_ylabel("Conditional Probability")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.savefig(os.path.join(FIGS, "m55_fig2_densities_tail_dep.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [06] Fig 2 saved: copula densities & tail dependence curves")

# =============================================================================
# 9. FIGURE 3 -- Portfolio Loss Distribution & AIC Model Selection
# =============================================================================
fig = plt.figure(figsize=(15, 9), facecolor=DARK)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M55 -- Portfolio Loss: Copula Impact & Model Selection",
             color=TEXT, fontsize=11)

# 9A: Portfolio loss distribution comparison
ax = fig.add_subplot(gs[0, :])
bins = np.linspace(
    min(loss_ga.min(), loss_tc.min(), loss_cl.min(), loss_gu.min()),
    max(np.percentile(loss_ga, 99.5), np.percentile(loss_tc, 99.5),
        np.percentile(loss_cl, 99.5), np.percentile(loss_gu, 99.5)),
    80
)
for (name, los, col) in [("Gaussian", loss_ga, ACCENT),
                           ("t-copula", loss_tc, RED),
                           ("Clayton",  loss_cl, GREEN),
                           ("Gumbel",   loss_gu, GOLD)]:
    ax.hist(los, bins=bins, density=True, alpha=0.35,
            color=col, label=name, edgecolor=DARK, linewidth=0.2)
    v99, _ = risk(los, 0.99)
    ax.axvline(v99, color=col, lw=1.5, ls="--", alpha=0.9)

ax.set_title("Portfolio Loss Distribution by Copula\n"
             "(dashed lines: 99% VaR per copula)")
ax.set_xlabel("Portfolio Loss")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9B: AIC bar chart
ax = fig.add_subplot(gs[1, 0])
names_aic = list(aic.keys())
vals_aic  = list(aic.values())
cols_aic  = [GREEN if n == best else ACCENT for n in names_aic]
bars = ax.bar(names_aic, vals_aic, color=cols_aic, edgecolor=DARK, linewidth=0.5)
for bar, v in zip(bars, vals_aic):
    ax.text(bar.get_x() + bar.get_width()/2, v + 5,
            f"{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_title(f"AIC Model Selection\n(Best: {best})")
ax.set_ylabel("AIC (lower = better fit)")
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

# 9C: Concordance (Kendall's tau) implied by fitted parameters
def tau_from_theta_clayton(theta): return theta / (theta + 2)
def tau_from_theta_gumbel(theta):  return 1 - 1 / theta
def tau_from_rho_gaussian(rho):    return 2 / np.pi * np.arcsin(rho)

tau_ga = tau_from_rho_gaussian(rho_ga)
tau_tc = tau_from_rho_gaussian(rho_t)   # approx (same formula)
tau_cl = tau_from_theta_clayton(theta_cl)
tau_gu = tau_from_theta_gumbel(theta_gu)
tau_emp = stats.kendalltau(X, Y).statistic

ax = fig.add_subplot(gs[1, 1])
names_tau = ["Gaussian", "t-copula", "Clayton", "Gumbel", "Empirical"]
taus      = [tau_ga, tau_tc, tau_cl, tau_gu, tau_emp]
cols_tau  = [ACCENT, RED, GREEN, GOLD, PURPLE]
bars = ax.bar(names_tau, taus, color=cols_tau, edgecolor=DARK, linewidth=0.5)
ax.axhline(tau_emp, color=PURPLE, lw=1.2, ls="--",
           label=f"Empirical tau={tau_emp:.4f}")
for bar, v in zip(bars, taus):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_title("Kendall's Tau Implied by\nFitted Copula Parameters")
ax.set_ylabel("Kendall's tau")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

fig.savefig(os.path.join(FIGS, "m55_fig3_portfolio_model_selection.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [07] Fig 3 saved: portfolio loss distributions & AIC selection")

# =============================================================================
# SUMMARY
# =============================================================================
v99_ga, c99_ga = risk(loss_ga, 0.99)
v99_tc, c99_tc = risk(loss_tc, 0.99)
v99_cl, c99_cl = risk(loss_cl, 0.99)
v99_gu, c99_gu = risk(loss_gu, 0.99)

print()
print("  MODULE 55 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Sklar: F(x,y)=C(F_X(x),F_Y(y)) -- separates margins & dependence")
print("  [2] Gaussian copula: lambda_L=lambda_U=0 -- underestimates joint extremes")
print(f"  [3] t-copula: lambda_L=lambda_U={lam_t:.4f} -- symmetric tail co-movement")
print(f"  [4] Clayton: lambda_L={lam_cl_L:.4f} lambda_U=0 -- lower tail dependence")
print(f"  [5] Gumbel:  lambda_L=0 lambda_U={lam_gu_U:.4f} -- upper tail dependence")
print(f"  [6] Best fit: {best} (AIC={aic[best]:.1f})")
print(f"  [7] 99% VaR spread: Gaussian={v99_ga*100:.3f}%  "
      f"t={v99_tc*100:.3f}%  "
      f"Clayton={v99_cl*100:.3f}%  "
      f"Gumbel={v99_gu*100:.3f}%")
print(f"  [8] Copula choice impacts 99% CVaR by up to "
      f"{abs(c99_tc-c99_ga)*100:.3f}% (t vs Gaussian)")
print("  NEXT: M56 -- Algorithmic Trading: Momentum & Mean Reversion Signals")
print()

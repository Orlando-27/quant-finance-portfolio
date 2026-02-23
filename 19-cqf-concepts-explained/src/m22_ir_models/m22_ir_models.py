#!/usr/bin/env python3
"""
M22 — Interest Rate Models: Vasicek & CIR
==========================================
Module 22 | CQF Concepts Explained
Group 4   | Fixed Income & Interest Rates

Theory
------
Short-rate models describe the instantaneous risk-free rate r_t and
derive the full yield curve from a single state variable.

Vasicek (1977)
--------------
    dr_t = kappa*(theta - r_t)*dt + sigma*dW_t

Ornstein-Uhlenbeck process. Analytically tractable.
Conditional distribution: r_T | r_0 ~ N(mu_T, var_T)
    mu_T  = r_0*exp(-kappa*T) + theta*(1 - exp(-kappa*T))
    var_T = sigma^2/(2*kappa) * (1 - exp(-2*kappa*T))

Zero-coupon bond price P(t,T) = A(t,T)*exp(-B(t,T)*r_t)
    B(tau) = (1 - exp(-kappa*tau)) / kappa          tau = T - t
    ln A(tau) = (theta - sigma^2/(2*kappa^2))*(B(tau)-tau)
                - sigma^2*B(tau)^2 / (4*kappa)

Yield: R(tau) = -ln P / tau = [B(tau)*r_t - ln A(tau)] / tau

CIR (Cox-Ingersoll-Ross, 1985)
-------------------------------
    dr_t = kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*dW_t

Square-root diffusion. Conditional distribution: scaled non-central chi-squared.
Feller condition for positivity: 2*kappa*theta > sigma^2

Bond price: same A-B form, different coefficients:
    h  = sqrt(kappa^2 + 2*sigma^2)
    B(tau) = 2*(exp(h*tau)-1) / ((kappa+h)*(exp(h*tau)-1) + 2*h)
    A(tau) = [2*h*exp((kappa+h)*tau/2) /
              ((kappa+h)*(exp(h*tau)-1)+2*h)]^(2*kappa*theta/sigma^2)

Calibration
-----------
Fit {kappa, theta, sigma} to observed zero-coupon yields by minimising
sum of squared pricing errors:
    min_{params} sum_i [R_model(tau_i; params, r0) - R_obs(tau_i)]^2

References
----------
- Vasicek, O. (1977). An equilibrium characterization of the term structure.
  Journal of Financial Economics, 5(2), 177-188.
- Cox, J., Ingersoll, J., Ross, S. (1985). A theory of the term structure of
  interest rates. Econometrica, 53(2), 385-408.
- Brigo, D., Mercurio, F. (2006). Interest Rate Models — Theory and Practice.
  Springer Finance.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, ncx2

# ── Styling constants ────────────────────────────────────────────────────────
DARK   = "#0a0a0a"
PANEL  = "#111111"
GRID   = "#1e1e1e"
WHITE  = "#e8e8e8"
BLUE   = "#4a9eff"
GREEN  = "#00d4aa"
ORANGE = "#ff8c42"
RED    = "#ff4757"
PURPLE = "#a855f7"
YELLOW = "#ffd700"

WATERMARK = "Jose O. Bobadilla | CQF"
OUT_DIR = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   WHITE,
    "axes.titlecolor":   WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "text.color":        WHITE,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def watermark(ax):
    ax.text(0.99, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=WHITE, alpha=0.35, ha="right", va="bottom",
            fontstyle="italic")

# ============================================================
# SECTION 1 — MODEL DEFINITIONS
# ============================================================

class VasicekModel:
    """
    Vasicek (1977) short-rate model.
    Parameters: kappa (mean-reversion speed), theta (long-run mean),
                sigma (volatility), r0 (initial rate).
    """
    def __init__(self, kappa, theta, sigma, r0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0    = r0

    def bond_price(self, tau):
        """Closed-form zero-coupon bond price P(0,tau)."""
        k, th, s, r = self.kappa, self.theta, self.sigma, self.r0
        B = (1 - np.exp(-k * tau)) / k
        lnA = (th - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)
        return np.exp(lnA - B * r)

    def yield_curve(self, tau):
        """Spot yield R(tau) = -ln P(0,tau) / tau."""
        return -np.log(self.bond_price(tau)) / tau

    def simulate(self, T, n_steps, n_paths, seed=42):
        """
        Euler-Maruyama simulation of r_t paths.
        Returns (times, paths) arrays of shape (n_steps+1,) and (n_paths, n_steps+1).
        """
        rng   = np.random.default_rng(seed)
        dt    = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        k, th, s = self.kappa, self.theta, self.sigma
        sqrt_dt = np.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i+1] = r + k*(th - r)*dt + s*sqrt_dt*Z[:, i]
        return times, paths

    def terminal_distribution(self, T, r_grid):
        """Analytical N(mu_T, var_T) terminal density."""
        k, th, s = self.kappa, self.theta, self.sigma
        mu_T  = self.r0 * np.exp(-k*T) + th * (1 - np.exp(-k*T))
        var_T = s**2 / (2*k) * (1 - np.exp(-2*k*T))
        return norm.pdf(r_grid, mu_T, np.sqrt(var_T)), mu_T, var_T


class CIRModel:
    """
    Cox-Ingersoll-Ross (1985) short-rate model.
    Feller condition: 2*kappa*theta > sigma^2 (ensures r_t > 0).
    """
    def __init__(self, kappa, theta, sigma, r0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0    = r0
        feller = 2 * kappa * theta / sigma**2
        self.feller_ratio = feller  # must be > 1 for strict positivity

    def _h(self):
        return np.sqrt(self.kappa**2 + 2 * self.sigma**2)

    def bond_price(self, tau):
        """Closed-form zero-coupon bond price P(0,tau)."""
        k, th, s, r = self.kappa, self.theta, self.sigma, self.r0
        h = self._h()
        exp_ht = np.exp(h * tau)
        denom  = (k + h) * (exp_ht - 1) + 2 * h
        B = 2 * (exp_ht - 1) / denom
        lnA = (2*k*th / s**2) * np.log(2*h*np.exp((k+h)*tau/2) / denom)
        return np.exp(lnA - B * r)

    def yield_curve(self, tau):
        return -np.log(self.bond_price(tau)) / tau

    def simulate(self, T, n_steps, n_paths, seed=42):
        """
        Full truncation Euler scheme for CIR (ensures r >= 0 at each step).
        Higham & Mao (2005) full-truncation variant.
        """
        rng   = np.random.default_rng(seed)
        dt    = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        k, th, s = self.kappa, self.theta, self.sigma
        sqrt_dt = np.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            r_pos = np.maximum(r, 0.0)   # full truncation
            paths[:, i+1] = r + k*(th - r_pos)*dt + s*np.sqrt(r_pos)*sqrt_dt*Z[:, i]
            paths[:, i+1] = np.maximum(paths[:, i+1], 0.0)
        return times, paths

    def terminal_distribution(self, T, r_grid):
        """
        Analytical non-central chi-squared terminal density.
        Scaled: c*r_T ~ chi^2(nu, lambda) where
            c      = 4*kappa / (sigma^2*(1-exp(-kappa*T)))
            nu     = 4*kappa*theta / sigma^2  (degrees of freedom)
            lambda = c*r0*exp(-kappa*T)       (non-centrality)
        """
        k, th, s = self.kappa, self.theta, self.sigma
        c  = 4 * k / (s**2 * (1 - np.exp(-k*T)))
        nu = 4 * k * th / s**2
        lam = c * self.r0 * np.exp(-k*T)
        # pdf of r: f(r) = c * chi2.pdf(c*r, nu, lam)
        pdf = c * ncx2.pdf(c * r_grid, nu, lam)
        mu_T = self.r0 * np.exp(-k*T) + th * (1 - np.exp(-k*T))
        var_T = (self.r0 * s**2 * np.exp(-k*T) / k * (1 - np.exp(-k*T))
                 + th * s**2 / (2*k) * (1 - np.exp(-k*T))**2)
        return pdf, mu_T, var_T


# ============================================================
# SECTION 2 — CALIBRATION TO OBSERVED YIELD CURVE
# ============================================================

def calibrate_vasicek(tau_obs, y_obs, r0):
    """
    Calibrate Vasicek {kappa, theta, sigma} by minimising SSE
    between model yields and observed yields.
    """
    def objective(params):
        k, th, s = params
        if k <= 0 or s <= 0:
            return 1e10
        mdl = VasicekModel(k, th, s, r0)
        y_mdl = mdl.yield_curve(tau_obs)
        return np.sum((y_mdl - y_obs)**2)

    best_res, best_val = None, np.inf
    for k0 in [0.1, 0.3, 0.5]:
        for th0 in [0.03, 0.05, 0.07]:
            res = minimize(objective, [k0, th0, 0.01],
                           bounds=[(1e-4,5), (0.001,0.2), (1e-4,0.5)],
                           method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
    return best_res.x


def calibrate_cir(tau_obs, y_obs, r0):
    """
    Calibrate CIR {kappa, theta, sigma} subject to Feller condition.
    """
    def objective(params):
        k, th, s = params
        if k <= 0 or th <= 0 or s <= 0:
            return 1e10
        mdl = CIRModel(k, th, s, r0)
        y_mdl = mdl.yield_curve(tau_obs)
        # soft Feller penalty
        feller_pen = max(0, 1 - 2*k*th/s**2) * 1e4
        return np.sum((y_mdl - y_obs)**2) + feller_pen

    best_res, best_val = None, np.inf
    for k0 in [0.1, 0.3, 0.5]:
        for th0 in [0.03, 0.05, 0.07]:
            res = minimize(objective, [k0, th0, 0.05],
                           bounds=[(1e-4,5), (0.001,0.2), (1e-4,0.5)],
                           method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
    return best_res.x


# ============================================================
# SECTION 3 — FIGURES
# ============================================================

# Model parameters (representative US market circa 2024)
kappa_v = 0.30;  theta_v = 0.045; sigma_v = 0.015; r0 = 0.052
kappa_c = 0.35;  theta_c = 0.048; sigma_c = 0.080

vasicek = VasicekModel(kappa_v, theta_v, sigma_v, r0)
cir     = CIRModel    (kappa_c, theta_c, sigma_c, r0)

print("[M22] Vasicek — Feller ratio: N/A (Gaussian, may go negative)")
print(f"[M22] CIR     — Feller ratio: 2kappa*theta/sigma^2 = "
      f"{2*kappa_c*theta_c/sigma_c**2:.4f} (>1 required)")

tau_grid = np.linspace(0.25, 30, 200)

# ── SYNTHETIC OBSERVED YIELD CURVE (US Treasury-like, upward sloping) ───────
tau_obs = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
y_obs   = np.array([0.052, 0.051, 0.050, 0.048, 0.047, 0.046,
                    0.047, 0.048, 0.050, 0.051])

# Calibrate
print("[M22] Calibrating Vasicek to observed curve ...")
kv, thv, sv = calibrate_vasicek(tau_obs, y_obs, r0)
vasicek_cal = VasicekModel(kv, thv, sv, r0)
print(f"      kappa={kv:.4f}, theta={thv:.4f}, sigma={sv:.4f}")

print("[M22] Calibrating CIR to observed curve ...")
kc, thc, sc = calibrate_cir(tau_obs, y_obs, r0)
cir_cal = CIRModel(kc, thc, sc, r0)
print(f"      kappa={kc:.4f}, theta={thc:.4f}, sigma={sc:.4f}")

# ==============================================================
# FIGURE 1 — Yield curves + calibration + term premium
# ==============================================================
t0 = time.perf_counter()
print("[M22] Figure 1: Yield curves and calibration ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M22 — Interest Rate Models: Vasicek & CIR\nYield Curves",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Prior (uncalibrated) model yields
ax = axes[0]
ax.plot(tau_grid, vasicek.yield_curve(tau_grid)*100,
        color=BLUE,   lw=2.5, label=f"Vasicek (kappa={kappa_v}, theta={theta_v})")
ax.plot(tau_grid, cir.yield_curve(tau_grid)*100,
        color=GREEN,  lw=2.5, label=f"CIR (kappa={kappa_c}, theta={kappa_c})")
ax.scatter(tau_obs, y_obs*100, color=YELLOW, s=60, zorder=5, label="Observed")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
ax.set_title("Prior Parameters vs Observed", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Calibrated models
ax = axes[1]
ax.plot(tau_grid, vasicek_cal.yield_curve(tau_grid)*100,
        color=BLUE,   lw=2.5, label="Vasicek (calibrated)")
ax.plot(tau_grid, cir_cal.yield_curve(tau_grid)*100,
        color=GREEN,  lw=2.5, label="CIR (calibrated)")
ax.scatter(tau_obs, y_obs*100, color=YELLOW, s=60, zorder=5, label="Observed")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
ax.set_title("Calibrated Models vs Observed", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Pricing errors (bps)
ax = axes[2]
err_v = (vasicek_cal.yield_curve(tau_obs) - y_obs) * 1e4
err_c = (cir_cal.yield_curve(tau_obs)     - y_obs) * 1e4
ax.bar(tau_obs - 0.3, err_v, width=0.5, color=BLUE,  alpha=0.8, label="Vasicek error")
ax.bar(tau_obs + 0.3, err_c, width=0.5, color=GREEN, alpha=0.8, label="CIR error")
ax.axhline(0, color=WHITE, lw=1, linestyle=":")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Pricing error (bps)")
ax.set_title("Calibration Residuals (basis points)", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m22_01_yield_curves.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ==============================================================
# FIGURE 2 — Monte Carlo paths + terminal distributions
# ==============================================================
t0 = time.perf_counter()
print("[M22] Figure 2: Simulated paths and terminal distributions ...")

N_PATHS = 500; N_STEPS = 252; T_SIM = 5.0

times_v, paths_v = vasicek.simulate(T_SIM, N_STEPS, N_PATHS)
times_c, paths_c = cir.simulate(T_SIM, N_STEPS, N_PATHS)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M22 — Monte Carlo Simulation: Vasicek vs CIR\n"
             "500 paths, T=5Y, dt=1/252", color=WHITE, fontsize=12, fontweight="bold")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Vasicek paths
ax = fig.add_subplot(gs[0, :2])
for j in range(min(100, N_PATHS)):
    ax.plot(times_v, paths_v[j]*100, lw=0.4, alpha=0.3, color=BLUE)
ax.plot(times_v, paths_v.mean(axis=0)*100, color=WHITE, lw=2,
        label=f"Mean path (theta={theta_v*100:.1f}%)")
ax.axhline(theta_v*100, color=ORANGE, lw=1.5, linestyle="--",
           label=f"Long-run mean theta={theta_v*100:.1f}%")
ax.fill_between(times_v,
                np.percentile(paths_v, 5, axis=0)*100,
                np.percentile(paths_v, 95, axis=0)*100,
                color=BLUE, alpha=0.12, label="5th-95th percentile")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Short rate (%)")
ax.set_title("Vasicek — Gaussian Mean-Reversion\n"
             "dr = kappa*(theta-r)dt + sigma*dW", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# CIR paths
ax = fig.add_subplot(gs[1, :2])
for j in range(min(100, N_PATHS)):
    ax.plot(times_c, paths_c[j]*100, lw=0.4, alpha=0.3, color=GREEN)
ax.plot(times_c, paths_c.mean(axis=0)*100, color=WHITE, lw=2,
        label=f"Mean path (theta={theta_c*100:.1f}%)")
ax.axhline(theta_c*100, color=ORANGE, lw=1.5, linestyle="--",
           label=f"Long-run mean theta={theta_c*100:.1f}%")
ax.fill_between(times_c,
                np.percentile(paths_c, 5, axis=0)*100,
                np.percentile(paths_c, 95, axis=0)*100,
                color=GREEN, alpha=0.12, label="5th-95th percentile")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Short rate (%)")
ax.set_title("CIR — Square-Root Mean-Reversion (Always Positive)\n"
             "dr = kappa*(theta-r)dt + sigma*sqrt(r)*dW", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Terminal distribution — Vasicek
ax = fig.add_subplot(gs[0, 2])
r_grid = np.linspace(-0.02, 0.14, 300)
pdf_v, mu_v, var_v = vasicek.terminal_distribution(T_SIM, r_grid)
ax.hist(paths_v[:, -1]*100, bins=40, density=True, color=BLUE, alpha=0.5,
        label="MC histogram")
ax.plot(r_grid*100, pdf_v/100, color=ORANGE, lw=2,
        label=f"Analytic N({mu_v*100:.2f}%, {np.sqrt(var_v)*100:.2f}%)")
ax.set_xlabel("r_T (%)"); ax.set_ylabel("Density")
ax.set_title(f"Vasicek Terminal\nDistribution at T={T_SIM}Y", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Terminal distribution — CIR
ax = fig.add_subplot(gs[1, 2])
r_grid_c = np.linspace(0, 0.18, 300)
pdf_c, mu_c, var_c = cir.terminal_distribution(T_SIM, r_grid_c)
ax.hist(paths_c[:, -1]*100, bins=40, density=True, color=GREEN, alpha=0.5,
        label="MC histogram")
ax.plot(r_grid_c*100, pdf_c/100, color=ORANGE, lw=2,
        label=f"Analytic ncx2 (mu={mu_c*100:.2f}%)")
ax.set_xlabel("r_T (%)"); ax.set_ylabel("Density")
ax.set_title(f"CIR Terminal Distribution\n(Non-Central Chi-Squared) T={T_SIM}Y",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

p2 = os.path.join(OUT_DIR, "m22_02_simulation.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ==============================================================
# FIGURE 3 — Parameter sensitivity: kappa, theta, sigma
# ==============================================================
t0 = time.perf_counter()
print("[M22] Figure 3: Parameter sensitivity analysis ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M22 — Parameter Sensitivity: Vasicek (top) | CIR (bottom)\n"
             "Effect of kappa, theta, sigma on the yield curve",
             color=WHITE, fontsize=12, fontweight="bold")

tau = np.linspace(0.25, 20, 200)
colors = [BLUE, GREEN, ORANGE, RED, PURPLE]

# Vasicek — vary kappa
ax = axes[0, 0]
for i, k in enumerate([0.05, 0.15, 0.30, 0.60, 1.20]):
    y = VasicekModel(k, 0.05, 0.015, 0.02).yield_curve(tau)*100
    ax.plot(tau, y, color=colors[i], lw=2, label=f"kappa={k}")
ax.set_title("Vasicek: vary kappa\n(mean-reversion speed)", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7)
ax.grid(True); watermark(ax)

# Vasicek — vary theta
ax = axes[0, 1]
for i, th in enumerate([0.02, 0.03, 0.05, 0.07, 0.10]):
    y = VasicekModel(0.30, th, 0.015, 0.02).yield_curve(tau)*100
    ax.plot(tau, y, color=colors[i], lw=2, label=f"theta={th*100:.0f}%")
ax.set_title("Vasicek: vary theta\n(long-run mean)", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7)
ax.grid(True); watermark(ax)

# Vasicek — vary sigma
ax = axes[0, 2]
for i, s in enumerate([0.005, 0.010, 0.020, 0.040, 0.060]):
    y = VasicekModel(0.30, 0.05, s, 0.02).yield_curve(tau)*100
    ax.plot(tau, y, color=colors[i], lw=2, label=f"sigma={s*100:.1f}%")
ax.set_title("Vasicek: vary sigma\n(volatility — Ito adjustment)", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7)
ax.grid(True); watermark(ax)

# CIR — vary kappa
ax = axes[1, 0]
for i, k in enumerate([0.05, 0.15, 0.35, 0.70, 1.50]):
    y = CIRModel(k, 0.05, 0.08, 0.02).yield_curve(tau)*100
    ax.plot(tau, y, color=colors[i], lw=2, label=f"kappa={k}")
ax.set_title("CIR: vary kappa", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7)
ax.grid(True); watermark(ax)

# CIR — vary theta
ax = axes[1, 1]
for i, th in enumerate([0.02, 0.03, 0.05, 0.07, 0.10]):
    y = CIRModel(0.35, th, 0.08, 0.02).yield_curve(tau)*100
    ax.plot(tau, y, color=colors[i], lw=2, label=f"theta={th*100:.0f}%")
ax.set_title("CIR: vary theta", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7)
ax.grid(True); watermark(ax)

# CIR — vary sigma (Feller boundary)
ax = axes[1, 2]
for i, s in enumerate([0.03, 0.06, 0.10, 0.15, 0.20]):
    feller = 2*0.35*0.05/s**2
    y = CIRModel(0.35, 0.05, s, 0.02).yield_curve(tau)*100
    lbl = f"sigma={s*100:.0f}%  (Feller={feller:.2f})"
    style = "--" if feller < 1 else "-"
    ax.plot(tau, y, color=colors[i], lw=2, linestyle=style, label=lbl)
ax.set_title("CIR: vary sigma\n(dashed = Feller violated)", color=WHITE, fontsize=9)
ax.set_xlabel("Maturity"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=6.5)
ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m22_03_sensitivity.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 22 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Vasicek: dr = k(theta-r)dt + sigma*dW  (Gaussian OU)")
print("  [2] CIR:     dr = k(theta-r)dt + sigma*sqrt(r)*dW")
print("  [3] Bond price: P(0,T) = A(T)*exp(-B(T)*r0)  (closed form)")
print("  [4] CIR Feller condition: 2*kappa*theta/sigma^2 > 1")
print("  [5] Calibration: min SSE between model and observed yields")
print("  [6] Vasicek terminal: Normal | CIR terminal: Ncx2")
print(f"  Vasicek (calibrated): kappa={kv:.4f} theta={thv:.4f} sigma={sv:.4f}")
print(f"  CIR (calibrated):     kappa={kc:.4f} theta={thc:.4f} sigma={sc:.4f}")
print(f"  CIR Feller ratio: {2*kc*thc/sc**2:.4f}")
print("=" * 65)

#!/usr/bin/env python3
"""
M35 — Stochastic Volatility: Heston Model
==========================================
Module 35 | CQF Concepts Explained
Group 7   | Derivatives Pricing

Theory
------
Black-Scholes assumes constant volatility — empirically rejected by the
volatility smile/skew in option markets. Stochastic volatility models
let sigma itself follow a random process.

Heston (1993) Model
--------------------
Under risk-neutral measure Q:
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_1^Q
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_2^Q
    dW_1^Q * dW_2^Q = rho * dt

Parameters:
    v_0   : initial variance (spot variance)
    kappa : mean-reversion speed of variance
    theta : long-run variance (long-run vol = sqrt(theta))
    xi    : vol-of-vol (volatility of variance)
    rho   : correlation between spot and variance shocks
            rho < 0 => leverage effect (left skew for equity)

Feller Condition: 2*kappa*theta > xi^2
    Ensures variance process v_t stays strictly positive.

Characteristic Function (Heston, 1993)
----------------------------------------
The log-price x = ln(S_T/S_0) has characteristic function:
    phi(u) = E^Q[exp(i*u*x)]
           = exp(i*u*(r-q)*T + C(u,T) + D(u,T)*v_0)

where:
    d(u) = sqrt((kappa - i*rho*xi*u)^2 + xi^2*(i*u + u^2))
    g(u) = (kappa - i*rho*xi*u - d) / (kappa - i*rho*xi*u + d)
    C(u) = (r-q)*i*u*T + (kappa*theta/xi^2)*[(kappa-i*rho*xi*u-d)*T
            - 2*ln((1-g*exp(-d*T))/(1-g))]
    D(u) = (kappa-i*rho*xi*u-d)/xi^2 * (1-exp(-d*T))/(1-g*exp(-d*T))

European Call Price (Gil-Pelaez inversion):
    C = S*exp(-q*T)*P_1 - K*exp(-r*T)*P_2

    P_j = 1/2 + (1/pi) * integral_0^inf Re[exp(-i*u*ln(K/F)) * phi_j(u)/iu] du

where phi_1(u) = phi(u-i)/phi(-i)  and  phi_2(u) = phi(u).

Monte Carlo Simulation (Euler-Maruyama)
-----------------------------------------
Full truncation scheme for v_t (Higham-Mao):
    v_{t+dt} = v_t + kappa*(theta-v_t^+)*dt + xi*sqrt(v_t^+)*sqrt(dt)*Z2
    v_t^+ = max(v_t, 0)   (full truncation)

    S_{t+dt} = S_t * exp((r-q-v_t^+/2)*dt + sqrt(v_t^+*dt)*Z1)
    Z1 = eps_1; Z2 = rho*eps_1 + sqrt(1-rho^2)*eps_2
    eps_1, eps_2 ~ N(0,1) independent

References
----------
- Heston, S. (1993). A closed-form solution for options with stochastic
  volatility with applications to bond and currency options.
  Review of Financial Studies, 6(2), 327-343.
- Broadie, M., Kaya, O. (2006). Exact simulation of stochastic volatility
  and other affine jump-diffusion processes. Operations Research, 54(2).
- Gatheral, J. (2006). The Volatility Surface. Wiley Finance.
- Bergomi, L. (2016). Stochastic Volatility Modeling. CRC Press.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize

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
# SECTION 1 — HESTON CHARACTERISTIC FUNCTION & PRICER
# ============================================================

def heston_cf(u, S0, K, r, q, T, v0, kappa, theta, xi, rho):
    """
    Heston characteristic function phi(u) = E[exp(i*u*ln(S_T/S_0))].
    Uses the 'Little Trap' formulation (Albrecher et al. 2007)
    to avoid branch-cut issues.
    """
    i   = 1j
    lnS = np.log(S0)
    b   = kappa - rho*xi*i*u
    d   = np.sqrt(b**2 + xi**2*(i*u + u**2))
    # Little Trap: use (b-d) in numerator instead of (b+d)
    g   = (b - d) / (b + d)
    exp_dT = np.exp(-d*T)

    C = ((r-q)*i*u*T
         + kappa*theta/xi**2 * ((b-d)*T - 2*np.log((1-g*exp_dT)/(1-g))))
    D = (b-d)/xi**2 * (1-exp_dT)/(1-g*exp_dT)
    return np.exp(C + D*v0 + i*u*lnS)

def heston_call(S0, K, r, q, T, v0, kappa, theta, xi, rho,
                n_quad=128, u_max=100.0):
    """
    Heston European call price via Gil-Pelaez Fourier inversion.
    C = S*exp(-q*T)*P1 - K*exp(-r*T)*P2
    """
    lnK = np.log(K)
    F   = S0 * np.exp((r-q)*T)

    def integrand_P2(u):
        phi = heston_cf(u, S0, K, r, q, T, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j*u*lnK) * phi / (1j*u))

    def integrand_P1(u):
        phi_u = heston_cf(u-1j, S0, K, r, q, T, v0, kappa, theta, xi, rho)
        phi_0 = heston_cf(-1j,  S0, K, r, q, T, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j*u*lnK) * phi_u / (1j*u*phi_0))

    I2, _ = quad(integrand_P2, 1e-6, u_max, limit=200)
    I1, _ = quad(integrand_P1, 1e-6, u_max, limit=200)
    P2 = 0.5 + I2/np.pi
    P1 = 0.5 + I1/np.pi
    return S0*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2

def bs_call(S, K, r, T, sigma, q=0.0):
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_iv(price, S, K, r, T, q=0.0):
    """Brent's method to invert BS for IV."""
    from scipy.optimize import brentq
    intrinsic = max(S*np.exp(-q*T) - K*np.exp(-r*T), 0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, r, T, s, q)-price,
                      1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan

# ============================================================
# SECTION 2 — MONTE CARLO SIMULATION (HESTON)
# ============================================================

def heston_mc(S0, K, r, q, T, v0, kappa, theta, xi, rho,
              n_steps=252, n_paths=20000, seed=42):
    """
    Euler-Maruyama simulation of Heston model with full truncation.
    Returns (S_paths, v_paths) of shape (n_paths, n_steps+1).
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_paths, n_steps+1)); S[:, 0] = S0
    v = np.zeros((n_paths, n_steps+1)); v[:, 0] = v0

    for i in range(n_steps):
        eps1 = rng.standard_normal(n_paths)
        eps2 = rng.standard_normal(n_paths)
        Z1   = eps1
        Z2   = rho*eps1 + np.sqrt(1-rho**2)*eps2

        v_pos = np.maximum(v[:, i], 0.0)   # full truncation
        v[:, i+1] = (v[:, i]
                     + kappa*(theta - v_pos)*dt
                     + xi*np.sqrt(v_pos)*sqrt_dt*Z2)
        v[:, i+1] = np.maximum(v[:, i+1], 0.0)

        S[:, i+1] = S[:, i] * np.exp(
            (r - q - 0.5*v_pos)*dt + np.sqrt(v_pos)*sqrt_dt*Z1
        )

    return S, v

# ============================================================
# SECTION 3 — DIAGNOSTICS
# ============================================================

S0 = 100.0; K0 = 100.0; r = 0.05; q = 0.02; T = 1.0

# Base Heston parameters (equity calibrated)
PARAMS = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
feller = 2*PARAMS["kappa"]*PARAMS["theta"] / PARAMS["xi"]**2

print(f"[M35] Heston Model")
print(f"      Params: v0={PARAMS['v0']}  kappa={PARAMS['kappa']}  "
      f"theta={PARAMS['theta']}  xi={PARAMS['xi']}  rho={PARAMS['rho']}")
print(f"      Feller ratio: 2*kappa*theta/xi^2 = {feller:.4f}  "
      f"({'OK' if feller>1 else 'VIOLATED'})")

h_price = heston_call(S0, K0, r, q, T, **PARAMS)
bs_atm  = bs_call(S0, K0, r, T, np.sqrt(PARAMS["theta"]), q)
print(f"      Heston ATM call = {h_price:.6f}")
print(f"      BS (flat vol)   = {bs_atm:.6f}")

# IV smile across strikes
strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
iv_heston = []
print(f"\n[M35] Implied vol smile:")
for K_ in strikes:
    p_h = heston_call(S0, K_, r, q, T, **PARAMS)
    iv_ = bs_iv(p_h, S0, K_, r, T, q)
    iv_heston.append(iv_)
    print(f"      K={K_:4.0f}: Heston={p_h:.4f}  IV={iv_*100:.2f}%")

# MC simulation
print(f"\n[M35] Monte Carlo simulation (20,000 paths, 252 steps) ...")
S_paths, v_paths = heston_mc(S0, K0, r, q, T, **PARAMS,
                              n_steps=252, n_paths=20000)

payoffs = np.maximum(S_paths[:, -1] - K0, 0)
mc_price = np.exp(-r*T) * payoffs.mean()
mc_se    = np.exp(-r*T) * payoffs.std() / np.sqrt(len(payoffs))
print(f"      MC price = {mc_price:.4f}  SE = {mc_se:.4f}")
print(f"      Fourier  = {h_price:.4f}  diff = {abs(mc_price-h_price):.4f}")

# ============================================================
# FIGURE 1 — Paths, variance evolution, terminal distribution
# ============================================================
t0 = time.perf_counter()
print("\n[M35] Figure 1: Simulated paths and distributions ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M35 — Heston Model: Monte Carlo Simulation\n"
             "dS=r*S*dt+sqrt(v)*S*dW1  |  dv=kappa*(theta-v)*dt+xi*sqrt(v)*dW2",
             color=WHITE, fontsize=12, fontweight="bold")

times_plot = np.linspace(0, T, 253)

# (a) Stock price paths
ax = axes[0]
for j in range(100):
    ax.plot(times_plot, S_paths[j], lw=0.4, alpha=0.25, color=BLUE)
ax.plot(times_plot, S_paths.mean(axis=0), color=WHITE, lw=2,
        label=f"Mean path")
ax.fill_between(times_plot,
                np.percentile(S_paths, 5,  axis=0),
                np.percentile(S_paths, 95, axis=0),
                color=BLUE, alpha=0.15, label="5th-95th pct")
ax.axhline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.6)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Stock price S_t")
ax.set_title("Heston Stock Price Paths (100 shown)\n"
             "Volatility clusters visible in path dispersion",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Variance paths
ax = axes[1]
for j in range(100):
    ax.plot(times_plot, np.sqrt(v_paths[j])*100, lw=0.4, alpha=0.25,
            color=GREEN)
ax.plot(times_plot, np.sqrt(v_paths.mean(axis=0))*100, color=WHITE,
        lw=2, label="Mean vol path")
ax.axhline(np.sqrt(PARAMS["theta"])*100, color=ORANGE, lw=2,
           linestyle="--", label=f"LR vol = {np.sqrt(PARAMS['theta'])*100:.1f}%")
ax.fill_between(times_plot,
                np.sqrt(np.percentile(v_paths,5,axis=0))*100,
                np.sqrt(np.percentile(v_paths,95,axis=0))*100,
                color=GREEN, alpha=0.15)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Instantaneous vol sqrt(v_t) (%)")
ax.set_title("Variance Process sqrt(v_t)\n"
             "Mean-reverting to theta=4% with vol-of-vol xi=30%",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Terminal distribution: Heston vs Lognormal (BS)
ax = axes[2]
S_T = S_paths[:, -1]
S_grid = np.linspace(S_T.min(), S_T.max(), 300)
# Lognormal reference
mu_ln = np.log(S0)+(r-q-0.5*PARAMS["theta"])*T
sig_ln = np.sqrt(PARAMS["theta"]*T)
pdf_ln = (1/(S_grid*sig_ln*np.sqrt(2*np.pi))
          *np.exp(-(np.log(S_grid)-mu_ln)**2/(2*sig_ln**2)))

ax.hist(S_T, bins=60, density=True, color=BLUE, alpha=0.55,
        label="Heston MC terminal S_T")
ax.plot(S_grid, pdf_ln, color=ORANGE, lw=2.5,
        label="Lognormal (BS flat vol)")
ax.axvline(K0, color=RED, lw=1.5, linestyle="--", label=f"Strike K={K0}")
ax.set_xlabel("S_T"); ax.set_ylabel("Density")
ax.set_title("Terminal Distribution: Heston vs Lognormal\n"
             "Heston: heavier left tail (rho=-0.7 => skew)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m35_01_paths.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — IV smile and parameter sensitivity
# ============================================================
t0 = time.perf_counter()
print("[M35] Figure 2: IV smile and parameter sensitivity ...")

K_arr  = np.linspace(75, 125, 25)
T_arr  = [0.25, 0.50, 1.00, 2.00]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M35 — Heston Implied Volatility Surface\n"
             "Smile | Skew vs rho | Vol-of-vol vs xi",
             color=WHITE, fontsize=12, fontweight="bold")

colors4 = [BLUE, GREEN, ORANGE, RED]

# (a) IV smile for different maturities
ax = axes[0]
for T_, col in zip(T_arr, colors4):
    iv_T = []
    for K_ in K_arr:
        p_h = heston_call(S0, K_, r, q, T_, **PARAMS)
        iv_T.append(bs_iv(p_h, S0, K_, r, T_, q) or np.nan)
    ax.plot(K_arr, np.array(iv_T)*100, color=col, lw=2,
            label=f"T={T_}Y")
ax.axvline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.5, label="ATM")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("Heston IV Smile — Multiple Maturities\n"
             "Left skew from rho=-0.7 (leverage effect)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Effect of rho on smile shape
ax = axes[1]
rho_vals = [-0.8, -0.4, 0.0, 0.4, 0.8]
rho_cols  = [RED, ORANGE, WHITE, GREEN, BLUE]
for rho_, col in zip(rho_vals, rho_cols):
    p_rho = PARAMS.copy(); p_rho["rho"] = rho_
    iv_rho = []
    for K_ in K_arr:
        p_h = heston_call(S0, K_, r, q, 1.0, **p_rho)
        iv_rho.append(bs_iv(p_h, S0, K_, r, 1.0, q) or np.nan)
    ax.plot(K_arr, np.array(iv_rho)*100, color=col, lw=2,
            label=f"rho={rho_}")
ax.axvline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("IV Smile vs Correlation rho\n"
             "rho<0: left skew | rho>0: right skew",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Effect of xi (vol-of-vol) on smile curvature
ax = axes[2]
xi_vals = [0.10, 0.20, 0.30, 0.50, 0.80]
xi_cols  = [BLUE, GREEN, ORANGE, RED, PURPLE]
for xi_, col in zip(xi_vals, xi_cols):
    p_xi = PARAMS.copy(); p_xi["xi"] = xi_
    # Check Feller
    if 2*p_xi["kappa"]*p_xi["theta"] < xi_**2:
        label = f"xi={xi_} (Feller viol.)"
        ls = "--"
    else:
        label = f"xi={xi_}"; ls = "-"
    iv_xi = []
    for K_ in K_arr:
        try:
            p_h = heston_call(S0, K_, r, q, 1.0, **p_xi)
            iv_xi.append(bs_iv(p_h, S0, K_, r, 1.0, q) or np.nan)
        except Exception:
            iv_xi.append(np.nan)
    ax.plot(K_arr, np.array(iv_xi)*100, color=col, lw=2,
            linestyle=ls, label=label)
ax.axvline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("IV Smile vs Vol-of-Vol xi\n"
             "Higher xi => more curvature (wider smile)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m35_02_iv_smile.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Calibration and Fourier vs MC comparison
# ============================================================
t0 = time.perf_counter()
print("[M35] Figure 3: Calibration and Fourier vs MC accuracy ...")

# Synthetic "market" smile using Heston with known params
PARAMS_TRUE = dict(v0=0.05, kappa=1.5, theta=0.05, xi=0.40, rho=-0.65)
K_calib = np.array([85, 90, 95, 100, 105, 110, 115])
T_calib = 1.0

iv_mkt = []
for K_ in K_calib:
    p_h = heston_call(S0, K_, r, q, T_calib, **PARAMS_TRUE)
    iv_mkt.append(bs_iv(p_h, S0, K_, r, T_calib, q))
iv_mkt = np.array(iv_mkt)

# Calibrate (minimize SSE of IV)
def calib_objective(x):
    v0_, kappa_, theta_, xi_, rho_ = x
    if (v0_<=0 or kappa_<=0 or theta_<=0 or xi_<=0
            or rho_<=-1 or rho_>=1):
        return 1e10
    sse = 0
    for K_, iv_obs in zip(K_calib, iv_mkt):
        try:
            p_h  = heston_call(S0, K_, r, q, T_calib, v0_, kappa_,
                               theta_, xi_, rho_)
            iv_m = bs_iv(p_h, S0, K_, r, T_calib, q)
            if iv_m and not np.isnan(iv_m):
                sse += (iv_m - iv_obs)**2
            else:
                sse += 1e4
        except Exception:
            sse += 1e4
    return sse

x0  = [0.04, 2.0, 0.04, 0.3, -0.7]
bds = [(0.001,0.5),(0.1,10),(0.001,0.5),(0.01,1.5),(-0.99,0.99)]
res = minimize(calib_objective, x0, method="L-BFGS-B", bounds=bds,
               options={"maxiter":300, "ftol":1e-14})
v0_c, kap_c, th_c, xi_c, rho_c = res.x
PARAMS_CAL = dict(v0=v0_c, kappa=kap_c, theta=th_c, xi=xi_c, rho=rho_c)

print(f"      True:  v0={PARAMS_TRUE['v0']}  kappa={PARAMS_TRUE['kappa']}  "
      f"theta={PARAMS_TRUE['theta']}  xi={PARAMS_TRUE['xi']}  rho={PARAMS_TRUE['rho']}")
print(f"      Calib: v0={v0_c:.4f}  kappa={kap_c:.4f}  "
      f"theta={th_c:.4f}  xi={xi_c:.4f}  rho={rho_c:.4f}")

iv_cal = []
for K_ in K_calib:
    p_h = heston_call(S0, K_, r, q, T_calib, **PARAMS_CAL)
    iv_cal.append(bs_iv(p_h, S0, K_, r, T_calib, q))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M35 — Heston Calibration and Fourier vs MC\n"
             "Fit market IV smile | Pricing accuracy comparison",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Calibration fit
ax = axes[0]
ax.plot(K_calib, iv_mkt*100,  color=YELLOW, lw=2.5, marker="o", ms=8,
        label="Market IV (synthetic)")
ax.plot(K_calib, np.array(iv_cal)*100, color=RED, lw=2,
        marker="s", ms=6, linestyle="--", label="Heston (calibrated)")
for K_, iv_o, iv_c in zip(K_calib, iv_mkt, iv_cal):
    if iv_c:
        ax.annotate(f"{(iv_c-iv_o)*100:+.2f}%", (K_, iv_c*100),
                    textcoords="offset points", xytext=(0,6),
                    fontsize=6, color=RED)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("Heston Calibration to IV Smile\n"
             "Minimise sum((IV_model-IV_mkt)^2)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Fourier vs MC pricing accuracy across strikes
ax = axes[1]
K_comp = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
fourier_prices = [heston_call(S0, K_, r, q, T, **PARAMS) for K_ in K_comp]
# MC prices via same paths
mc_prices = []
for K_ in K_comp:
    pays = np.maximum(S_paths[:, -1] - K_, 0)
    mc_prices.append(np.exp(-r*T)*pays.mean())

ax.plot(K_comp, fourier_prices, color=BLUE, lw=2.5, marker="o", ms=6,
        label="Fourier (semi-analytic)")
ax.plot(K_comp, mc_prices, color=GREEN, lw=2, marker="s", ms=6,
        linestyle="--", label="Monte Carlo (20k paths)")
# Error bars
mc_se_arr = [np.exp(-r*T)*np.maximum(S_paths[:,-1]-K_,0).std()
             /np.sqrt(len(S_paths)) for K_ in K_comp]
ax.fill_between(K_comp,
                np.array(mc_prices)-2*np.array(mc_se_arr),
                np.array(mc_prices)+2*np.array(mc_se_arr),
                color=GREEN, alpha=0.20, label="MC ± 2 SE")
ax.set_xlabel("Strike K"); ax.set_ylabel("Call price")
ax.set_title("Fourier vs MC Pricing\n"
             "Fourier exact, MC has sampling error",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Mean-reversion of variance: kappa effect
ax = axes[2]
times_v = np.linspace(0, T, 253)
v_mean  = v_paths.mean(axis=0)
v_theory = PARAMS["theta"] + (PARAMS["v0"]-PARAMS["theta"])*np.exp(
    -PARAMS["kappa"]*times_v)

ax.plot(times_v, np.sqrt(v_mean)*100, color=GREEN, lw=2.5,
        label="MC mean vol sqrt(E[v_t])")
ax.plot(times_v, np.sqrt(v_theory)*100, color=ORANGE, lw=2,
        linestyle="--",
        label=f"Theory: sqrt(theta+(v0-theta)e^{{-kt}})")
ax.axhline(np.sqrt(PARAMS["theta"])*100, color=RED, lw=1.5,
           linestyle=":", label=f"LR mean vol = {np.sqrt(PARAMS['theta'])*100:.1f}%")
ax.fill_between(times_v,
                np.sqrt(np.percentile(v_paths,10,axis=0))*100,
                np.sqrt(np.percentile(v_paths,90,axis=0))*100,
                color=GREEN, alpha=0.15, label="10th-90th pct")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Instantaneous vol (%)")
ax.set_title(f"Variance Mean-Reversion\n"
             f"kappa={PARAMS['kappa']}  theta={PARAMS['theta']}  "
             f"v0={PARAMS['v0']}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m35_03_calibration.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 35 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Heston: dv=kappa*(theta-v)*dt + xi*sqrt(v)*dW2")
print("  [2] Feller: 2*kappa*theta > xi^2 => v_t > 0 a.s.")
print("  [3] rho<0 => leverage effect => left skew in IV")
print("  [4] xi controls smile curvature (vol-of-vol)")
print("  [5] Semi-analytic price via Fourier inversion (Gil-Pelaez)")
print("  [6] Calibration: min SSE of model vs market IV smile")
print(f"  Feller ratio: {feller:.4f}  ({'satisfied' if feller>1 else 'violated'})")
print(f"  Fourier price (ATM): {h_price:.6f}")
print(f"  MC price     (ATM): {mc_price:.4f}  SE={mc_se:.4f}")
print(f"  Calibration SSE: {res.fun*10000:.4f} (bps^2 units)")
print("=" * 65)

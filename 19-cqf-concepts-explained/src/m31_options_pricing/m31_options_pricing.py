#!/usr/bin/env python3
"""
M31 — Options Pricing: Black-Scholes and Extensions
=====================================================
Module 31 | CQF Concepts Explained
Group 7   | Derivatives Pricing

Theory
------
Black-Scholes-Merton Model (1973)
----------------------------------
Assume the underlying follows GBM under the risk-neutral measure Q:
    dS_t = r * S_t * dt + sigma * S_t * dW_t^Q

The no-arbitrage price of a European call satisfies the PDE:
    dV/dt + (1/2)*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

Closed-form solution (Black-Scholes formula):
    C = S*N(d1) - K*exp(-r*T)*N(d2)
    P = K*exp(-r*T)*N(-d2) - S*N(-d1)

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Interpretation:
    S*N(d1)        = risk-neutral expected asset price * P(exercise)
    K*exp(-r*T)*N(d2) = PV of strike * risk-neutral probability of exercise
    N(d2)          = Q(S_T > K) = risk-neutral probability ITM

The Greeks (sensitivities)
---------------------------
Delta:  dC/dS    = N(d1)                      [hedge ratio]
Gamma:  d2C/dS2  = N'(d1) / (S*sigma*sqrt(T)) [convexity]
Theta:  dC/dt    = -S*N'(d1)*sigma/(2*sqrt(T)) - r*K*e^{-rT}*N(d2)
Vega:   dC/dsigma = S*N'(d1)*sqrt(T)           [vol sensitivity]
Rho:    dC/dr    = K*T*e^{-rT}*N(d2)           [rate sensitivity]

Put-Call Parity
---------------
    C - P = S*exp(-q*T) - K*exp(-r*T)
Holds regardless of the model (pure no-arbitrage).

Extensions
-----------
Merton (1973) — continuous dividend yield q:
    d1 = [ln(S/K) + (r - q + sigma^2/2)*T] / (sigma*sqrt(T))
    C  = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)

Black (1976) — options on futures:
    C  = exp(-r*T) * [F*N(d1) - K*N(d2)]
    d1 = [ln(F/K) + sigma^2*T/2] / (sigma*sqrt(T))

Implied Volatility
-------------------
sigma_imp = BS^{-1}(C_mkt; S, K, r, T)
Solved numerically (Newton-Raphson or Brent's method):
    sigma_{n+1} = sigma_n - (BS(sigma_n) - C_mkt) / Vega(sigma_n)

Volatility Smile / Skew
-------------------------
Under BS, IV should be constant across strikes. In practice:
    - Equity options: left skew (downside puts more expensive)
    - FX options: symmetric smile
    - Commodity options: right skew (upside calls expensive)
The smile reflects fat tails, jumps, and stochastic volatility
in the true data-generating process.

References
----------
- Black, F., Scholes, M. (1973). The pricing of options and corporate
  liabilities. Journal of Political Economy, 81(3), 637-654.
- Merton, R.C. (1973). Theory of rational option pricing.
  Bell Journal of Economics, 4(1), 141-183.
- Black, F. (1976). The pricing of commodity contracts.
  Journal of Financial Economics, 3(1-2), 167-179.
- Hull, J.C. (2022). Options, Futures, and Other Derivatives. Ch. 15-19.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

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
# SECTION 1 — BLACK-SCHOLES ENGINE
# ============================================================

def bs_d1d2(S, K, r, T, sigma, q=0.0):
    """Compute d1 and d2 for Black-Scholes-Merton."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    return d1, d2

def bs_price(S, K, r, T, sigma, q=0.0, option="call"):
    """Black-Scholes-Merton option price."""
    d1, d2 = bs_d1d2(S, K, r, T, sigma, q)
    if option == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_greeks(S, K, r, T, sigma, q=0.0, option="call"):
    """Compute all five primary Greeks."""
    d1, d2 = bs_d1d2(S, K, r, T, sigma, q)
    sqrt_T  = np.sqrt(T)
    npd1    = norm.pdf(d1)
    exp_qT  = np.exp(-q*T)
    exp_rT  = np.exp(-r*T)

    gamma   = exp_qT * npd1 / (S * sigma * sqrt_T)
    vega    = S * exp_qT * npd1 * sqrt_T / 100   # per 1% vol move

    if option == "call":
        delta = exp_qT * norm.cdf(d1)
        theta = (- S*exp_qT*npd1*sigma/(2*sqrt_T)
                 - r*K*exp_rT*norm.cdf(d2)
                 + q*S*exp_qT*norm.cdf(d1)) / 365
        rho   = K*T*exp_rT*norm.cdf(d2) / 100
    else:
        delta = -exp_qT * norm.cdf(-d1)
        theta = (- S*exp_qT*npd1*sigma/(2*sqrt_T)
                 + r*K*exp_rT*norm.cdf(-d2)
                 - q*S*exp_qT*norm.cdf(-d1)) / 365
        rho   = -K*T*exp_rT*norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "rho": rho}

def implied_vol(price_mkt, S, K, r, T, q=0.0, option="call"):
    """Brent's method to invert BS for implied volatility."""
    intrinsic = max(bs_price(S, K, r, T, 1e-6, q, option), 0)
    if price_mkt <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(
            lambda s: bs_price(S, K, r, T, s, q, option) - price_mkt,
            1e-4, 5.0, xtol=1e-8, maxiter=200
        )
    except ValueError:
        return np.nan

# ============================================================
# SECTION 2 — BASE CASE DIAGNOSTICS
# ============================================================

S0 = 100.0; K0 = 100.0; r0 = 0.05; T0 = 1.0; sigma0 = 0.20; q0 = 0.02

C0 = bs_price(S0, K0, r0, T0, sigma0, q0, "call")
P0 = bs_price(S0, K0, r0, T0, sigma0, q0, "put")
gk = bs_greeks(S0, K0, r0, T0, sigma0, q0, "call")

print(f"[M31] Black-Scholes-Merton Base Case")
print(f"      S={S0}, K={K0}, r={r0*100:.0f}%, T={T0}Y, "
      f"sigma={sigma0*100:.0f}%, q={q0*100:.0f}%")
print(f"      Call price = {C0:.4f}  |  Put price = {P0:.4f}")
pcp = C0 - P0 - (S0*np.exp(-q0*T0) - K0*np.exp(-r0*T0))
print(f"      Put-Call Parity check: C-P-(Fe^{{-rT}}-Ke^{{-rT}}) = {pcp:.2e}")
print(f"      Greeks (call):")
for k, v in gk.items():
    print(f"        {k:8s} = {v:10.6f}")

# Simulate volatility smile (equity left skew via mixing model)
def synthetic_iv_smile(K_arr, S, r, T, atm_vol=0.20, skew=-0.3, kurt=0.5):
    """
    Parametric IV smile: sigma(K) = atm_vol + skew*(ln(K/S)) + kurt*(ln(K/S))^2
    Negative skew: puts more expensive (equity downside protection premium).
    """
    log_m = np.log(K_arr / S)
    return atm_vol + skew * log_m + kurt * log_m**2

# ============================================================
# FIGURE 1 — BS price surface and Greeks
# ============================================================
t0 = time.perf_counter()
print("\n[M31] Figure 1: BS price surface and Greeks profiles ...")

S_arr = np.linspace(60, 140, 200)
T_arr = np.linspace(0.05, 2.0, 200)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M31 — Black-Scholes Option Pricing\n"
             "Price Surface | Greeks | Put-Call Parity",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Call price vs spot for several maturities
ax = axes[0]
for T_, col in [(0.25,BLUE),(0.50,GREEN),(1.00,ORANGE),(2.00,RED)]:
    prices = [bs_price(s, K0, r0, T_, sigma0, q0, "call") for s in S_arr]
    ax.plot(S_arr, prices, color=col, lw=2, label=f"T={T_}Y")
# Intrinsic value
ax.plot(S_arr, np.maximum(S_arr - K0, 0), color=WHITE, lw=1.5,
        linestyle=":", alpha=0.7, label="Intrinsic (T=0)")
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5, label=f"K={K0}")
ax.set_xlabel("Spot S"); ax.set_ylabel("Call price")
ax.set_title(f"BS Call Price vs Spot\nK={K0}, r={r0*100:.0f}%, "
             f"sigma={sigma0*100:.0f}%, q={q0*100:.0f}%",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Greeks vs spot (T=1Y ATM)
ax = axes[1]
delta_arr = [bs_greeks(s, K0, r0, T0, sigma0, q0, "call")["delta"] for s in S_arr]
gamma_arr = [bs_greeks(s, K0, r0, T0, sigma0, q0, "call")["gamma"]*100 for s in S_arr]
vega_arr  = [bs_greeks(s, K0, r0, T0, sigma0, q0, "call")["vega"]  for s in S_arr]

ax2 = ax.twinx()
ax.plot(S_arr, delta_arr, color=BLUE,   lw=2.5, label="Delta")
ax.plot(S_arr, vega_arr,  color=GREEN,  lw=2,   label="Vega (per 1% vol)")
ax2.plot(S_arr, gamma_arr, color=ORANGE, lw=2, linestyle="--",
         label="Gamma x100")
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Delta / Vega")
ax2.set_ylabel("Gamma x100", color=ORANGE)
ax.set_title("Call Greeks vs Spot  (T=1Y)\nDelta: hedge ratio | Gamma: convexity",
             color=WHITE, fontsize=9)
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=7)
ax.grid(True); watermark(ax)

# (c) Theta decay for ATM call
ax = axes[2]
T_decay = np.linspace(0.01, 2.0, 300)
for moneyness, lbl, col in [(0.9,"ITM S=110",BLUE),(1.0,"ATM S=100",GREEN),
                             (1.1,"OTM S=90", ORANGE)]:
    S_m = K0 * moneyness
    th  = [bs_greeks(S_m, K0, r0, t, sigma0, q0, "call")["theta"]
           for t in T_decay]
    pr  = [bs_price(S_m, K0, r0, t, sigma0, q0, "call") for t in T_decay]
    ax.plot(T_decay, th, color=col, lw=2, label=lbl)
ax.axhline(0, color=WHITE, lw=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("Time to expiry T (years)")
ax.set_ylabel("Theta (per calendar day)")
ax.set_title("Theta Decay vs Time to Expiry\n"
             "ATM theta peaks near expiry (time decay accelerates)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m31_01_bs_price_greeks.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Implied Volatility surface (smile/skew)
# ============================================================
t0 = time.perf_counter()
print("[M31] Figure 2: Implied volatility smile and surface ...")

K_arr   = np.linspace(70, 130, 100)
T_smiles = [0.25, 0.50, 1.00, 2.00]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M31 — Implied Volatility: Smile, Skew, and Surface\n"
             "Deviation from BS flat vol assumption",
             color=WHITE, fontsize=12, fontweight="bold")

smile_cols = [BLUE, GREEN, ORANGE, RED]

# (a) IV smile: equity left skew
ax = axes[0]
for T_, col in zip(T_smiles, smile_cols):
    iv_eq = synthetic_iv_smile(K_arr, S0, r0, T_, atm_vol=0.20,
                               skew=-0.30, kurt=0.40)
    ax.plot(K_arr, iv_eq*100, color=col, lw=2, label=f"T={T_}Y")
ax.axvline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.6, label="ATM")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("Equity Volatility Skew\n"
             "Puts more expensive: crash risk / leverage effect",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) IV smile: FX symmetric smile
ax = axes[1]
for T_, col in zip(T_smiles, smile_cols):
    iv_fx = synthetic_iv_smile(K_arr, S0, r0, T_, atm_vol=0.12,
                               skew=0.00, kurt=0.80)
    ax.plot(K_arr, iv_fx*100, color=col, lw=2, label=f"T={T_}Y")
ax.axvline(S0, color=YELLOW, lw=1, linestyle="--", alpha=0.6, label="ATM")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility (%)")
ax.set_title("FX Volatility Smile (Symmetric)\n"
             "Both tails expensive: jump risk in both directions",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) 3D IV surface (equity skew)
ax = axes[2]
K_3d  = np.linspace(75, 125, 40)
T_3d  = np.linspace(0.1, 2.0, 30)
KK, TT = np.meshgrid(K_3d, T_3d)
IV_3d = synthetic_iv_smile(KK, S0, r0, TT, atm_vol=0.20,
                           skew=-0.25, kurt=0.35) * 100
c = ax.contourf(K_3d, T_3d, IV_3d, levels=25, cmap="plasma")
plt.colorbar(c, ax=ax, label="IV (%)")
ax.set_xlabel("Strike K"); ax.set_ylabel("Maturity T (years)")
ax.set_title("Implied Volatility Surface (Equity)\n"
             "Term structure + strike skew",
             color=WHITE, fontsize=9)
watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m31_02_iv_smile_surface.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Extensions: Merton dividends, Black-76, IV inversion
# ============================================================
t0 = time.perf_counter()
print("[M31] Figure 3: Extensions and IV inversion (Newton-Raphson) ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M31 — BS Extensions and Implied Volatility Inversion\n"
             "Merton (dividends) | Black-76 (futures) | Newton-Raphson IV",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Effect of dividend yield on call price
ax = axes[0]
q_arr = np.linspace(0, 0.10, 200)
for T_, col in [(0.25,BLUE),(0.50,GREEN),(1.00,ORANGE),(2.00,RED)]:
    prices_q = [bs_price(S0, K0, r0, T_, sigma0, q_, "call") for q_ in q_arr]
    ax.plot(q_arr*100, prices_q, color=col, lw=2, label=f"T={T_}Y")
ax.set_xlabel("Continuous dividend yield q (%)")
ax.set_ylabel("Call price")
ax.set_title("Merton Extension: Effect of Dividends\n"
             "Higher q reduces call price (forward price falls)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Black-76: option on futures
ax = axes[1]
F_arr = np.linspace(70, 130, 200)
# Black-76: C = e^{-rT}[F*N(d1) - K*N(d2)], d1=[ln(F/K)+sigma^2*T/2]/(sigma*sqrt(T))
def black76_call(F, K, r, T, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    return np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))

for T_, col in [(0.25,BLUE),(0.50,GREEN),(1.00,ORANGE)]:
    b76  = [black76_call(f, K0, r0, T_, sigma0) for f in F_arr]
    bs_c = [bs_price(f, K0, r0, T_, sigma0, q=r0, option="call")
            for f in F_arr]    # BS with q=r gives cost-of-carry = 0
    ax.plot(F_arr, b76,  color=col, lw=2.5, label=f"Black-76 T={T_}Y")
    ax.plot(F_arr, bs_c, color=col, lw=1.5, linestyle="--", alpha=0.6)
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5, label=f"K={K0}")
ax.set_xlabel("Futures price F"); ax.set_ylabel("Call price")
ax.set_title("Black-76: Options on Futures\nSolid=Black-76  Dashed=BS(q=r)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Newton-Raphson IV convergence for one option
ax = axes[2]
# Generate a market price and show NR iterations
C_mkt  = 10.5   # observed market price
sigma_init = 0.30
sigma_it   = sigma_init
iters_s = [sigma_it]; iters_err = [abs(bs_price(S0,K0,r0,T0,sigma_it,q0)-C_mkt)]
for _ in range(20):
    C_it   = bs_price(S0, K0, r0, T0, sigma_it, q0, "call")
    vega_it = bs_greeks(S0, K0, r0, T0, sigma_it, q0, "call")["vega"] * 100
    if abs(vega_it) < 1e-10:
        break
    sigma_it -= (C_it - C_mkt) / vega_it
    sigma_it  = max(sigma_it, 1e-4)
    iters_s.append(sigma_it)
    err = abs(bs_price(S0,K0,r0,T0,sigma_it,q0)-C_mkt)
    iters_err.append(err)
    if err < 1e-8:
        break

iv_true = implied_vol(C_mkt, S0, K0, r0, T0, q0, "call")
print(f"      IV inversion: C_mkt={C_mkt:.2f}  =>  sigma_IV={iv_true*100:.4f}%")
print(f"      Newton-Raphson converged in {len(iters_s)-1} iterations")

ax.semilogy(range(len(iters_err)), iters_err, color=GREEN, lw=2.5,
            marker="o", ms=6, label="NR: |C_model - C_mkt|")
ax.axhline(1e-8, color=YELLOW, lw=1.5, linestyle="--",
           label="Convergence threshold 1e-8")
for i, (s_, e_) in enumerate(zip(iters_s, iters_err)):
    if i < 5:
        ax.annotate(f"s={s_*100:.2f}%", (i, e_),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=ORANGE)
ax.set_xlabel("Iteration"); ax.set_ylabel("|Error| (log scale)")
ax.set_title(f"Newton-Raphson IV Inversion\n"
             f"C_mkt={C_mkt:.2f}  =>  IV={iv_true*100:.3f}%  "
             f"({len(iters_s)-1} iters)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m31_03_extensions_iv.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
d1_0, d2_0 = bs_d1d2(S0, K0, r0, T0, sigma0, q0)
print()
print("=" * 65)
print("  MODULE 31 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] C = S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)  (BSM formula)")
print("  [2] N(d2) = risk-neutral P(S_T > K) at expiry")
print("  [3] Delta=N(d1): shares to hold for delta-neutral hedge")
print("  [4] Gamma: convexity — gains from large moves in either direction")
print("  [5] Theta: time decay accelerates near expiry for ATM options")
print("  [6] IV smile: fat tails + jumps beyond lognormal assumption")
print(f"  Base call: C={C0:.4f}  d1={d1_0:.4f}  d2={d2_0:.4f}")
print(f"  PCP check: {pcp:.2e}  (zero = parity holds)")
print(f"  Greeks: Delta={gk['delta']:.4f}  Gamma={gk['gamma']:.6f}  "
      f"Vega={gk['vega']:.4f}  Theta={gk['theta']:.6f}")
print(f"  IV inversion (C={C_mkt}): sigma_IV = {iv_true*100:.4f}%")
print("=" * 65)

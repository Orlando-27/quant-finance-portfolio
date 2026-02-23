#!/usr/bin/env python3
"""
M25 — Credit Risk: PD, LGD, CVA
=================================
Module 25 | CQF Concepts Explained
Group 5   | Credit & Counterparty Risk

Theory
------
Credit risk is the risk that a counterparty fails to meet its
contractual obligations. Three core components:

    EL = PD * LGD * EAD

    PD  : Probability of Default in a given horizon
    LGD : Loss Given Default = 1 - Recovery Rate (R)
    EAD : Exposure at Default

Reduced-Form (Intensity) Model
--------------------------------
Default time tau is the first jump of a Poisson process with
(possibly stochastic) intensity lambda_t (hazard rate).

Survival probability to T:
    Q(0, T) = E[exp(-integral_0^T lambda_t dt)]

For constant lambda:
    Q(0, T) = exp(-lambda * T)

Marginal PD in [T_{i-1}, T_i]:
    PD_i = Q(0, T_{i-1}) - Q(0, T_i) = Q(T_{i-1}) * (1 - exp(-lambda*dt))

CDS Pricing (bootstrapping lambda)
------------------------------------
A CDS pays LGD on default; the protection buyer pays a spread s.

Fee leg PV   = s * sum_i delta_i * P(0,t_i) * Q(0,t_i)
Protect. leg = LGD * sum_i P(0,t_i) * PD_i

At par: s = LGD * [sum_i P_i * PD_i] / [sum_i delta_i * P_i * Q_i]
    => CDS spread s = LGD * lambda  (approx. for flat curves)

Merton Structural Model
------------------------
Firm value V_t follows GBM. Default at T if V_T < D (face value of debt).

    PD_Merton = N(-d2)
    d2 = [ln(V0/D) + (mu - sigma_V^2/2)*T] / (sigma_V * sqrt(T))

Under risk-neutral measure (replace mu with r):
    PD_RN = N(-d2^Q)    (risk-neutral / market-implied PD)

Credit Valuation Adjustment (CVA)
----------------------------------
CVA corrects derivative prices for counterparty default risk.
For a unilateral CVA (own default ignored):

    CVA = (1-R) * sum_i P(0,T_i) * EPE(T_i) * PD_i

where EPE(T_i) = E[max(V_i, 0)] is the Expected Positive Exposure.

For a swap: EPE profile is computed via Monte Carlo of the
underlying risk factor (interest rates), then convolved with the
default probability term structure.

References
----------
- Merton, R.C. (1974). On the pricing of corporate debt.
  Journal of Finance, 29(2), 449-470.
- Duffie, D., Singleton, K. (1999). Modeling term structures of
  defaultable bonds. Review of Financial Studies, 12(4), 687-720.
- Gregory, J. (2015). The xVA Challenge. 3rd ed. Wiley.
- Hull, J.C. (2022). Options, Futures, and Other Derivatives. Ch. 24-25.
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
# SECTION 1 — REDUCED-FORM MODEL
# ============================================================

def survival_prob(lam, T):
    """Q(0,T) = exp(-lambda*T) for constant hazard rate."""
    return np.exp(-lam * T)

def marginal_pd(lam, t_prev, t_curr):
    """PD in [t_prev, t_curr] = Q(t_prev) - Q(t_curr)."""
    return survival_prob(lam, t_prev) - survival_prob(lam, t_curr)

def cds_par_spread(lam, R, tenor, freq=4, r=0.04):
    """
    CDS par spread: s = LGD * protection_leg_pv / fee_annuity
    freq: payment frequency per year (4 = quarterly)
    """
    LGD   = 1 - R
    times = np.arange(1/freq, tenor + 1e-9, 1/freq)
    delta = 1 / freq
    P     = np.exp(-r * times)        # risk-free discount factors
    Q     = survival_prob(lam, times) # survival probs

    # Protection leg: pays LGD at each interval if default occurs
    t_prev = np.concatenate([[0], times[:-1]])
    PD_i   = marginal_pd(lam, t_prev, times)
    prot_leg = LGD * np.sum(P * PD_i)

    # Fee leg: pays s * delta while alive
    fee_annuity = np.sum(delta * P * Q)
    return prot_leg / fee_annuity

def bootstrap_hazard(cds_spread, R, tenor, freq=4, r=0.04):
    """Back out constant hazard rate from observed CDS spread."""
    def objective(lam):
        return cds_par_spread(lam, R, tenor, freq, r) - cds_spread
    return brentq(objective, 1e-6, 5.0)

# ============================================================
# SECTION 2 — MERTON STRUCTURAL MODEL
# ============================================================

def merton_pd(V0, D, mu, sigma_V, T, risk_neutral=False, r=0.04):
    """
    Merton (1974) default probability.
    risk_neutral=True: replace mu with r (market-implied PD).
    """
    drift = r if risk_neutral else mu
    d2 = (np.log(V0/D) + (drift - 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    return norm.cdf(-d2), d2

def merton_equity_value(V0, D, r, sigma_V, T):
    """
    Equity = call on firm value: E = V0*N(d1) - D*exp(-r*T)*N(d2)
    """
    d1 = (np.log(V0/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V0*norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2)

def merton_debt_value(V0, D, r, sigma_V, T):
    """Debt PV = V0 - Equity (put-call parity on firm value)."""
    E = merton_equity_value(V0, D, r, sigma_V, T)
    return V0 - E

# ============================================================
# SECTION 3 — CVA COMPUTATION
# ============================================================

def compute_epe_swap(R_fixed, T_swap, n_steps, n_paths,
                     r0, kappa, theta, sigma_r, seed=42):
    """
    Expected Positive Exposure profile for a receive-fixed IRS.
    Simulate interest rate paths (Vasicek), compute swap MTM at each
    time step, take the positive part, and average.
    Returns (times, EPE array).
    """
    rng    = np.random.default_rng(seed)
    dt     = T_swap / n_steps
    times  = np.linspace(0, T_swap, n_steps + 1)
    r_paths = np.zeros((n_paths, n_steps + 1))
    r_paths[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        r = r_paths[:, i]
        r_paths[:, i+1] = r + kappa*(theta-r)*dt + sigma_r*sqrt_dt*Z[:, i]

    # At each future time t_k, compute residual swap value
    EPE = np.zeros(n_steps + 1)
    for k, t_k in enumerate(times):
        tau_rem = T_swap - t_k
        if tau_rem < 1e-6:
            EPE[k] = 0.0
            continue
        # Vasicek annuity factor at each path's r_k
        r_k = r_paths[:, k]
        # Approximate annuity using closed-form Vasicek bond prices
        # sum delta_i * P(t_k, t_i) for remaining payment dates
        pay_times = np.arange(0.5, tau_rem + 1e-9, 0.5)  # semi-annual
        if len(pay_times) == 0:
            EPE[k] = 0.0
            continue
        # Vasicek: B(tau) = (1-exp(-k*tau))/k
        B_arr = (1 - np.exp(-kappa * pay_times)) / kappa
        lnA   = ((theta - sigma_r**2/(2*kappa**2))*(B_arr - pay_times)
                 - sigma_r**2*B_arr**2/(4*kappa))
        # P(t_k, t_i) for each path: exp(lnA - B*r_k)
        # r_k shape (n_paths,), B_arr shape (n_pay,)
        P_rem = np.exp(lnA[np.newaxis,:] - B_arr[np.newaxis,:]*r_k[:,np.newaxis])
        annuity_k = np.sum(0.5 * P_rem, axis=1)  # (n_paths,)
        P_T   = np.exp(lnA[-1] - B_arr[-1]*r_k)  # P(t_k, T_swap)
        R_par_k = (1 - P_T) / annuity_k
        mtm_k   = (R_fixed - R_par_k) * annuity_k  # per unit notional
        EPE[k]  = np.mean(np.maximum(mtm_k, 0))

    return times, EPE

def compute_cva(EPE, times, lam, R, r=0.04):
    """
    CVA = LGD * sum_i P(0,t_i) * EPE(t_i) * PD_i
    """
    LGD    = 1 - R
    dt_arr = np.diff(times, prepend=0)
    P_arr  = np.exp(-r * times)
    t_prev = np.concatenate([[0], times[:-1]])
    PD_arr = marginal_pd(lam, t_prev, times)
    return LGD * np.sum(P_arr * EPE * PD_arr)

# ============================================================
# DIAGNOSTICS
# ============================================================
lam_base = 0.02   # 2% annual hazard rate
R_rec    = 0.40   # 40% recovery
r_rf     = 0.04   # 4% risk-free rate

# CDS spreads from hazard rates
tenors_cds = [1, 3, 5, 7, 10]
spreads_bps = [cds_par_spread(lam_base, R_rec, T_, r=r_rf)*1e4
               for T_ in tenors_cds]
print("[M25] Flat hazard rate model (lambda=2%)")
for T_, s_ in zip(tenors_cds, spreads_bps):
    surv = survival_prob(lam_base, T_) * 100
    print(f"      {T_:2d}Y CDS = {s_:6.2f}bps  |  Survival prob = {surv:.2f}%")

# Merton model
V0 = 100; D = 70; mu = 0.08; sigma_V = 0.25; T_merton = 1.0
pd_phys, d2_phys = merton_pd(V0, D, mu, sigma_V, T_merton, risk_neutral=False, r=r_rf)
pd_rn,   d2_rn   = merton_pd(V0, D, mu, sigma_V, T_merton, risk_neutral=True,  r=r_rf)
print(f"\n[M25] Merton Model: V0={V0}, D={D}, sigma={sigma_V}, T={T_merton}Y")
print(f"      Physical PD  = {pd_phys*100:.4f}%  (d2={d2_phys:.4f})")
print(f"      Risk-neutral PD = {pd_rn*100:.4f}%  (d2={d2_rn:.4f})")

# CVA
print("\n[M25] Computing EPE profile via Monte Carlo (2000 paths) ...")
R_fixed_swap = 0.048
times_epe, EPE = compute_epe_swap(
    R_fixed_swap, T_swap=5.0, n_steps=60, n_paths=2000,
    r0=0.05, kappa=0.3, theta=0.045, sigma_r=0.015
)
cva_val = compute_cva(EPE, times_epe, lam_base, R_rec, r=r_rf)
print(f"      CVA (N=1, lambda=2%, R=40%) = {cva_val*100:.4f}% of notional")

# ============================================================
# FIGURE 1 — Survival curve, hazard rate, CDS spread
# ============================================================
t0 = time.perf_counter()
print("[M25] Figure 1: Survival curves and CDS spreads ...")

T_arr  = np.linspace(0.01, 10, 300)
hazard_scenarios = [
    ("lambda=1%  (AAA/AA)", 0.010, BLUE),
    ("lambda=2%  (A/BBB)",  0.020, GREEN),
    ("lambda=5%  (BB)",     0.050, ORANGE),
    ("lambda=10% (B/CCC)",  0.100, RED),
    ("lambda=20% (distressed)", 0.200, PURPLE),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M25 — Credit Risk: Reduced-Form Model\nSurvival Curves | CDS Spreads | Marginal PD",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Survival probability
ax = axes[0]
for label, lam_, col in hazard_scenarios:
    ax.plot(T_arr, survival_prob(lam_, T_arr)*100, color=col, lw=2, label=label)
ax.set_xlabel("Horizon T (years)"); ax.set_ylabel("Survival probability (%)")
ax.set_title("Q(0,T) = exp(-lambda*T)\nCredit quality deterioration",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) CDS par spread vs tenor
ax = axes[1]
for label, lam_, col in hazard_scenarios:
    s_arr = [cds_par_spread(lam_, R_rec, T_, r=r_rf)*1e4 for T_ in tenors_cds]
    ax.plot(tenors_cds, s_arr, color=col, lw=2, marker="o", ms=5, label=label)
ax.set_xlabel("Tenor (years)"); ax.set_ylabel("CDS spread (bps)")
ax.set_title(f"CDS Par Spread  (R={R_rec*100:.0f}%)\ns ~ LGD * lambda  (approx.)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Marginal PD term structure (annual buckets)
ax = axes[2]
buckets = np.arange(0, 10)
for label, lam_, col in hazard_scenarios[1:4]:   # A/BBB, BB, B
    pd_marg = [marginal_pd(lam_, t, t+1)*100 for t in buckets]
    ax.bar(buckets + 0.5 + hazard_scenarios[1:4].index((label,lam_,col))*0.25,
           pd_marg, width=0.22, color=col, alpha=0.8, label=label)
ax.set_xlabel("Year"); ax.set_ylabel("Marginal PD (%)")
ax.set_title("Annual Marginal Default Probability\nQ(t-1) - Q(t)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m25_01_survival_cds.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Merton structural model
# ============================================================
t0 = time.perf_counter()
print("[M25] Figure 2: Merton structural model ...")

V0_arr    = np.linspace(50, 200, 300)
sigma_arr = np.linspace(0.05, 0.60, 300)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M25 — Merton Structural Model\n"
             "PD = N(-d2)  |  Default when V_T < D",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) PD vs leverage (V0/D)
ax = axes[0]
for T_, col in [(0.5,BLUE),(1.0,GREEN),(2.0,ORANGE),(5.0,RED)]:
    pd_arr = [merton_pd(v, D, mu, sigma_V, T_, risk_neutral=True, r=r_rf)[0]*100
              for v in V0_arr]
    ax.plot(V0_arr/D, pd_arr, color=col, lw=2, label=f"T={T_}Y")
ax.axvline(1.0, color=WHITE, lw=1, linestyle=":", alpha=0.6, label="V0=D (critical)")
ax.set_xlabel("Leverage ratio V0/D"); ax.set_ylabel("PD (%)")
ax.set_title("Merton PD vs Leverage\nD=70, sigma=25%, mu=r",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) PD vs asset volatility
ax = axes[1]
for T_, col in [(0.5,BLUE),(1.0,GREEN),(2.0,ORANGE),(5.0,RED)]:
    pd_arr = [merton_pd(V0, D, mu, s, T_, risk_neutral=True, r=r_rf)[0]*100
              for s in sigma_arr]
    ax.plot(sigma_arr*100, pd_arr, color=col, lw=2, label=f"T={T_}Y")
ax.set_xlabel("Asset volatility sigma_V (%)"); ax.set_ylabel("PD (%)")
ax.set_title("Merton PD vs Asset Volatility\nV0=100, D=70, mu=r",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Physical vs risk-neutral PD (credit risk premium)
ax = axes[2]
mu_arr = np.linspace(0.00, 0.20, 200)
pd_phys_arr = [merton_pd(V0, D, m, sigma_V, 1.0, risk_neutral=False, r=r_rf)[0]*100
               for m in mu_arr]
pd_rn_arr   = [merton_pd(V0, D, m, sigma_V, 1.0, risk_neutral=True,  r=r_rf)[0]*100
               for m in mu_arr]
ax.plot(mu_arr*100, pd_phys_arr, color=ORANGE, lw=2.5, label="Physical PD (uses mu)")
ax.axhline(pd_rn_arr[0], color=BLUE, lw=2, linestyle="--",
           label=f"Risk-neutral PD = {pd_rn_arr[0]:.2f}% (fixed, uses r)")
ax.fill_between(mu_arr*100, pd_phys_arr, pd_rn_arr[0],
                where=np.array(pd_phys_arr) < pd_rn_arr[0],
                color=GREEN, alpha=0.15, label="Credit risk premium region")
ax.set_xlabel("Physical drift mu (%)"); ax.set_ylabel("PD (%)")
ax.set_title("Physical vs Risk-Neutral PD\n"
             "Market spreads > actuarial PD => credit risk premium",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m25_02_merton.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — CVA: EPE profile and CVA decomposition
# ============================================================
t0 = time.perf_counter()
print("[M25] Figure 3: CVA — EPE profile and sensitivity ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M25 — Credit Valuation Adjustment (CVA)\n"
             "CVA = LGD * sum P(t_i) * EPE(t_i) * PD_i",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) EPE profile
ax = axes[0]
ax.fill_between(times_epe, EPE*100, color=BLUE, alpha=0.35, label="EPE profile")
ax.plot(times_epe, EPE*100, color=BLUE, lw=2)
ax.set_xlabel("Time (years)"); ax.set_ylabel("EPE (% of notional)")
ax.set_title(f"Expected Positive Exposure — Receive-Fixed IRS\n"
             f"R_fixed={R_fixed_swap*100:.2f}%, T=5Y  (Vasicek rates, 2000 paths)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) CVA sensitivity to CDS spread (hazard rate)
lam_arr = np.linspace(0.005, 0.10, 60)
cva_arr = [compute_cva(EPE, times_epe, lam_, R_rec, r=r_rf)*1e4
           for lam_ in lam_arr]
# Approx CDS spread: s ~ LGD * lambda
cds_approx = lam_arr * (1-R_rec) * 1e4

ax = axes[1]
ax.plot(cds_approx, cva_arr, color=RED, lw=2.5, label="CVA (bps of notional)")
ax.fill_between(cds_approx, cva_arr, color=RED, alpha=0.15)
ax.axvline(lam_base*(1-R_rec)*1e4, color=ORANGE, lw=1.5, linestyle="--",
           label=f"Base case: {lam_base*(1-R_rec)*1e4:.0f}bps")
ax.set_xlabel("CDS spread (bps)"); ax.set_ylabel("CVA (bps of notional)")
ax.set_title("CVA vs Counterparty Credit Quality\nHigher spread => higher CVA cost",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) CVA sensitivity to recovery rate
R_arr   = np.linspace(0.0, 0.9, 60)
cva_R   = [compute_cva(EPE, times_epe, lam_base, Rv, r=r_rf)*1e4 for Rv in R_arr]

ax = axes[2]
ax.plot(R_arr*100, cva_R, color=PURPLE, lw=2.5, label="CVA vs Recovery")
ax.fill_between(R_arr*100, cva_R, color=PURPLE, alpha=0.15)
ax.axvline(R_rec*100, color=YELLOW, lw=1.5, linestyle="--",
           label=f"Base R={R_rec*100:.0f}%")
ax.set_xlabel("Recovery rate R (%)"); ax.set_ylabel("CVA (bps of notional)")
ax.set_title("CVA vs Recovery Rate\nCVA = (1-R)*...  => linear in LGD",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m25_03_cva.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 25 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] EL = PD * LGD * EAD  (expected credit loss)")
print("  [2] Reduced form: Q(0,T) = exp(-lambda*T)  (survival prob)")
print("  [3] CDS spread: s = LGD * lambda  (flat curve approximation)")
print("  [4] Merton: PD = N(-d2)  defaults when V_T < D (debt face)")
print("  [5] Physical PD < Risk-neutral PD  (credit risk premium)")
print("  [6] CVA = LGD * sum P(t)*EPE(t)*PD(t)  (counterparty adj.)")
print(f"  lambda=2%: 5Y survival = {survival_prob(0.02,5)*100:.2f}%  "
      f"| 5Y CDS ~ {cds_par_spread(0.02, R_rec, 5, r=r_rf)*1e4:.1f}bps")
print(f"  Merton PD (RN): {pd_rn*100:.4f}%  |  Physical: {pd_phys*100:.4f}%")
print(f"  CVA (N=1): {cva_val*1e4:.4f}bps of notional")
print("=" * 65)

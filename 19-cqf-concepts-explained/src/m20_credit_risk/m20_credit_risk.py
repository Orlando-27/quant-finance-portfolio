#!/usr/bin/env python3
"""
M20 — Credit Risk: PD, LGD, EAD, CVA
=======================================
Module 5 of 9 | CQF Concepts Explained

Theory
------
Expected Loss (EL) framework:
    EL = PD * LGD * EAD

    PD  : Probability of Default
    LGD : Loss Given Default (1 - Recovery Rate)
    EAD : Exposure at Default

Merton (1974) structural model:
    Firm value V_t follows GBM:
        dV_t = mu_V * V_t * dt + sigma_V * V_t * dW_t

    Default occurs at maturity T if V_T < D (face value of debt).

    PD = N(-d2)  where d2 = (ln(V0/D) + (r - sigma_V^2/2)*T) / (sigma_V*sqrt(T))

    Equity = call option on firm value: E = V*N(d1) - D*e^{-rT}*N(d2)
    Debt   = D*e^{-rT} - put(V, D) = V - E  (Modigliani-Miller)

    Credit spread = -ln(1 - PD*LGD) / T

Hazard rate (reduced-form) model:
    Default intensity lambda_t (hazard rate)
    P(tau > t) = exp(-integral_0^t lambda_s ds)
    For constant lambda: P(tau > t) = e^{-lambda*t}
    PD(0,T) = 1 - e^{-lambda*T}

Credit Valuation Adjustment (CVA):
    CVA = (1-R) * integral_0^T EE(t) * dPD(0,t)
        approx (1-R) * sum_i EE(t_i) * PD(t_{i-1}, t_i)

    where:
    R    : recovery rate
    EE(t): Expected Exposure at time t (risk-neutral)
    PD(t_{i-1}, t_i): marginal default probability in period [t_{i-1}, t_i]

    CVA is the market value of counterparty credit risk.

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
from scipy.optimize import brentq

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
# Merton model
# ---------------------------------------------------------------------------
def merton_d1d2(V, D, T, r, sigma_V):
    d1 = (np.log(V/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    return d1, d2

def merton_pd(V, D, T, r, sigma_V):
    """Risk-neutral PD = N(-d2)."""
    _, d2 = merton_d1d2(V, D, T, r, sigma_V)
    return norm.cdf(-d2)

def merton_equity(V, D, T, r, sigma_V):
    """Equity = call on firm value."""
    d1, d2 = merton_d1d2(V, D, T, r, sigma_V)
    return V*norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2)

def merton_credit_spread(V, D, T, r, sigma_V, recovery=0.40):
    pd  = merton_pd(V, D, T, r, sigma_V)
    lgd = 1 - recovery
    cs  = -np.log(1 - pd*lgd) / T
    return cs

def merton_calibrate(E0, sigma_E, D, T, r):
    """
    Calibrate Merton V0 and sigma_V from observable equity E0, sigma_E.
    Uses the system:
        E0 = V0*N(d1) - D*e^{-rT}*N(d2)
        sigma_E * E0 = V0 * sigma_V * N(d1)
    """
    def equations(params):
        V, sv = params
        if V <= 0 or sv <= 0: return [1e6, 1e6]
        d1, d2 = merton_d1d2(V, D, T, r, sv)
        eq1 = merton_equity(V, D, T, r, sv) - E0
        eq2 = V*sv*norm.cdf(d1) - sigma_E*E0
        return [eq1, eq2]

    from scipy.optimize import fsolve
    V0_init = E0 + D*np.exp(-r*T)
    sv_init = sigma_E * E0 / V0_init
    sol = fsolve(equations, [V0_init, sv_init], full_output=True)
    V0, sigma_V = sol[0]
    return abs(V0), abs(sigma_V)

# ---------------------------------------------------------------------------
# Hazard rate / reduced-form
# ---------------------------------------------------------------------------
def hazard_survival(lam, t):
    return np.exp(-lam * t)

def hazard_pd_cumulative(lam, t):
    return 1 - hazard_survival(lam, t)

def hazard_marginal_pd(lam, t1, t2):
    return hazard_survival(lam, t1) - hazard_survival(lam, t2)

# ---------------------------------------------------------------------------
# CVA calculation
# ---------------------------------------------------------------------------
def compute_cva(ee_profile, t_grid, lam, recovery, r):
    """
    CVA = (1-R) * sum_i EE(t_i) * PD(t_{i-1}, t_i) * disc(t_i)
    """
    lgd = 1 - recovery
    cva = 0.0
    for i in range(1, len(t_grid)):
        t1   = t_grid[i-1]
        t2   = t_grid[i]
        ee   = ee_profile[i]
        mpd  = hazard_marginal_pd(lam, t1, t2)
        disc = np.exp(-r * t2)
        cva += lgd * ee * mpd * disc
    return cva

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Merton
V0      = 100.0   # firm value
D       = 70.0    # face value of debt
T_mert  = 5.0     # debt maturity
r       = 0.05
sigma_V = 0.25    # firm vol
RR      = 0.40    # recovery rate

# Hazard
lambda_ = 0.02    # constant hazard rate (2% per year)

# CVA parameters
T_cva    = 5.0
N_CVA    = 20          # time steps for CVA
t_cva    = np.linspace(0, T_cva, N_CVA+1)
NOTIONAL = 10_000_000  # USD 10M IRS

pd_mert  = merton_pd(V0, D, T_mert, r, sigma_V)
cs_mert  = merton_credit_spread(V0, D, T_mert, r, sigma_V, RR)

print(f"[M20] Merton: V0={V0}, D={D}, T={T_mert}Y, sigma_V={sigma_V:.0%}")
print(f"      PD(Merton) = {pd_mert:.4%}  |  Credit spread = {cs_mert*1e4:.2f} bps")
print(f"      Hazard rate lambda = {lambda_:.2%}  |  "
      f"5Y PD = {hazard_pd_cumulative(lambda_, T_mert):.4%}")


# ===========================================================================
# FIGURE 1 — Merton Model: firm value, PD, credit spread
# ===========================================================================
print("[M20] Figure 1: Merton structural model ...")
t0 = time.perf_counter()

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
fig.suptitle(
    "M20 — Merton Structural Model: Firm Value, Default Probability, Credit Spread\n"
    f"V0={V0}, D={D}, T={T_mert}Y, r={r:.0%}, sigma_V={sigma_V:.0%}, Recovery={RR:.0%}",
    color=WHITE, fontsize=10
)

# (0,0) Firm value paths
rng = np.random.default_rng(42)
N_PATHS = 1000; N_STEPS = 252*int(T_mert)
dt_m = T_mert / N_STEPS
Z_m  = rng.standard_normal((N_PATHS, N_STEPS))
log_V = np.cumsum((r - 0.5*sigma_V**2)*dt_m
                  + sigma_V*np.sqrt(dt_m)*Z_m, axis=1)
V_paths = V0 * np.exp(np.hstack([np.zeros((N_PATHS,1)), log_V]))
t_m = np.linspace(0, T_mert, N_STEPS+1)
defaulted = V_paths[:, -1] < D

ax = axes[0, 0]
for k in range(min(80, N_PATHS)):
    col = RED if defaulted[k] else BLUE
    ax.plot(t_m, V_paths[k], color=col, alpha=0.08, lw=0.6)
ax.axhline(D, color=YELLOW, lw=2.5, linestyle="--",
           label=f"Debt face value D={D}")
ax.plot(t_m, V_paths.mean(axis=0), color=WHITE, lw=2, label="Mean V_t")
n_def = defaulted.sum()
ax.set_xlabel("Time (years)"); ax.set_ylabel("Firm Value V_t")
ax.set_title(f"Firm Value Paths\n"
             f"Defaults (V_T<D): {n_def}/{N_PATHS} = {n_def/N_PATHS:.2%}  "
             f"(Merton PD={pd_mert:.2%})",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,1) PD vs leverage ratio V0/D and sigma_V
ax = axes[0, 1]
lev_range  = np.linspace(0.5, 3.0, 200)
V_range    = lev_range * D
sigmas_pd  = [0.10, 0.20, 0.30, 0.40]
colors_sig = [BLUE, GREEN, YELLOW, ORANGE]
for sv, col in zip(sigmas_pd, colors_sig):
    pd_arr = [merton_pd(v, D, T_mert, r, sv) for v in V_range]
    ax.plot(lev_range, np.array(pd_arr)*100, color=col, lw=2,
            label=f"sigma_V={sv:.0%}")
ax.axvline(V0/D, color=WHITE, lw=1.5, linestyle=":",
           label=f"Current V/D={V0/D:.2f}")
ax.set_xlabel("Coverage ratio V0/D"); ax.set_ylabel("PD (%)")
ax.set_title("Merton PD vs Coverage Ratio V/D\n"
             "Higher vol or lower coverage => higher PD",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,2) Credit spread vs maturity
ax = axes[0, 2]
T_range = np.linspace(0.25, 10, 200)
for sv, col in zip(sigmas_pd, colors_sig):
    cs_arr = [merton_credit_spread(V0, D, t, r, sv, RR)*1e4 for t in T_range]
    ax.plot(T_range, cs_arr, color=col, lw=2, label=f"sigma_V={sv:.0%}")
ax.set_xlabel("Maturity T (years)"); ax.set_ylabel("Credit Spread (bps)")
ax.set_title("Merton Credit Spread Term Structure\n"
             "Hump-shape for IG; monotone for HY",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,0) Hazard rate survival function
ax = axes[1, 0]
T_haz = np.linspace(0, 10, 300)
lambdas = [0.005, 0.01, 0.02, 0.05]
cols_h  = [BLUE, GREEN, YELLOW, RED]
for lam, col in zip(lambdas, cols_h):
    ax.plot(T_haz, hazard_survival(lam, T_haz)*100, color=col, lw=2,
            label=f"lambda={lam:.1%}  (5Y PD={hazard_pd_cumulative(lam,5):.2%})")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Survival probability (%)")
ax.set_title("Hazard Rate Model: Survival Function\n"
             "P(tau > t) = exp(-lambda*t)",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,1) Merton calibration from equity
ax = axes[1, 1]
E0_range    = np.linspace(20, 60, 30)
sigma_E_fix = 0.40
V0_cal = []; sv_cal = []; pd_cal = []
for e0 in E0_range:
    try:
        v0c, svc = merton_calibrate(e0, sigma_E_fix, D, T_mert, r)
        pd_c = merton_pd(v0c, D, T_mert, r, svc)
        V0_cal.append(v0c); sv_cal.append(svc); pd_cal.append(pd_c)
    except Exception:
        V0_cal.append(np.nan); sv_cal.append(np.nan); pd_cal.append(np.nan)

ax2 = ax.twinx()
ax.plot(E0_range, np.array(pd_cal)*100, color=RED,    lw=2.5,
        label="Calibrated PD (%)")
ax2.plot(E0_range, np.array(sv_cal)*100, color=ORANGE, lw=2,
         linestyle="--", label="Calibrated sigma_V (%)")
ax.set_xlabel("Observed Equity E0"); ax.set_ylabel("Calibrated PD (%)", color=RED)
ax2.set_ylabel("Calibrated sigma_V (%)", color=ORANGE)
ax2.tick_params(colors=ORANGE)
ax.set_title(f"Merton Calibration from Equity\n"
             f"sigma_E={sigma_E_fix:.0%}, D={D}",
             color=WHITE, fontsize=8)
ax.legend(loc="upper right", fontsize=7)
ax2.legend(loc="upper left", fontsize=7)
ax.grid(True); watermark(ax)

# (1,2) Summary table
ax = axes[1, 2]
ax.axis("off")
rows = [
    ["Metric",           "Merton",              "Hazard (lambda=2%)"],
    ["5Y PD",            f"{pd_mert:.4%}",       f"{hazard_pd_cumulative(lambda_,5):.4%}"],
    ["1Y PD",            f"{merton_pd(V0,D,1,r,sigma_V):.4%}",
                          f"{hazard_pd_cumulative(lambda_,1):.4%}"],
    ["Credit spread",    f"{cs_mert*1e4:.2f} bps", f"{lambda_*(1-RR)*1e4:.2f} bps"],
    ["V0 / D",           f"{V0/D:.3f}",          "N/A"],
    ["sigma_V",          f"{sigma_V:.0%}",        "N/A"],
    ["Recovery",         f"{RR:.0%}",             f"{RR:.0%}"],
    ["EL = PD*LGD",      f"{pd_mert*(1-RR):.4%}", f"{hazard_pd_cumulative(lambda_,5)*(1-RR):.4%}"],
]
cols_row = [[PANEL]*3]+[[DARK if i%2==0 else PANEL]*3 for i in range(len(rows)-1)]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=cols_row)
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.50)
for (ri, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=YELLOW if ri==0 else WHITE,
                        weight="bold" if ri==0 else "normal")
ax.set_title("Merton vs Hazard Rate Summary", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m20_01_merton_model.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — CVA: Expected Exposure profile + CVA term structure
# ===========================================================================
print("[M20] Figure 2: CVA calculation ...")
t0 = time.perf_counter()

# Simulate IRS Expected Exposure profile (interest rate swap)
# EE(t) approximated by ATM swaption price profile for par swap
# Use simplified GBM on swap MTM
r_swap   = 0.05
sigma_r  = 0.015     # rate vol
K_swap   = r_swap    # at-the-money
N_IRS    = 5000
t_steps  = np.linspace(0, T_cva, N_CVA+1)
dt_cva   = T_cva / N_CVA

rng_cva = np.random.default_rng(42)
Z_cva   = rng_cva.standard_normal((N_IRS, N_CVA))

# Simulate rate paths (Vasicek-like)
kappa_r = 0.5; theta_r = 0.05; sigma_rr = 0.015
rates   = np.zeros((N_IRS, N_CVA+1))
rates[:, 0] = r_swap
for i in range(N_CVA):
    rates[:, i+1] = (rates[:, i]
                     + kappa_r*(theta_r - rates[:, i])*dt_cva
                     + sigma_rr*np.sqrt(dt_cva)*Z_cva[:, i])
    rates[:, i+1] = np.maximum(rates[:, i+1], 0.0)

# Swap MTM at each time: simplified as (rate - K) * annuity
# Annuity = sum_j disc(t_j - t_i) for remaining cashflows
def annuity(rate, t_start, T_end, freq=0.5):
    times = np.arange(t_start + freq, T_end + 1e-9, freq)
    if len(times) == 0: return 0.0
    return np.sum(freq * np.exp(-rate * (times - t_start)))

EE_profile  = np.zeros(N_CVA+1)
PFE_profile = np.zeros(N_CVA+1)   # 97.5th percentile (Potential Future Exposure)

for i, t in enumerate(t_steps):
    if t >= T_cva - 1e-9:
        EE_profile[i]  = 0.0
        PFE_profile[i] = 0.0
        continue
    mtm_i = np.array([
        (rates[k, i] - K_swap) * annuity(rates[k, i], t, T_cva)
        for k in range(N_IRS)
    ]) * NOTIONAL
    EE_i  = np.maximum(mtm_i, 0).mean()       # expected positive exposure
    PFE_i = np.percentile(np.maximum(mtm_i, 0), 97.5)
    EE_profile[i]  = EE_i
    PFE_profile[i] = PFE_i

# CVA for different hazard rates
lambda_vals  = [0.005, 0.01, 0.02, 0.05]
colors_lam   = [BLUE, GREEN, YELLOW, RED]
cva_by_lam   = [compute_cva(EE_profile, t_steps, lam, RR, r) for lam in lambda_vals]

# CVA main calculation
cva_main = compute_cva(EE_profile, t_steps, lambda_, RR, r)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M20 — CVA: Expected Exposure and Credit Valuation Adjustment\n"
    f"IRS Notional=USD {NOTIONAL:,}  |  T={T_cva}Y  |  lambda={lambda_:.1%}  "
    f"|  Recovery={RR:.0%}  |  CVA = USD {cva_main:,.0f}",
    color=WHITE, fontsize=10
)

# (0) Exposure profile
ax = axes[0]
ax.plot(t_steps, EE_profile/1e3,  color=BLUE,   lw=2.5,
        label="EE (Expected Exposure)")
ax.plot(t_steps, PFE_profile/1e3, color=ORANGE, lw=2, linestyle="--",
        label="PFE 97.5th pct")
ax.fill_between(t_steps, 0, EE_profile/1e3, color=BLUE, alpha=0.15)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Exposure (USD thousands)")
ax.set_title(f"IRS Exposure Profile\n"
             f"Peak EE = USD {EE_profile.max():,.0f}  "
             f"|  Peak PFE = USD {PFE_profile.max():,.0f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (1) CVA term structure (cumulative)
ax = axes[1]
cva_cumul = np.zeros(N_CVA+1)
for i in range(1, N_CVA+1):
    cva_cumul[i] = compute_cva(EE_profile[:i+1], t_steps[:i+1], lambda_, RR, r)

ax.plot(t_steps, cva_cumul/1e3, color=RED, lw=2.5,
        label="Cumulative CVA")
ax.fill_between(t_steps, 0, cva_cumul/1e3, color=RED, alpha=0.15)
ax.axhline(cva_main/1e3, color=YELLOW, lw=1.5, linestyle="--",
           label=f"Total CVA = USD {cva_main:,.0f}")
ax.set_xlabel("Time (years)"); ax.set_ylabel("CVA (USD thousands)")
ax.set_title("Cumulative CVA Term Structure\n"
             "CVA = (1-R) * sum EE(t) * dPD(t) * disc(t)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (2) CVA sensitivity to hazard rate
ax = axes[2]
ax.bar([f"{lam:.1%}" for lam in lambda_vals],
       [cva/1e3 for cva in cva_by_lam],
       color=colors_lam, alpha=0.75, edgecolor=GREY)
for i, (lam, cva) in enumerate(zip(lambda_vals, cva_by_lam)):
    ax.text(i, cva/1e3 + 0.5, f"USD {cva:,.0f}", ha="center",
            fontsize=8, color=WHITE)
ax.set_xlabel("Hazard Rate lambda"); ax.set_ylabel("CVA (USD thousands)")
ax.set_title("CVA Sensitivity to Hazard Rate\n"
             "CVA ~ linear in PD for small spreads",
             color=WHITE, fontsize=9)
ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m20_02_cva.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Portfolio credit risk: EL, UL, Vasicek model
# ===========================================================================
print("[M20] Figure 3: Portfolio credit risk ...")
t0 = time.perf_counter()

# Vasicek single-factor model for portfolio loss distribution
# Loan portfolio: N=100 homogeneous loans
N_LOANS  = 100
PD_LOAN  = 0.02     # 2% annual PD per loan
LGD_LOAN = 0.45     # 45% LGD
EAD_LOAN = 100_000  # USD 100k per loan
RHO_LOAN = 0.15     # asset correlation
N_SIM    = 100_000

rng_p = np.random.default_rng(42)
Z_sys  = rng_p.standard_normal(N_SIM)     # systematic factor
Z_idio = rng_p.standard_normal((N_SIM, N_LOANS))  # idiosyncratic

# Asset returns for each loan in each scenario
A = (np.sqrt(RHO_LOAN) * Z_sys[:, None]
     + np.sqrt(1-RHO_LOAN) * Z_idio)     # (N_SIM, N_LOANS)

# Default threshold
threshold = norm.ppf(PD_LOAN)
defaults  = (A < threshold).astype(float)

# Portfolio loss rate
loss_rate = (defaults * LGD_LOAN).mean(axis=1)

# Vasicek analytical WCDR at alpha
def vasicek_wcdr(pd, rho, alpha):
    """Worst-case default rate at confidence alpha."""
    return norm.cdf(
        (norm.ppf(pd) - np.sqrt(rho)*norm.ppf(alpha))
        / np.sqrt(1 - rho)
    )

# Vasicek PDF of loss rate
def vasicek_pdf(l, pd, rho, lgd=1.0):
    """PDF of portfolio loss rate l under Vasicek."""
    x = l / lgd
    if x <= 0 or x >= 1: return 0.0
    val = np.sqrt((1-rho)/rho) * np.exp(
        -0.5*((np.sqrt(1-rho)*norm.ppf(x) - norm.ppf(pd))**2 / rho
              - norm.ppf(x)**2)
    ) / norm.pdf(norm.ppf(x))
    return val / lgd

alphas_wcdr = [0.95, 0.99, 0.999]
wcdr_vals   = [vasicek_wcdr(PD_LOAN, RHO_LOAN, a) for a in alphas_wcdr]

EL_port = N_LOANS * PD_LOAN * LGD_LOAN * EAD_LOAN
UL_99   = np.percentile(loss_rate, 99) * N_LOANS * EAD_LOAN - EL_port

l_range = np.linspace(0.001, 0.15, 300)
pdf_vas = np.array([vasicek_pdf(l, PD_LOAN, RHO_LOAN, LGD_LOAN) for l in l_range])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M20 — Portfolio Credit Risk: Vasicek Model\n"
    f"N={N_LOANS} loans  |  PD={PD_LOAN:.0%}  |  LGD={LGD_LOAN:.0%}  "
    f"|  EAD=USD {EAD_LOAN:,}  |  rho={RHO_LOAN:.0%}",
    color=WHITE, fontsize=10
)

# (0) Portfolio loss distribution
ax = axes[0]
ax.hist(loss_rate*100, bins=80, density=True, color=BLUE, alpha=0.55,
        edgecolor="none", label=f"MC loss rate  (N={N_SIM:,})")
ax.plot(l_range*100, pdf_vas/100, color=YELLOW, lw=2.5,
        label="Vasicek analytical PDF")
el_rate = PD_LOAN * LGD_LOAN
ax.axvline(el_rate*100,              color=GREEN, lw=2, linestyle=":",
           label=f"EL = {el_rate:.2%}")
ax.axvline(np.percentile(loss_rate, 99)*100, color=ORANGE, lw=2, linestyle="--",
           label=f"99% VaR loss rate = {np.percentile(loss_rate,99):.2%}")
ax.axvline(np.percentile(loss_rate, 99.9)*100, color=RED, lw=2, linestyle="--",
           label=f"99.9% loss rate = {np.percentile(loss_rate,99.9):.2%}")
ax.set_xlabel("Portfolio Loss Rate (%)"); ax.set_ylabel("Density")
ax.set_title("Portfolio Loss Distribution\n"
             "Vasicek single-factor model",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1) WCDR vs confidence and rho
ax = axes[1]
alphas_range = np.linspace(0.80, 0.9999, 300)
rho_vals_w   = [0.05, 0.12, 0.20, 0.35]
colors_rho_w = [BLUE, GREEN, YELLOW, RED]
for rho_w, col in zip(rho_vals_w, colors_rho_w):
    wcdr_arr = [vasicek_wcdr(PD_LOAN, rho_w, a) for a in alphas_range]
    ax.plot(alphas_range*100, np.array(wcdr_arr)*100, color=col, lw=2,
            label=f"rho={rho_w:.0%}")
ax.axhline(PD_LOAN*100, color=WHITE, lw=1.5, linestyle=":",
           label=f"Unconditional PD={PD_LOAN:.0%}")
ax.set_xlabel("Confidence Level (%)"); ax.set_ylabel("WCDR (%)")
ax.set_title("Worst-Case Default Rate (WCDR)\n"
             "Basel II formula: WCDR(alpha, rho)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) EL / UL / EC across PD spectrum
ax = axes[2]
pd_range  = np.linspace(0.001, 0.20, 100)
el_arr    = pd_range * LGD_LOAN
wcdr_99   = np.array([vasicek_wcdr(pd, RHO_LOAN, 0.99) for pd in pd_range])
wcdr_999  = np.array([vasicek_wcdr(pd, RHO_LOAN, 0.999) for pd in pd_range])
ul_99_arr = (wcdr_99  - pd_range) * LGD_LOAN
ul_999_arr= (wcdr_999 - pd_range) * LGD_LOAN

ax.plot(pd_range*100, el_arr*100,    color=GREEN,  lw=2.5, label="EL = PD*LGD")
ax.plot(pd_range*100, ul_99_arr*100, color=YELLOW, lw=2.5, label="UL 99% (EC)")
ax.plot(pd_range*100, ul_999_arr*100,color=RED,    lw=2.5, label="UL 99.9% (EC)")
ax.fill_between(pd_range*100, el_arr*100, ul_99_arr*100,
                color=YELLOW, alpha=0.10)
ax.fill_between(pd_range*100, ul_99_arr*100, ul_999_arr*100,
                color=RED, alpha=0.10)
ax.axvline(PD_LOAN*100, color=WHITE, lw=1.5, linestyle=":",
           label=f"Current PD={PD_LOAN:.0%}")
ax.set_xlabel("PD (%)"); ax.set_ylabel("Loss Rate (%)")
ax.set_title("EL vs Economic Capital (UL)\n"
             "EC = WCDR*LGD - EL  (unexpected loss)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m20_03_portfolio_credit.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 20 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] EL = PD * LGD * EAD  (expected loss)")
print("  [2] Merton: default when V_T < D  (equity = call on firm)")
print("  [3] Hazard rate: P(tau>t) = exp(-lambda*t)")
print(f"  [4] CVA = (1-R)*sum EE(t)*dPD(t)*disc(t) = USD {cva_main:,.0f}")
print(f"  [5] Vasicek WCDR 99% = {vasicek_wcdr(PD_LOAN,RHO_LOAN,0.99):.4%}")
print(f"      Vasicek WCDR 99.9% = {vasicek_wcdr(PD_LOAN,RHO_LOAN,0.999):.4%}")
print(f"  [6] EL (portfolio) = USD {EL_port:,.0f}  |  UL 99% = USD {UL_99:,.0f}")
print(f"  [7] Merton PD = {pd_mert:.4%}  |  Credit spread = {cs_mert*1e4:.2f} bps")
print("=" * 65)

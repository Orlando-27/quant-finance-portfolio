#!/usr/bin/env python3
"""
M24 — Swap Pricing: IRS and Currency Swaps
===========================================
Module 24 | CQF Concepts Explained
Group 4   | Fixed Income & Interest Rates

Theory
------
Interest Rate Swap (IRS)
------------------------
A plain-vanilla IRS exchanges a fixed coupon stream for a floating
rate stream (typically SOFR or LIBOR-equivalent) on a notional N.

Fixed leg cash flows at t_1, ..., t_n:
    CF_fixed_i = N * R * delta_i

Floating leg cash flows (set-in-advance, paid-in-arrears):
    CF_float_i = N * L(t_{i-1}, t_i) * delta_i
    where L(t_{i-1}, t_i) is the spot LIBOR/SOFR for [t_{i-1}, t_i].

Pricing via discount factors P(0, t_i):
    PV(fixed)   = N * R * sum_i delta_i * P(0, t_i)
    PV(floating) = N * [P(0, t_0) - P(0, t_n)]  (telescoping identity)

The telescoping identity: the floating leg is worth par minus the PV
of the final notional repayment, regardless of the float rate path.

Par Swap Rate (at-the-money):
    R_par = [P(0, t_0) - P(0, t_n)] / [sum_i delta_i * P(0, t_i)]
    (numerator = annuity factor of floating leg value)

For a swap starting today, t_0 = 0, P(0,0) = 1:
    R_par = [1 - P(0, t_n)] / A(0, T)   where A = swap annuity factor

Mark-to-Market Value
--------------------
At time t > 0 for a receiver swap (receive fixed, pay floating):
    V_t = N * (R - R_t^par) * A_t
where R_t^par is the current par swap rate and A_t the current annuity.

DV01 (Dollar Value of 1bp)
--------------------------
    DV01 = |dV/dy| * 0.0001 = N * A * 0.0001  (for fixed-rate receiver)

Currency Swap
-------------
Exchange of fixed-rate cash flows in currency 1 vs fixed-rate cash
flows in currency 2, with notional exchange at inception and maturity.

Valued by discounting each leg in its own currency and converting
at the spot FX rate S0 (units of domestic per unit of foreign):
    V_domestic = PV(domestic leg) - S0 * PV(foreign leg)

References
----------
- Hull, J.C. (2022). Options, Futures, and Other Derivatives. 11th ed.
  Chapters 7-8.
- Brigo, D., Mercurio, F. (2006). Interest Rate Models. Springer.
  Chapter 1: The term structure of interest rates.
- Andersen, L., Piterbarg, V. (2010). Interest Rate Modeling. Vol. 1.
  Atlantic Financial Press.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# SECTION 1 — DISCOUNT CURVE (Nelson-Siegel parametric)
# ============================================================

def nelson_siegel_yields(tau, beta0, beta1, beta2, lam):
    """Nelson-Siegel spot yield curve: R(tau) in decimal."""
    x = tau / lam
    ex = np.exp(-x)
    return (beta0
            + beta1 * (1 - ex) / x
            + beta2 * ((1 - ex) / x - ex))

def discount_factor(tau, beta0, beta1, beta2, lam):
    """P(0, tau) = exp(-R(tau)*tau)."""
    R = nelson_siegel_yields(tau, beta0, beta1, beta2, lam)
    return np.exp(-R * tau)

# Representative USD yield curve parameters (circa 2024)
NS_USD = dict(beta0=0.048, beta1=-0.012, beta2=0.018, lam=2.5)
# Representative EUR yield curve
NS_EUR = dict(beta0=0.035, beta1=-0.008, beta2=0.015, lam=2.5)

# ============================================================
# SECTION 2 — IRS PRICING ENGINE
# ============================================================

def build_schedule(T, freq_per_year=2):
    """
    Build swap payment schedule.
    freq_per_year=2 -> semi-annual, =4 -> quarterly, =1 -> annual.
    Returns payment times and accrual fractions delta_i.
    """
    n = int(T * freq_per_year)
    times  = np.array([(i+1) / freq_per_year for i in range(n)])
    deltas = np.full(n, 1.0 / freq_per_year)
    return times, deltas

def par_swap_rate(T, freq_per_year=2, ns_params=None):
    """
    Par swap rate: R = (1 - P(0,T)) / annuity_factor.
    Floating leg telescopes to [P(0,0) - P(0,T)] = 1 - P(0,T).
    """
    if ns_params is None:
        ns_params = NS_USD
    times, deltas = build_schedule(T, freq_per_year)
    P = discount_factor(times, **ns_params)
    P_T = discount_factor(np.array([T]), **ns_params)[0]
    annuity = np.sum(deltas * P)
    return (1 - P_T) / annuity, annuity

def irs_value(N, R_fixed, T, freq_per_year=2, ns_params=None, receive_fixed=True):
    """
    IRS mark-to-market value for the fixed-rate receiver.
    V = N * (R_fixed - R_par) * annuity
    """
    if ns_params is None:
        ns_params = NS_USD
    R_par, annuity = par_swap_rate(T, freq_per_year, ns_params)
    sign = 1 if receive_fixed else -1
    return sign * N * (R_fixed - R_par) * annuity, R_par

def swap_dv01(N, T, freq_per_year=2, ns_params=None):
    """
    DV01 of a receive-fixed IRS: value change per 1bp parallel shift.
    DV01 = N * annuity * 0.0001
    """
    if ns_params is None:
        ns_params = NS_USD
    _, annuity = par_swap_rate(T, freq_per_year, ns_params)
    return N * annuity * 1e-4

def swap_cashflows(N, R_fixed, T, freq_per_year=2, ns_params=None):
    """
    Return detailed cash flows for fixed and floating legs.
    Floating rate for each period implied from the forward rate.
    """
    if ns_params is None:
        ns_params = NS_USD
    times, deltas = build_schedule(T, freq_per_year)
    P = discount_factor(times, **ns_params)

    cf_fixed = N * R_fixed * deltas
    # Forward rate: (P_{i-1}/P_i - 1) / delta_i
    P_prev = np.concatenate([[1.0], P[:-1]])
    fwd    = (P_prev / P - 1) / deltas
    cf_float = N * fwd * deltas

    pv_fixed = np.sum(cf_fixed * P)
    pv_float = np.sum(cf_float * P)
    return times, cf_fixed, cf_float, pv_fixed, pv_float

# ============================================================
# SECTION 3 — CURRENCY SWAP
# ============================================================

def cross_currency_swap_value(N_d, R_d, N_f, R_f, T,
                               freq_per_year, spot_fx,
                               ns_domestic, ns_foreign):
    """
    Cross-currency swap: receive domestic fixed, pay foreign fixed.
    V = PV_domestic - spot_fx * PV_foreign
    Includes notional exchange at inception (t=0) and maturity.
    N_d: domestic notional, N_f: foreign notional (= N_d / spot_fx)
    """
    times, deltas = build_schedule(T, freq_per_year)
    P_d = discount_factor(times, **ns_domestic)
    P_f = discount_factor(times, **ns_foreign)
    P_d_T = discount_factor(np.array([T]), **ns_domestic)[0]
    P_f_T = discount_factor(np.array([T]), **ns_foreign)[0]

    pv_domestic = N_d * R_d * np.sum(deltas * P_d) + N_d * P_d_T
    pv_foreign  = N_f * R_f * np.sum(deltas * P_f) + N_f * P_f_T
    return pv_domestic - spot_fx * pv_foreign, pv_domestic, pv_foreign

# ============================================================
# DIAGNOSTICS
# ============================================================
N  = 1_000_000   # notional USD 1M
T  = 5.0         # 5-year swap
freq = 2         # semi-annual

R_par_5y, annuity_5y = par_swap_rate(T, freq, NS_USD)
dv01_5y = swap_dv01(N, T, freq, NS_USD)
print(f"[M24] 5Y USD IRS: par swap rate = {R_par_5y*100:.4f}%")
print(f"      Annuity factor  = {annuity_5y:.6f}")
print(f"      DV01 (N=1M)     = USD {dv01_5y:.2f}")

times_cf, cf_fix, cf_flt, pv_fix, pv_flt = swap_cashflows(
    N, R_par_5y, T, freq, NS_USD
)
print(f"      PV(fixed leg)   = {pv_fix:,.2f}")
print(f"      PV(float leg)   = {pv_flt:,.2f}")
print(f"      Swap value (par)= {pv_fix - pv_flt:,.4f}  (should be ~0)")

# ============================================================
# FIGURE 1 — Yield curve + par swap rates + cash flows
# ============================================================
t0 = time.perf_counter()
print("[M24] Figure 1: Yield curve, par rates, and cash flows ...")

tau_grid = np.linspace(0.25, 30, 300)
y_usd = nelson_siegel_yields(tau_grid, **NS_USD) * 100
y_eur = nelson_siegel_yields(tau_grid, **NS_EUR) * 100

# Par swap rates across tenors
tenors    = [1, 2, 3, 5, 7, 10, 15, 20, 30]
par_rates = [par_swap_rate(T_, freq, NS_USD)[0]*100 for T_ in tenors]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M24 — Swap Pricing: IRS\nYield Curves, Par Rates, Cash Flows",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Yield curves
ax = axes[0]
ax.plot(tau_grid, y_usd, color=BLUE,  lw=2.5, label="USD (Nelson-Siegel)")
ax.plot(tau_grid, y_eur, color=GREEN, lw=2.5, label="EUR (Nelson-Siegel)")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
ax.set_title("Input Yield Curves\n(Discount factor basis)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (b) Par swap rates (swap curve) vs spot yields
ax = axes[1]
spot_at_tenors = [nelson_siegel_yields(np.array([T_]), **NS_USD)[0]*100 for T_ in tenors]
ax.plot(tau_grid, y_usd, color=BLUE, lw=2, linestyle="--", label="Spot yield curve", alpha=0.7)
ax.plot(tenors, par_rates,      color=ORANGE, lw=2.5, marker="o", ms=6, label="Par swap rate")
ax.plot(tenors, spot_at_tenors, color=BLUE,   lw=0, marker="s", ms=5, alpha=0.6)
for T_, R_ in zip(tenors, par_rates):
    ax.annotate(f"{R_:.2f}%", (T_, R_), textcoords="offset points",
                xytext=(0, 7), fontsize=6.5, color=ORANGE, ha="center")
ax.set_xlabel("Tenor (years)"); ax.set_ylabel("Rate (%)")
ax.set_title("Par Swap Rate vs Spot Yield\nR_par = (1-P(T)) / annuity",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (c) Cash flows for 5Y swap
ax = axes[2]
w = 0.15
ax.bar(times_cf - w, cf_fix/1e3,  width=w*1.8, color=BLUE,  alpha=0.85, label="Fixed leg")
ax.bar(times_cf + w, cf_flt/1e3,  width=w*1.8, color=GREEN, alpha=0.85, label="Float leg (fwd)")
ax.axhline(0, color=WHITE, lw=0.8, linestyle=":")
ax.set_xlabel("Payment date (years)"); ax.set_ylabel("Cash flow (USD thousands)")
ax.set_title(f"5Y IRS Cash Flows  (N=1M, R={R_par_5y*100:.3f}%)\n"
             f"PV(fixed)={pv_fix/1e3:.1f}k  PV(float)={pv_flt/1e3:.1f}k",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m24_01_yield_par_cashflows.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — MTM value and DV01 sensitivity
# ============================================================
t0 = time.perf_counter()
print("[M24] Figure 2: Mark-to-market value and DV01 analysis ...")

# Parallel shift of the USD curve
shifts   = np.linspace(-0.02, 0.02, 200)   # -200bp to +200bp
R_locked = R_par_5y                         # fixed rate locked at inception

mtm_vals = []
par_vals  = []
for dy in shifts:
    ns_shifted = {k: (v + dy if k in ("beta0",) else v) for k, v in NS_USD.items()}
    V, R_p = irs_value(N, R_locked, T, freq, ns_shifted, receive_fixed=True)
    mtm_vals.append(V)
    par_vals.append(R_p * 100)

mtm_vals = np.array(mtm_vals)
par_vals  = np.array(par_vals)

# DV01 across tenors
tenors_dv  = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])
dv01_vals  = [swap_dv01(N, T_, freq, NS_USD) for T_ in tenors_dv]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M24 — IRS Mark-to-Market and Rate Sensitivity",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) MTM value vs parallel shift
ax = axes[0]
ax.plot(shifts*1e4, mtm_vals/1e3, color=BLUE, lw=2.5)
ax.axhline(0,  color=WHITE,  lw=1, linestyle=":", alpha=0.7)
ax.axvline(0,  color=ORANGE, lw=1, linestyle="--", alpha=0.7)
ax.fill_between(shifts*1e4, mtm_vals/1e3, 0,
                where=mtm_vals >= 0, color=GREEN, alpha=0.15, label="In-the-money")
ax.fill_between(shifts*1e4, mtm_vals/1e3, 0,
                where=mtm_vals <  0, color=RED,   alpha=0.15, label="Out-of-the-money")
ax.set_xlabel("Parallel yield shift (bps)"); ax.set_ylabel("MTM value (USD thousands)")
ax.set_title(f"Receive-Fixed 5Y Swap MTM\nvs Parallel Rate Shift  (R_fixed={R_locked*100:.3f}%)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Par swap rate vs shift (swap curve moves)
ax = axes[1]
ax.plot(shifts*1e4, par_vals, color=ORANGE, lw=2.5, label="New par rate (5Y)")
ax.axhline(R_locked*100, color=WHITE, lw=1.5, linestyle="--",
           label=f"Locked rate {R_locked*100:.3f}%")
ax.fill_between(shifts*1e4, par_vals, R_locked*100,
                where=np.array(par_vals) > R_locked*100, color=RED,   alpha=0.15)
ax.fill_between(shifts*1e4, par_vals, R_locked*100,
                where=np.array(par_vals) < R_locked*100, color=GREEN, alpha=0.15)
ax.set_xlabel("Parallel yield shift (bps)"); ax.set_ylabel("Par swap rate (%)")
ax.set_title("Par Swap Rate vs Yield Shift\nRed = swap moves against receiver",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) DV01 across tenors
ax = axes[2]
ax.bar(tenors_dv, dv01_vals, color=PURPLE, alpha=0.85, width=0.6)
for T_, d_ in zip(tenors_dv, dv01_vals):
    ax.text(T_, d_ + 2, f"{d_:.0f}", ha="center", fontsize=7, color=WHITE)
ax.set_xlabel("Swap tenor (years)"); ax.set_ylabel("DV01 (USD per 1bp, N=1M)")
ax.set_title("DV01 vs Tenor\nDV01 = N * annuity * 0.0001",
             color=WHITE, fontsize=9)
ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m24_02_mtm_dv01.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Cross-Currency Swap
# ============================================================
t0 = time.perf_counter()
print("[M24] Figure 3: Cross-currency swap (USD vs EUR) ...")

spot_fx = 1.08          # USD/EUR
N_usd   = 1_000_000
N_eur   = N_usd / spot_fx

R_usd_par, _ = par_swap_rate(T, freq, NS_USD)
R_eur_par, _ = par_swap_rate(T, freq, NS_EUR)

print(f"      USD par rate: {R_usd_par*100:.4f}%  |  EUR par rate: {R_eur_par*100:.4f}%")

V0, pv_d, pv_f = cross_currency_swap_value(
    N_usd, R_usd_par, N_eur, R_eur_par, T, freq,
    spot_fx, NS_USD, NS_EUR
)
print(f"      XCCY Swap value at inception: USD {V0:,.2f}  (should be ~0)")

# MTM sensitivity to FX rate
fx_grid  = np.linspace(0.85, 1.35, 200)
xccy_mtm = []
for s in fx_grid:
    V_, _, _ = cross_currency_swap_value(
        N_usd, R_usd_par, N_eur, R_eur_par, T, freq,
        s, NS_USD, NS_EUR
    )
    xccy_mtm.append(V_)

# MTM sensitivity to EUR rate shift (USD fixed)
eur_shifts = np.linspace(-0.015, 0.015, 200)
xccy_rate_mtm = []
for dy in eur_shifts:
    ns_e_shifted = {k: (v + dy if k == "beta0" else v) for k, v in NS_EUR.items()}
    V_, _, _ = cross_currency_swap_value(
        N_usd, R_usd_par, N_eur, R_eur_par, T, freq,
        spot_fx, NS_USD, ns_e_shifted
    )
    xccy_rate_mtm.append(V_)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M24 — Cross-Currency Swap: USD (receive) vs EUR (pay)\n"
             "FX sensitivity | Rate sensitivity | Leg decomposition",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) MTM vs spot FX
ax = axes[0]
ax.plot(fx_grid, np.array(xccy_mtm)/1e3, color=BLUE, lw=2.5)
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7)
ax.axvline(spot_fx, color=ORANGE, lw=1.5, linestyle="--",
           label=f"Spot FX = {spot_fx}")
ax.fill_between(fx_grid, np.array(xccy_mtm)/1e3, 0,
                where=np.array(xccy_mtm) >= 0, color=GREEN, alpha=0.15)
ax.fill_between(fx_grid, np.array(xccy_mtm)/1e3, 0,
                where=np.array(xccy_mtm) <  0, color=RED,   alpha=0.15)
ax.set_xlabel("Spot FX (USD/EUR)"); ax.set_ylabel("MTM value (USD thousands)")
ax.set_title("XCCY MTM vs FX Rate\n(Receive USD fixed, pay EUR fixed)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) MTM vs EUR rate shift
ax = axes[1]
ax.plot(eur_shifts*1e4, np.array(xccy_rate_mtm)/1e3, color=GREEN, lw=2.5)
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7)
ax.fill_between(eur_shifts*1e4, np.array(xccy_rate_mtm)/1e3, 0,
                where=np.array(xccy_rate_mtm) >= 0, color=GREEN, alpha=0.15,
                label="Gain (EUR rates rise => pay leg cheaper)")
ax.fill_between(eur_shifts*1e4, np.array(xccy_rate_mtm)/1e3, 0,
                where=np.array(xccy_rate_mtm) <  0, color=RED,   alpha=0.15,
                label="Loss")
ax.set_xlabel("EUR parallel shift (bps)"); ax.set_ylabel("MTM value (USD thousands)")
ax.set_title("XCCY MTM vs EUR Rate Shift\nFX fixed at spot",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Leg PV comparison at inception
ax = axes[2]
legs    = ["USD fixed\n(receive)", "EUR fixed\n(pay, conv.)", "Net MTM"]
pv_eur_usd = pv_f * spot_fx
vals    = [pv_d/1e3, pv_eur_usd/1e3, (pv_d - pv_eur_usd)/1e3]
cols    = [GREEN, RED, BLUE]
bars    = ax.bar(legs, vals, color=cols, alpha=0.85, width=0.5)
ax.axhline(0, color=WHITE, lw=0.8, linestyle=":")
for bar_, v_ in zip(bars, vals):
    ax.text(bar_.get_x() + bar_.get_width()/2, v_ + (2 if v_ >= 0 else -5),
            f"{v_:,.1f}k", ha="center", fontsize=8, color=WHITE)
ax.set_ylabel("PV (USD thousands)")
ax.set_title(f"XCCY Leg PVs at Inception\nN_USD={N_usd/1e3:.0f}k  "
             f"N_EUR={N_eur/1e3:.0f}k  FX={spot_fx}",
             color=WHITE, fontsize=9)
ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m24_03_xccy_swap.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 24 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] IRS: fixed leg PV = N*R*sum(delta_i*P_i)")
print("  [2] Float leg telescopes: PV = N*(1 - P(0,T))")
print("  [3] Par rate: R = (1-P(T)) / annuity  (MTM=0 at inception)")
print("  [4] MTM(t) = N*(R_fixed - R_par(t)) * annuity(t)")
print("  [5] DV01 = N * annuity * 0.0001  (rate sensitivity)")
print("  [6] XCCY: V = PV_domestic - spot_FX * PV_foreign")
print(f"  5Y USD par rate  : {R_par_5y*100:.4f}%")
print(f"  5Y EUR par rate  : {R_eur_par*100:.4f}%")
print(f"  5Y DV01 (N=1M)   : USD {dv01_5y:.2f}")
print(f"  XCCY MTM (t=0)   : USD {V0:,.2f}  (par => ~0)")
print("=" * 65)

#!/usr/bin/env python3
"""
=============================================================================
MODULE 2: LEVERAGED BUYOUT (LBO) MODEL
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A complete LBO model simulating the acquisition of a target company
("IndustrialCo") by a private equity sponsor.  The model implements:

    1. Sources & Uses of Funds (transaction structuring)
    2. Multi-Tranche Debt Schedule (Senior A, Senior B, Mezzanine)
    3. Operating Model (5-year projection with margin expansion)
    4. Free Cash Flow to Equity & Mandatory Debt Repayment (cash sweep)
    5. Exit Valuation (multiple expansion / contraction scenarios)
    6. Returns Analysis (IRR, MOIC, equity value creation bridge)
    7. Sensitivity Analysis (Entry Multiple vs Exit Multiple, Leverage)

Theoretical Foundations
-----------------------
The LBO creates value through three primary levers:
    (a) Debt Paydown   -- Using FCF to amortize acquisition debt
    (b) EBITDA Growth   -- Organic growth + operational improvements
    (c) Multiple Expansion -- Exiting at a higher EV/EBITDA than entry

    Equity Return = f(Entry Price, Exit Price, Leverage, Cash Generation)

    IRR = r such that:  Equity_0 * (1+r)^T = Equity_T
    MOIC = Equity_T / Equity_0

References
----------
    - Rosenbaum, J. & Pearl, J. (2020). "Investment Banking", 3rd ed., Wiley.
    - Pignataro, P. (2013). "Leveraged Buyouts", Wiley Finance.
    - Stowell, D. (2017). "Investment Banks, Hedge Funds, and Private Equity",
      3rd ed., Academic Press.

Output
------
    - Console: Sources & Uses, Debt Schedule, Returns Analysis
    - Figures: 6 publication-quality charts saved to outputs/figures/lbo/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency, fmt_multiple
)
from src.common.finance_utils import (
    compute_irr, compute_moic, unlevered_fcf
)


# =============================================================================
# 1. TRANSACTION ASSUMPTIONS
# =============================================================================
print_section("MODULE 2: LBO MODEL -- IndustrialCo Acquisition")

# --- Target company ---
COMPANY = "IndustrialCo"
LTM_REVENUE  = 2_000e6       # $2.0B LTM revenue
LTM_EBITDA   = 400e6         # $400M LTM EBITDA (20% margin)
LTM_DA       = 80e6          # $80M D&A
LTM_CAPEX    = 100e6         # $100M CapEx

# --- Transaction terms ---
ENTRY_MULTIPLE = 10.0         # EV/EBITDA entry multiple
TRANSACTION_FEES = 0.025      # 2.5% of EV (advisory + financing fees)
MANAGEMENT_ROLLOVER = 0.05    # 5% of equity from management

ENTERPRISE_VALUE = LTM_EBITDA * ENTRY_MULTIPLE
FEES = ENTERPRISE_VALUE * TRANSACTION_FEES

# --- Debt tranches ---
# Senior A: amortizing, lowest cost
# Senior B: bullet (no amort), medium cost
# Mezzanine: bullet, highest cost (PIK component)
SENIOR_A_TURNS   = 2.5        # 2.5x EBITDA
SENIOR_B_TURNS   = 1.5        # 1.5x EBITDA
MEZZ_TURNS       = 1.0        # 1.0x EBITDA
TOTAL_DEBT_TURNS = SENIOR_A_TURNS + SENIOR_B_TURNS + MEZZ_TURNS  # 5.0x

SENIOR_A_AMT  = LTM_EBITDA * SENIOR_A_TURNS    # $1,000M
SENIOR_B_AMT  = LTM_EBITDA * SENIOR_B_TURNS    # $600M
MEZZ_AMT      = LTM_EBITDA * MEZZ_TURNS        # $400M
TOTAL_DEBT    = SENIOR_A_AMT + SENIOR_B_AMT + MEZZ_AMT  # $2,000M

SENIOR_A_RATE  = 0.050        # 5.0% cash interest
SENIOR_B_RATE  = 0.065        # 6.5% cash interest
MEZZ_CASH_RATE = 0.060        # 6.0% cash interest
MEZZ_PIK_RATE  = 0.040        # 4.0% PIK (paid-in-kind, accrues)

# Senior A amortization: 10% per year mandatory
SENIOR_A_AMORT_PCT = 0.10

# --- Equity ---
TOTAL_SOURCES = ENTERPRISE_VALUE + FEES
SPONSOR_EQUITY = TOTAL_SOURCES - TOTAL_DEBT
MGMT_ROLLOVER_AMT = SPONSOR_EQUITY * MANAGEMENT_ROLLOVER
SPONSOR_CHECK = SPONSOR_EQUITY - MGMT_ROLLOVER_AMT

# --- Operating assumptions (5-year hold) ---
N_YEARS = 5
YEARS = np.arange(0, N_YEARS + 1)  # 0 = entry, 1-5 = projection

REVENUE_GROWTH   = np.array([0.06, 0.06, 0.05, 0.05, 0.04])
EBITDA_MARGIN    = np.array([0.21, 0.22, 0.23, 0.23, 0.24])  # Margin expansion
DA_PCT_REV       = np.full(N_YEARS, 0.04)
CAPEX_PCT_REV    = np.array([0.05, 0.05, 0.045, 0.045, 0.04])
TAX_RATE         = 0.25
NWC_PCT_REV      = np.array([0.10, 0.10, 0.10, 0.095, 0.095])

# --- Exit assumptions ---
EXIT_MULTIPLE    = 10.5       # Slight multiple expansion

# --- Cash sweep: excess FCF applied to Senior A first ---
CASH_SWEEP_PCT   = 0.75       # 75% of excess cash flow to debt repayment
MIN_CASH_BALANCE = 50e6       # Minimum cash retained

print(f"  Target            : {COMPANY}")
print(f"  Enterprise Value  : {fmt_currency(ENTERPRISE_VALUE)}")
print(f"  Entry Multiple    : {ENTRY_MULTIPLE:.1f}x EV/EBITDA")
print(f"  Total Debt        : {fmt_currency(TOTAL_DEBT)} ({TOTAL_DEBT_TURNS:.1f}x)")
print(f"  Sponsor Equity    : {fmt_currency(SPONSOR_EQUITY)}")


# =============================================================================
# 2. SOURCES & USES
# =============================================================================
print_section("SOURCES & USES OF FUNDS")

sources_rows = [
    ["Senior Debt A (Term Loan A)", fmt_currency(SENIOR_A_AMT),
     f"{SENIOR_A_TURNS:.1f}x", f"{SENIOR_A_RATE:.1%}"],
    ["Senior Debt B (Term Loan B)", fmt_currency(SENIOR_B_AMT),
     f"{SENIOR_B_TURNS:.1f}x", f"{SENIOR_B_RATE:.1%}"],
    ["Mezzanine Debt", fmt_currency(MEZZ_AMT),
     f"{MEZZ_TURNS:.1f}x", f"{MEZZ_CASH_RATE:.1%} + {MEZZ_PIK_RATE:.1%} PIK"],
    ["Sponsor Equity", fmt_currency(SPONSOR_CHECK), "", ""],
    ["Management Rollover", fmt_currency(MGMT_ROLLOVER_AMT), "", ""],
    ["TOTAL SOURCES", fmt_currency(TOTAL_SOURCES), "", ""],
]
print_table("SOURCES", ["Source", "Amount", "Turns", "Rate"], sources_rows)

uses_rows = [
    ["Enterprise Value", fmt_currency(ENTERPRISE_VALUE), f"{ENTERPRISE_VALUE/TOTAL_SOURCES*100:.1f}%"],
    ["Transaction Fees", fmt_currency(FEES), f"{FEES/TOTAL_SOURCES*100:.1f}%"],
    ["TOTAL USES", fmt_currency(TOTAL_SOURCES), "100.0%"],
]
print_table("USES", ["Use", "Amount", "% of Total"], uses_rows)


# =============================================================================
# 3. OPERATING MODEL -- 5-YEAR PROJECTION
# =============================================================================
print_section("OPERATING MODEL -- 5-YEAR PROJECTION")

revenue = np.zeros(N_YEARS + 1)
revenue[0] = LTM_REVENUE
for i in range(N_YEARS):
    revenue[i + 1] = revenue[i] * (1.0 + REVENUE_GROWTH[i])

ebitda = np.zeros(N_YEARS + 1)
ebitda[0] = LTM_EBITDA
for i in range(N_YEARS):
    ebitda[i + 1] = revenue[i + 1] * EBITDA_MARGIN[i]

da = np.zeros(N_YEARS + 1)
da[0] = LTM_DA
for i in range(N_YEARS):
    da[i + 1] = revenue[i + 1] * DA_PCT_REV[i]

ebit = ebitda - da

capex = np.zeros(N_YEARS + 1)
capex[0] = LTM_CAPEX
for i in range(N_YEARS):
    capex[i + 1] = revenue[i + 1] * CAPEX_PCT_REV[i]

nwc = np.zeros(N_YEARS + 1)
nwc[0] = LTM_REVENUE * 0.10
for i in range(N_YEARS):
    nwc[i + 1] = revenue[i + 1] * NWC_PCT_REV[i]
delta_nwc = np.diff(nwc)

year_labels = ["Entry"] + [f"Year {y}" for y in range(1, N_YEARS + 1)]

op_headers = ["Item"] + year_labels
op_rows = [
    ["Revenue"]       + [fmt_currency(v) for v in revenue],
    ["  Growth %"]    + ["--"] + [f"{g:.1%}" for g in REVENUE_GROWTH],
    ["EBITDA"]        + [fmt_currency(v) for v in ebitda],
    ["  Margin %"]    + [f"{ebitda[0]/revenue[0]:.1%}"] +
                        [f"{m:.1%}" for m in EBITDA_MARGIN],
    ["(-) D&A"]       + [fmt_currency(v) for v in da],
    ["EBIT"]          + [fmt_currency(v) for v in ebit],
    ["(-) CapEx"]     + [fmt_currency(v) for v in capex],
    ["(-) Delta NWC"] + ["--"] + [fmt_currency(v) for v in delta_nwc],
]
print_table("OPERATING MODEL", op_headers, op_rows)


# =============================================================================
# 4. DEBT SCHEDULE WITH CASH SWEEP
# =============================================================================
print_section("DEBT SCHEDULE")

# Arrays to track debt balances over time (0 = closing, 1-5 = year-end)
sen_a_bal  = np.zeros(N_YEARS + 1)
sen_b_bal  = np.zeros(N_YEARS + 1)
mezz_bal   = np.zeros(N_YEARS + 1)

sen_a_bal[0]  = SENIOR_A_AMT
sen_b_bal[0]  = SENIOR_B_AMT
mezz_bal[0]   = MEZZ_AMT

# Interest arrays
sen_a_interest = np.zeros(N_YEARS + 1)
sen_b_interest = np.zeros(N_YEARS + 1)
mezz_cash_int  = np.zeros(N_YEARS + 1)
mezz_pik_int   = np.zeros(N_YEARS + 1)
total_interest = np.zeros(N_YEARS + 1)

# Amortization & sweep
sen_a_mandatory_amort = np.zeros(N_YEARS + 1)
sweep_applied         = np.zeros(N_YEARS + 1)
total_debt_repaid     = np.zeros(N_YEARS + 1)

# Cash balance
cash_balance = np.zeros(N_YEARS + 1)
cash_balance[0] = MIN_CASH_BALANCE

for yr in range(1, N_YEARS + 1):
    idx = yr  # projection index (1-based)

    # --- Interest expense (on beginning balance) ---
    sen_a_interest[idx] = sen_a_bal[idx - 1] * SENIOR_A_RATE
    sen_b_interest[idx] = sen_b_bal[idx - 1] * SENIOR_B_RATE
    mezz_cash_int[idx]  = mezz_bal[idx - 1] * MEZZ_CASH_RATE
    mezz_pik_int[idx]   = mezz_bal[idx - 1] * MEZZ_PIK_RATE
    total_interest[idx] = (sen_a_interest[idx] + sen_b_interest[idx] +
                           mezz_cash_int[idx])

    # --- PIK accrual (added to mezzanine balance) ---
    mezz_bal_after_pik = mezz_bal[idx - 1] + mezz_pik_int[idx]

    # --- Mandatory Senior A amortization ---
    mandatory = min(SENIOR_A_AMT * SENIOR_A_AMORT_PCT, sen_a_bal[idx - 1])
    sen_a_mandatory_amort[idx] = mandatory

    # --- Pre-debt FCF ---
    # EBIT - taxes + D&A - CapEx - delta NWC - cash interest - mandatory amort
    taxes = max(0, (ebit[idx] - total_interest[idx])) * TAX_RATE
    pre_debt_fcf = (ebit[idx] - taxes + da[idx] - capex[idx] -
                    delta_nwc[idx - 1] - total_interest[idx] - mandatory)

    # --- Cash sweep ---
    available_for_sweep = max(0, pre_debt_fcf - MIN_CASH_BALANCE + cash_balance[idx - 1])
    sweep = available_for_sweep * CASH_SWEEP_PCT

    # Apply sweep: Senior A first, then Senior B
    remaining_sen_a = sen_a_bal[idx - 1] - mandatory
    sweep_to_a = min(sweep, max(0, remaining_sen_a))
    sweep_remaining = sweep - sweep_to_a
    sweep_to_b = min(sweep_remaining, sen_b_bal[idx - 1])
    sweep_applied[idx] = sweep_to_a + sweep_to_b

    # --- End-of-year balances ---
    sen_a_bal[idx] = max(0, sen_a_bal[idx - 1] - mandatory - sweep_to_a)
    sen_b_bal[idx] = max(0, sen_b_bal[idx - 1] - sweep_to_b)
    mezz_bal[idx]  = mezz_bal_after_pik  # PIK accrues, no cash repayment

    total_debt_repaid[idx] = mandatory + sweep_to_a + sweep_to_b

    # Cash balance
    cash_balance[idx] = (cash_balance[idx - 1] + pre_debt_fcf -
                         sweep_applied[idx])
    cash_balance[idx] = max(cash_balance[idx], MIN_CASH_BALANCE)

total_debt_arr = sen_a_bal + sen_b_bal + mezz_bal
net_debt_arr = total_debt_arr - cash_balance
leverage_arr = np.where(ebitda > 0, total_debt_arr / ebitda, 0)

# Print debt schedule
debt_headers = ["Item"] + year_labels
debt_rows = [
    ["Senior A - Beg Bal"]   + [fmt_currency(v) for v in
                                 [SENIOR_A_AMT] + list(sen_a_bal[:N_YEARS])],
    ["  Interest"]           + ["--"] + [fmt_currency(v) for v in sen_a_interest[1:]],
    ["  Mandatory Amort"]    + ["--"] + [fmt_currency(v) for v in sen_a_mandatory_amort[1:]],
    ["  Cash Sweep"]         + ["--"] + [fmt_currency(min(sweep_applied[i],
                                 max(0, sen_a_bal[i-1] - sen_a_mandatory_amort[i])))
                                 for i in range(1, N_YEARS + 1)],
    ["Senior A - End Bal"]   + [fmt_currency(v) for v in sen_a_bal],
    ["", ""] + [""] * N_YEARS,
    ["Senior B - Beg Bal"]   + [fmt_currency(v) for v in
                                 [SENIOR_B_AMT] + list(sen_b_bal[:N_YEARS])],
    ["  Interest"]           + ["--"] + [fmt_currency(v) for v in sen_b_interest[1:]],
    ["Senior B - End Bal"]   + [fmt_currency(v) for v in sen_b_bal],
    ["", ""] + [""] * N_YEARS,
    ["Mezzanine - Beg Bal"]  + [fmt_currency(v) for v in
                                 [MEZZ_AMT] + list(mezz_bal[:N_YEARS])],
    ["  Cash Interest"]      + ["--"] + [fmt_currency(v) for v in mezz_cash_int[1:]],
    ["  PIK Accrual"]        + ["--"] + [fmt_currency(v) for v in mezz_pik_int[1:]],
    ["Mezzanine - End Bal"]  + [fmt_currency(v) for v in mezz_bal],
    ["", ""] + [""] * N_YEARS,
    ["TOTAL DEBT"]           + [fmt_currency(v) for v in total_debt_arr],
    ["Net Debt"]             + [fmt_currency(v) for v in net_debt_arr],
    ["Leverage (Debt/EBITDA)"] + [f"{v:.1f}x" for v in leverage_arr],
]
print_table("DEBT SCHEDULE", debt_headers, debt_rows)


# =============================================================================
# 5. EXIT VALUATION & RETURNS ANALYSIS
# =============================================================================
print_section("EXIT VALUATION & RETURNS ANALYSIS")

# Exit enterprise value
exit_ebitda = ebitda[N_YEARS]
exit_ev = exit_ebitda * EXIT_MULTIPLE

# Equity at exit
exit_total_debt = total_debt_arr[N_YEARS]
exit_cash = cash_balance[N_YEARS]
exit_net_debt = exit_total_debt - exit_cash
exit_equity = exit_ev - exit_net_debt

# IRR calculation
# Cash flows: initial equity out (negative), terminal equity in (positive)
irr_cashflows = np.zeros(N_YEARS + 1)
irr_cashflows[0] = -SPONSOR_EQUITY
irr_cashflows[N_YEARS] = exit_equity

irr = compute_irr(irr_cashflows)
moic = compute_moic(exit_equity, SPONSOR_EQUITY)

# Value creation bridge
entry_equity = SPONSOR_EQUITY
ebitda_growth_effect = (exit_ebitda - LTM_EBITDA) * ENTRY_MULTIPLE
multiple_effect = (EXIT_MULTIPLE - ENTRY_MULTIPLE) * exit_ebitda
deleveraging_effect = (TOTAL_DEBT - exit_total_debt) + (exit_cash - MIN_CASH_BALANCE)
fees_effect = -FEES

bridge_check = (entry_equity + ebitda_growth_effect + multiple_effect +
                deleveraging_effect + fees_effect)

returns_rows = [
    ["Entry EV",                fmt_currency(ENTERPRISE_VALUE)],
    ["Entry Equity",            fmt_currency(SPONSOR_EQUITY)],
    ["", ""],
    ["Exit EBITDA",             fmt_currency(exit_ebitda)],
    ["Exit Multiple",           f"{EXIT_MULTIPLE:.1f}x"],
    ["Exit EV",                 fmt_currency(exit_ev)],
    ["(-) Exit Net Debt",       fmt_currency(exit_net_debt)],
    ["Exit Equity",             fmt_currency(exit_equity)],
    ["", ""],
    ["IRR",                     f"{irr:.1%}"],
    ["MOIC",                    f"{moic:.2f}x"],
    ["", ""],
    ["--- Value Creation Bridge ---", ""],
    ["Entry Equity",            fmt_currency(entry_equity)],
    ["(+) EBITDA Growth",       fmt_currency(ebitda_growth_effect)],
    ["(+) Multiple Expansion",  fmt_currency(multiple_effect)],
    ["(+) Deleveraging",        fmt_currency(deleveraging_effect)],
    ["(-) Fees & Costs",        fmt_currency(fees_effect)],
    ["= Exit Equity (check)",   fmt_currency(bridge_check)],
]
print_table("RETURNS ANALYSIS", ["Component", "Value"], returns_rows)


# =============================================================================
# 6. SENSITIVITY ANALYSIS
# =============================================================================
print_section("SENSITIVITY ANALYSIS")

# Grid 1: Entry Multiple vs Exit Multiple
entry_range = np.arange(8.0, 12.5, 0.5)
exit_range  = np.arange(8.0, 13.5, 0.5)

irr_grid_em = np.zeros((len(entry_range), len(exit_range)))
moic_grid_em = np.zeros((len(entry_range), len(exit_range)))

for i, entry_m in enumerate(entry_range):
    for j, exit_m in enumerate(exit_range):
        # Recalculate equity at entry
        ev_entry = LTM_EBITDA * entry_m
        fees_i = ev_entry * TRANSACTION_FEES
        eq_entry = ev_entry + fees_i - TOTAL_DEBT

        if eq_entry <= 0:
            irr_grid_em[i, j] = np.nan
            moic_grid_em[i, j] = np.nan
            continue

        # Exit equity (debt schedule remains same for simplicity)
        ev_exit = exit_ebitda * exit_m
        eq_exit = ev_exit - exit_net_debt

        cf = np.zeros(N_YEARS + 1)
        cf[0] = -eq_entry
        cf[N_YEARS] = eq_exit

        irr_grid_em[i, j] = compute_irr(cf)
        moic_grid_em[i, j] = compute_moic(eq_exit, eq_entry)

# Grid 2: Leverage (Total Debt/EBITDA) vs Exit Multiple
lev_range = np.arange(3.0, 6.5, 0.5)
irr_grid_lev = np.zeros((len(lev_range), len(exit_range)))

for i, lev in enumerate(lev_range):
    total_d = LTM_EBITDA * lev
    eq_entry = ENTERPRISE_VALUE + FEES - total_d
    if eq_entry <= 0:
        irr_grid_lev[i, :] = np.nan
        continue
    for j, exit_m in enumerate(exit_range):
        ev_exit = exit_ebitda * exit_m
        # Approximate exit debt (scale proportionally)
        scale = total_d / TOTAL_DEBT
        approx_exit_debt = exit_total_debt * scale
        approx_exit_cash = exit_cash
        eq_exit = ev_exit - approx_exit_debt + approx_exit_cash

        cf = np.zeros(N_YEARS + 1)
        cf[0] = -eq_entry
        cf[N_YEARS] = eq_exit
        irr_grid_lev[i, j] = compute_irr(cf)

# Print IRR sensitivity
hdr_em = ["Entry \\ Exit"] + [f"{m:.1f}x" for m in exit_range]
rows_em = []
for i, entry_m in enumerate(entry_range):
    row = [f"{entry_m:.1f}x"]
    for j in range(len(exit_range)):
        val = irr_grid_em[i, j]
        row.append(f"{val:.1%}" if not np.isnan(val) else "N/A")
    rows_em.append(row)
print_table("IRR SENSITIVITY: Entry vs Exit Multiple", hdr_em, rows_em)


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

# -------------------------------------------------------------------------
# FIGURE 1: Sources & Uses Horizontal Bar
# -------------------------------------------------------------------------
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

# Sources
src_labels = ["Senior A", "Senior B", "Mezzanine", "Sponsor Eq", "Mgmt Roll"]
src_values = [SENIOR_A_AMT/1e6, SENIOR_B_AMT/1e6, MEZZ_AMT/1e6,
              SPONSOR_CHECK/1e6, MGMT_ROLLOVER_AMT/1e6]
src_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["purple"], COLORS["pink"]]

bars_s = ax1a.barh(src_labels, src_values, color=src_colors, alpha=0.85,
                   edgecolor="none", height=0.6)
for bar, val in zip(bars_s, src_values):
    ax1a.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
              f"${val:,.0f}M", va="center", fontsize=9, color=COLORS["white"])
ax1a.set_title("Sources of Funds", fontsize=12, fontweight="bold")
ax1a.set_xlabel("USD (Millions)")
ax1a.invert_yaxis()

# Uses
use_labels = ["Enterprise\nValue", "Transaction\nFees"]
use_values = [ENTERPRISE_VALUE/1e6, FEES/1e6]
use_colors = [COLORS["teal"], COLORS["danger"]]

bars_u = ax1b.barh(use_labels, use_values, color=use_colors, alpha=0.85,
                   edgecolor="none", height=0.6)
for bar, val in zip(bars_u, use_values):
    ax1b.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
              f"${val:,.0f}M", va="center", fontsize=9, color=COLORS["white"])
ax1b.set_title("Uses of Funds", fontsize=12, fontweight="bold")
ax1b.set_xlabel("USD (Millions)")
ax1b.invert_yaxis()

fig1.suptitle(f"{COMPANY} LBO -- Sources & Uses", fontsize=14,
              fontweight="bold", y=1.02)
fig1.tight_layout()
save_figure(fig1, "lbo_01_sources_uses", subdir="lbo")

# -------------------------------------------------------------------------
# FIGURE 2: Debt Paydown Schedule (Stacked Area)
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

x2 = np.arange(N_YEARS + 1)
ax2.fill_between(x2, 0, sen_a_bal/1e6, alpha=0.7, color=COLORS["primary"],
                 label=f"Senior A ({SENIOR_A_RATE:.1%})")
ax2.fill_between(x2, sen_a_bal/1e6, (sen_a_bal + sen_b_bal)/1e6,
                 alpha=0.7, color=COLORS["secondary"],
                 label=f"Senior B ({SENIOR_B_RATE:.1%})")
ax2.fill_between(x2, (sen_a_bal + sen_b_bal)/1e6, total_debt_arr/1e6,
                 alpha=0.7, color=COLORS["accent"],
                 label=f"Mezzanine ({MEZZ_CASH_RATE:.1%} + {MEZZ_PIK_RATE:.1%} PIK)")

# Leverage ratio on secondary axis
ax2b = ax2.twinx()
ax2b.plot(x2, leverage_arr, color=COLORS["danger"], marker="s",
          linewidth=2, markersize=7, label="Leverage (x)", zorder=5)
ax2b.set_ylabel("Debt / EBITDA (x)", color=COLORS["danger"])
ax2b.tick_params(axis="y", labelcolor=COLORS["danger"])
ax2b.set_ylim(0, 6)

ax2.set_xlabel("Year")
ax2.set_ylabel("Debt Outstanding ($ Millions)")
ax2.set_title(f"{COMPANY} -- Debt Paydown Schedule", fontsize=14,
              fontweight="bold", pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(year_labels, fontsize=9)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

fig2.tight_layout()
save_figure(fig2, "lbo_02_debt_paydown", subdir="lbo")

# -------------------------------------------------------------------------
# FIGURE 3: EBITDA Growth & Margin Expansion
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))

x3 = np.arange(N_YEARS + 1)
bars3 = ax3.bar(x3, ebitda / 1e6, color=COLORS["secondary"], alpha=0.85,
                edgecolor="none", width=0.6)
for bar, val in zip(bars3, ebitda):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"${val/1e6:.0f}M", ha="center", fontsize=8, color=COLORS["white"])

ax3b = ax3.twinx()
margins = ebitda / revenue * 100
ax3b.plot(x3, margins, color=COLORS["accent"], marker="o",
          linewidth=2, markersize=6, label="EBITDA Margin %")
ax3b.set_ylabel("EBITDA Margin (%)", color=COLORS["accent"])
ax3b.tick_params(axis="y", labelcolor=COLORS["accent"])
ax3b.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct))

ax3.set_xlabel("Year")
ax3.set_ylabel("EBITDA ($ Millions)")
ax3.set_title(f"{COMPANY} -- EBITDA Growth & Margin Expansion",
              fontsize=14, fontweight="bold", pad=15)
ax3.set_xticks(x3)
ax3.set_xticklabels(year_labels)

fig3.tight_layout()
save_figure(fig3, "lbo_03_ebitda_growth", subdir="lbo")

# -------------------------------------------------------------------------
# FIGURE 4: Value Creation Bridge (Waterfall)
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(11, 6))

bridge_labels = ["Entry\nEquity", "EBITDA\nGrowth", "Multiple\nExpansion",
                 "Delev-\neraging", "Fees &\nCosts", "Exit\nEquity"]
bridge_values = [entry_equity, ebitda_growth_effect, multiple_effect,
                 deleveraging_effect, fees_effect, exit_equity]

# Calculate waterfall positions
cumvals = np.zeros(len(bridge_values))
cumvals[0] = bridge_values[0]
for k in range(1, len(bridge_values) - 1):
    cumvals[k] = cumvals[k-1] + bridge_values[k]
cumvals[-1] = bridge_values[-1]

bot = np.zeros(len(bridge_values))
bot[0] = 0
for k in range(1, len(bridge_values) - 1):
    if bridge_values[k] >= 0:
        bot[k] = cumvals[k] - bridge_values[k]
    else:
        bot[k] = cumvals[k]
bot[-1] = 0

bcolors = []
for k, v in enumerate(bridge_values):
    if k == 0:
        bcolors.append(COLORS["primary"])
    elif k == len(bridge_values) - 1:
        bcolors.append(COLORS["teal"])
    elif v >= 0:
        bcolors.append(COLORS["secondary"])
    else:
        bcolors.append(COLORS["danger"])

ax4.bar(bridge_labels, [abs(v)/1e6 for v in bridge_values],
        bottom=[b/1e6 for b in bot], color=bcolors, edgecolor="none",
        alpha=0.85, width=0.55)

for k, (lbl, val) in enumerate(zip(bridge_labels, bridge_values)):
    y_pos = (bot[k] + abs(val)/2) / 1e6
    sign = "+" if val > 0 and k > 0 else ""
    ax4.text(k, y_pos, f"{sign}${abs(val)/1e6:.0f}M",
             ha="center", va="center", fontsize=9, color=COLORS["white"],
             fontweight="bold")

ax4.set_title(f"{COMPANY} -- Equity Value Creation Bridge",
              fontsize=14, fontweight="bold", pad=15)
ax4.set_ylabel("USD (Millions)")

# Annotations
ax4.text(0.02, 0.95, f"IRR: {irr:.1%}  |  MOIC: {moic:.2f}x",
         transform=ax4.transAxes, fontsize=11, color=COLORS["accent"],
         fontweight="bold", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                   edgecolor=COLORS["accent"], alpha=0.8))

fig4.tight_layout()
save_figure(fig4, "lbo_04_value_bridge", subdir="lbo")

# -------------------------------------------------------------------------
# FIGURE 5: IRR Sensitivity Heatmap (Entry vs Exit Multiple)
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 7))

masked_irr = np.ma.masked_invalid(irr_grid_em * 100)  # to percent
im5 = ax5.imshow(masked_irr, cmap="RdYlGn", aspect="auto",
                 vmin=0, vmax=40)

for i in range(len(entry_range)):
    for j in range(len(exit_range)):
        val = irr_grid_em[i, j]
        if np.isnan(val):
            txt = "N/A"
            clr = "#666666"
        else:
            txt = f"{val:.0%}"
            clr = "black" if val > 0.20 else COLORS["white"]
        ax5.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=clr)

ax5.set_xticks(np.arange(len(exit_range)))
ax5.set_xticklabels([f"{m:.1f}x" for m in exit_range], rotation=45, ha="right")
ax5.set_yticks(np.arange(len(entry_range)))
ax5.set_yticklabels([f"{m:.1f}x" for m in entry_range])
ax5.set_xlabel("Exit EV/EBITDA Multiple")
ax5.set_ylabel("Entry EV/EBITDA Multiple")
ax5.set_title(f"{COMPANY} -- IRR Sensitivity: Entry vs Exit Multiple",
              fontsize=13, fontweight="bold", pad=15)

cbar5 = fig5.colorbar(im5, ax=ax5, shrink=0.8)
cbar5.set_label("IRR (%)", color=COLORS["white"])
cbar5.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar5.ax.axes, "yticklabels"), color=COLORS["white"])

fig5.tight_layout()
save_figure(fig5, "lbo_05_irr_sensitivity", subdir="lbo")

# -------------------------------------------------------------------------
# FIGURE 6: Returns Waterfall -- IRR vs Hold Period
# -------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(10, 6))

hold_periods = np.arange(3, 8)
irr_by_hold = []
moic_by_hold = []

for hp in hold_periods:
    # Project EBITDA forward to hold period
    projected_ebitda = LTM_EBITDA
    for yr in range(hp):
        growth = REVENUE_GROWTH[min(yr, N_YEARS - 1)]
        margin = EBITDA_MARGIN[min(yr, N_YEARS - 1)]
        projected_rev = revenue[0]
        for y in range(yr + 1):
            projected_rev *= (1.0 + REVENUE_GROWTH[min(y, N_YEARS - 1)])
        projected_ebitda = projected_rev * margin

    exit_ev_hp = projected_ebitda * EXIT_MULTIPLE
    # Approximate debt at hold period (linear interpolation)
    if hp <= N_YEARS:
        approx_debt = total_debt_arr[hp]
        approx_cash = cash_balance[hp]
    else:
        approx_debt = total_debt_arr[N_YEARS] * 0.85  # further paydown
        approx_cash = cash_balance[N_YEARS] * 1.1

    eq_exit_hp = exit_ev_hp - approx_debt + approx_cash
    cf_hp = np.zeros(hp + 1)
    cf_hp[0] = -SPONSOR_EQUITY
    cf_hp[-1] = eq_exit_hp

    irr_hp = compute_irr(cf_hp)
    moic_hp = compute_moic(eq_exit_hp, SPONSOR_EQUITY)
    irr_by_hold.append(irr_hp)
    moic_by_hold.append(moic_hp)

bars6 = ax6.bar(hold_periods, [i * 100 for i in irr_by_hold],
                color=COLORS["primary"], alpha=0.85, edgecolor="none", width=0.6)
for bar, irr_val, moic_val in zip(bars6, irr_by_hold, moic_by_hold):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{irr_val:.1%}\n{moic_val:.2f}x",
             ha="center", fontsize=9, color=COLORS["white"])

# PE benchmark lines
ax6.axhline(y=20, color=COLORS["secondary"], linewidth=1.2, linestyle="--",
            alpha=0.7, label="20% IRR Target")
ax6.axhline(y=25, color=COLORS["accent"], linewidth=1.2, linestyle="--",
            alpha=0.7, label="25% IRR Premium")

ax6.set_xlabel("Hold Period (Years)")
ax6.set_ylabel("IRR (%)")
ax6.set_title(f"{COMPANY} -- Returns by Hold Period",
              fontsize=14, fontweight="bold", pad=15)
ax6.set_xticks(hold_periods)
ax6.set_xticklabels([f"{hp}Y" for hp in hold_periods])
ax6.legend(loc="upper right")

fig6.tight_layout()
save_figure(fig6, "lbo_06_returns_by_hold", subdir="lbo")


# =============================================================================
# 8. SUMMARY
# =============================================================================
print_section("LBO MODEL SUMMARY")
print(f"  Target              : {COMPANY}")
print(f"  Entry EV            : {fmt_currency(ENTERPRISE_VALUE)} ({ENTRY_MULTIPLE:.1f}x)")
print(f"  Sponsor Equity      : {fmt_currency(SPONSOR_EQUITY)}")
print(f"  Total Leverage      : {TOTAL_DEBT_TURNS:.1f}x EBITDA at entry")
print(f"  Exit EV             : {fmt_currency(exit_ev)} ({EXIT_MULTIPLE:.1f}x)")
print(f"  Exit Equity         : {fmt_currency(exit_equity)}")
print(f"  Exit Leverage       : {leverage_arr[N_YEARS]:.1f}x EBITDA")
print(f"  IRR                 : {irr:.1%}")
print(f"  MOIC                : {moic:.2f}x")
print(f"  Hold Period         : {N_YEARS} years")
print(f"  Figures saved to    : outputs/figures/lbo/")
print(f"  Module 2 of 6 complete.")
print("=" * 70)

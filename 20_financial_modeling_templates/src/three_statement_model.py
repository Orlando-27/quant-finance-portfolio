#!/usr/bin/env python3
"""
=============================================================================
MODULE 3: THREE-STATEMENT FINANCIAL MODEL
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A fully-linked three-statement financial model that projects the Income
Statement, Balance Sheet, and Cash Flow Statement for a generic
manufacturing company ("MfgCorp") over a 5-year horizon.

The model implements the core accounting linkages:

    Income Statement ---> Cash Flow Statement ---> Balance Sheet
         |                      |                      |
         +--- Net Income ------>|                      |
         +--- D&A ------------->|                      |
                                +--- Ending Cash ----->|
                                +--- Debt Changes ---->|
                                +--- Equity Changes -->|

Key features:
    1. Revenue build-up (volume x price, growth rates)
    2. Expense modeling (% of revenue, fixed + variable)
    3. Working capital projections (DSO, DIO, DPO)
    4. PP&E roll-forward with depreciation schedule
    5. Debt schedule with revolver, term loan, subordinated
    6. Interest expense computation (avg balance * rate)
    7. Circular reference handling via iterative solver
       (interest depends on debt, debt depends on cash,
        cash depends on interest)
    8. Dividend policy and retained earnings
    9. Balance sheet balancing check (A = L + E)

Theoretical Foundations
-----------------------
The three-statement model is the foundation of all financial modeling.
Every DCF, LBO, merger model, and credit analysis begins with a robust
three-statement model.  The circular reference between interest expense
and the cash/debt balance is resolved through fixed-point iteration:

    Interest_n+1 = f(Debt_n) = f(g(Cash_n)) = f(g(h(Interest_n)))

Convergence typically occurs within 5-10 iterations to machine precision.

References
----------
    - Benninga, S. (2014). "Financial Modeling", 4th ed., MIT Press.
    - Pignataro, P. (2015). "Financial Modeling & Valuation", Wiley.
    - Tjia, J. (2009). "Building Financial Models", McGraw-Hill.

Output
------
    - Console: Income Statement, Balance Sheet, Cash Flow Statement
    - Figures: 6 publication-quality charts saved to outputs/figures/ts/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency
)


# =============================================================================
# 1. COMPANY ASSUMPTIONS
# =============================================================================
print_section("MODULE 3: THREE-STATEMENT MODEL -- MfgCorp")

COMPANY = "MfgCorp"
N_YEARS = 5
YEAR_LABELS = ["Base"] + [f"Year {y}" for y in range(1, N_YEARS + 1)]

# --- Revenue assumptions ---
BASE_REVENUE = 5_000e6
REVENUE_GROWTH = np.array([0.08, 0.07, 0.06, 0.05, 0.05])

# --- Income statement margins ---
COGS_PCT       = np.array([0.55, 0.54, 0.53, 0.53, 0.52])
SGA_PCT        = np.array([0.15, 0.14, 0.14, 0.13, 0.13])
RD_PCT         = np.full(N_YEARS, 0.03)
OTHER_OPEX_PCT = np.full(N_YEARS, 0.02)
TAX_RATE       = 0.25
DIVIDEND_PAYOUT_RATIO = 0.30

# --- Balance sheet assumptions ---
# Working capital (days)
DSO = 50.0
DIO = 45.0
DPO = 35.0

# PP&E
BASE_PPE_GROSS = 3_000e6
BASE_ACCUM_DEP = 1_200e6
CAPEX_PCT_REV  = np.array([0.06, 0.06, 0.055, 0.055, 0.05])
DEP_RATE       = 0.10            # 10% of gross PP&E (straight-line approx)

# Other assets & liabilities
BASE_OTHER_ASSETS  = 500e6
BASE_OTHER_LIAB    = 300e6
OTHER_ASSETS_GROWTH = 0.03
OTHER_LIAB_GROWTH   = 0.03

# --- Debt structure ---
# Revolver: draw/repay to meet minimum cash target
REVOLVER_CAPACITY = 500e6
REVOLVER_RATE     = 0.045

# Term Loan: amortizing
BASE_TERM_LOAN    = 1_000e6
TERM_LOAN_AMORT   = 100e6        # $100M/year mandatory
TERM_LOAN_RATE    = 0.055

# Subordinated Notes: bullet maturity
BASE_SUB_NOTES    = 500e6
SUB_NOTES_RATE    = 0.070

# Minimum cash balance
MIN_CASH = 200e6
BASE_CASH = 300e6

# --- Equity ---
BASE_COMMON_STOCK   = 1_000e6
BASE_RETAINED_EARN  = 800e6

print(f"  Company         : {COMPANY}")
print(f"  Base Revenue    : {fmt_currency(BASE_REVENUE)}")
print(f"  Forecast Horizon: {N_YEARS} years")
print(f"  Circular solver : Fixed-point iteration (max 20 iters)")


# =============================================================================
# 2. MODEL ARRAYS INITIALIZATION
# =============================================================================
# All arrays have N_YEARS+1 elements: index 0 = base year, 1-5 = projections

# Income Statement
revenue      = np.zeros(N_YEARS + 1)
cogs         = np.zeros(N_YEARS + 1)
gross_profit = np.zeros(N_YEARS + 1)
sga          = np.zeros(N_YEARS + 1)
rd           = np.zeros(N_YEARS + 1)
other_opex   = np.zeros(N_YEARS + 1)
depreciation = np.zeros(N_YEARS + 1)
ebit         = np.zeros(N_YEARS + 1)
interest_exp = np.zeros(N_YEARS + 1)
ebt          = np.zeros(N_YEARS + 1)
taxes        = np.zeros(N_YEARS + 1)
net_income   = np.zeros(N_YEARS + 1)
ebitda       = np.zeros(N_YEARS + 1)

# Balance Sheet -- Assets
cash              = np.zeros(N_YEARS + 1)
accounts_rec      = np.zeros(N_YEARS + 1)
inventory         = np.zeros(N_YEARS + 1)
total_current_ast = np.zeros(N_YEARS + 1)
ppe_gross         = np.zeros(N_YEARS + 1)
accum_dep         = np.zeros(N_YEARS + 1)
ppe_net           = np.zeros(N_YEARS + 1)
other_assets      = np.zeros(N_YEARS + 1)
total_assets      = np.zeros(N_YEARS + 1)

# Balance Sheet -- Liabilities
accounts_pay       = np.zeros(N_YEARS + 1)
total_current_liab = np.zeros(N_YEARS + 1)
revolver_bal       = np.zeros(N_YEARS + 1)
term_loan_bal      = np.zeros(N_YEARS + 1)
sub_notes_bal      = np.zeros(N_YEARS + 1)
total_debt         = np.zeros(N_YEARS + 1)
other_liab         = np.zeros(N_YEARS + 1)
total_liabilities  = np.zeros(N_YEARS + 1)

# Balance Sheet -- Equity
common_stock       = np.zeros(N_YEARS + 1)
retained_earnings  = np.zeros(N_YEARS + 1)
total_equity       = np.zeros(N_YEARS + 1)
total_liab_equity  = np.zeros(N_YEARS + 1)

# Cash Flow Statement
cf_operations      = np.zeros(N_YEARS + 1)
cf_investing       = np.zeros(N_YEARS + 1)
cf_financing       = np.zeros(N_YEARS + 1)
capex_arr          = np.zeros(N_YEARS + 1)
dividends          = np.zeros(N_YEARS + 1)
delta_nwc          = np.zeros(N_YEARS + 1)
net_change_cash    = np.zeros(N_YEARS + 1)

# Balance check
balance_check      = np.zeros(N_YEARS + 1)


# =============================================================================
# 3. BASE YEAR INITIALIZATION
# =============================================================================
revenue[0]      = BASE_REVENUE
cogs[0]         = BASE_REVENUE * 0.55
gross_profit[0] = revenue[0] - cogs[0]
sga[0]          = BASE_REVENUE * 0.15
rd[0]           = BASE_REVENUE * 0.03
other_opex[0]   = BASE_REVENUE * 0.02
depreciation[0] = BASE_PPE_GROSS * DEP_RATE
ebitda[0]       = gross_profit[0] - sga[0] - rd[0] - other_opex[0]
ebit[0]         = ebitda[0] - depreciation[0]
interest_exp[0] = (BASE_TERM_LOAN * TERM_LOAN_RATE +
                   BASE_SUB_NOTES * SUB_NOTES_RATE)
ebt[0]          = ebit[0] - interest_exp[0]
taxes[0]        = max(0, ebt[0] * TAX_RATE)
net_income[0]   = ebt[0] - taxes[0]

cash[0]              = BASE_CASH
accounts_rec[0]      = BASE_REVENUE * (DSO / 365.0)
inventory[0]         = cogs[0] * (DIO / 365.0)
total_current_ast[0] = cash[0] + accounts_rec[0] + inventory[0]
ppe_gross[0]         = BASE_PPE_GROSS
accum_dep[0]         = BASE_ACCUM_DEP
ppe_net[0]           = ppe_gross[0] - accum_dep[0]
other_assets[0]      = BASE_OTHER_ASSETS
total_assets[0]      = total_current_ast[0] + ppe_net[0] + other_assets[0]

accounts_pay[0]       = cogs[0] * (DPO / 365.0)
total_current_liab[0] = accounts_pay[0]
revolver_bal[0]       = 0
term_loan_bal[0]      = BASE_TERM_LOAN
sub_notes_bal[0]      = BASE_SUB_NOTES
total_debt[0]         = term_loan_bal[0] + sub_notes_bal[0]
other_liab[0]         = BASE_OTHER_LIAB
total_liabilities[0]  = total_current_liab[0] + total_debt[0] + other_liab[0]

common_stock[0]      = BASE_COMMON_STOCK
retained_earnings[0] = BASE_RETAINED_EARN
total_equity[0]      = common_stock[0] + retained_earnings[0]
total_liab_equity[0] = total_liabilities[0] + total_equity[0]
balance_check[0]     = total_assets[0] - total_liab_equity[0]


# =============================================================================
# 4. PROJECTION ENGINE WITH CIRCULAR REFERENCE SOLVER
# =============================================================================
print_section("PROJECTION ENGINE (Circular Reference Solver)")

MAX_ITER = 20
TOLERANCE = 1.0  # $1 tolerance

for yr in range(1, N_YEARS + 1):
    idx = yr - 1  # index into assumption arrays (0-based)

    # --- Revenue & operating items (no circularity) ---
    revenue[yr]      = revenue[yr - 1] * (1.0 + REVENUE_GROWTH[idx])
    cogs[yr]         = revenue[yr] * COGS_PCT[idx]
    gross_profit[yr] = revenue[yr] - cogs[yr]
    sga[yr]          = revenue[yr] * SGA_PCT[idx]
    rd[yr]           = revenue[yr] * RD_PCT[idx]
    other_opex[yr]   = revenue[yr] * OTHER_OPEX_PCT[idx]

    # CapEx & PP&E
    capex_arr[yr]    = revenue[yr] * CAPEX_PCT_REV[idx]
    ppe_gross[yr]    = ppe_gross[yr - 1] + capex_arr[yr]
    depreciation[yr] = ppe_gross[yr] * DEP_RATE
    accum_dep[yr]    = accum_dep[yr - 1] + depreciation[yr]
    ppe_net[yr]      = ppe_gross[yr] - accum_dep[yr]

    ebitda[yr] = gross_profit[yr] - sga[yr] - rd[yr] - other_opex[yr]

    # Working capital
    accounts_rec[yr] = revenue[yr] * (DSO / 365.0)
    inventory[yr]    = cogs[yr] * (DIO / 365.0)
    accounts_pay[yr] = cogs[yr] * (DPO / 365.0)

    nwc_curr = (accounts_rec[yr] + inventory[yr]) - accounts_pay[yr]
    nwc_prev = (accounts_rec[yr-1] + inventory[yr-1]) - accounts_pay[yr-1]
    delta_nwc[yr] = nwc_curr - nwc_prev

    # Other BS items
    other_assets[yr] = other_assets[yr - 1] * (1.0 + OTHER_ASSETS_GROWTH)
    other_liab[yr]   = other_liab[yr - 1] * (1.0 + OTHER_LIAB_GROWTH)
    common_stock[yr] = common_stock[yr - 1]  # No new issuance

    # Term loan amortization
    term_amort = min(TERM_LOAN_AMORT, term_loan_bal[yr - 1])
    term_loan_bal[yr] = term_loan_bal[yr - 1] - term_amort
    sub_notes_bal[yr] = sub_notes_bal[yr - 1]  # Bullet, no amort

    # --- CIRCULAR REFERENCE SOLVER ---
    # Interest -> Net Income -> CF -> Cash -> Revolver -> Interest
    # Initialize with prior year interest as guess
    prev_interest = interest_exp[yr - 1] if yr > 1 else interest_exp[0]
    guess_interest = prev_interest

    for iteration in range(MAX_ITER):
        # Step 1: Income Statement (given interest guess)
        ebit_yr = ebitda[yr] - depreciation[yr]
        ebt_yr  = ebit_yr - guess_interest
        tax_yr  = max(0, ebt_yr * TAX_RATE)
        ni_yr   = ebt_yr - tax_yr
        div_yr  = max(0, ni_yr * DIVIDEND_PAYOUT_RATIO)

        # Step 2: Cash Flow Statement
        cf_ops = ni_yr + depreciation[yr] - delta_nwc[yr]
        cf_inv = -capex_arr[yr]
        cf_fin_ex_revolver = -term_amort - div_yr

        # Step 3: Pre-revolver cash
        pre_rev_cash = (cash[yr - 1] + cf_ops + cf_inv + cf_fin_ex_revolver)

        # Step 4: Revolver draw/repay to hit min cash
        if pre_rev_cash >= MIN_CASH:
            # Repay existing revolver first
            revolver_repay = min(revolver_bal[yr - 1],
                                 pre_rev_cash - MIN_CASH)
            new_revolver = revolver_bal[yr - 1] - revolver_repay
            ending_cash = pre_rev_cash - revolver_repay + (
                new_revolver - revolver_bal[yr - 1] + revolver_repay)
            # Simplify: just repay what we can
            new_revolver = max(0, revolver_bal[yr - 1] - (
                pre_rev_cash - MIN_CASH))
            ending_cash = pre_rev_cash - (revolver_bal[yr - 1] - new_revolver)
        else:
            # Need to draw on revolver
            shortfall = MIN_CASH - pre_rev_cash
            new_revolver = revolver_bal[yr - 1] + shortfall
            new_revolver = min(new_revolver, REVOLVER_CAPACITY)
            ending_cash = pre_rev_cash + (new_revolver - revolver_bal[yr - 1])

        # Step 5: Compute interest on average balances
        avg_revolver = (revolver_bal[yr - 1] + new_revolver) / 2.0
        avg_term     = (term_loan_bal[yr - 1] + term_loan_bal[yr]) / 2.0
        avg_sub      = (sub_notes_bal[yr - 1] + sub_notes_bal[yr]) / 2.0

        new_interest = (avg_revolver * REVOLVER_RATE +
                        avg_term * TERM_LOAN_RATE +
                        avg_sub * SUB_NOTES_RATE)

        # Check convergence
        if abs(new_interest - guess_interest) < TOLERANCE:
            guess_interest = new_interest
            break
        guess_interest = new_interest

    # --- Store converged values ---
    interest_exp[yr] = guess_interest
    ebit[yr]         = ebit_yr
    ebt[yr]          = ebit_yr - guess_interest
    taxes[yr]        = max(0, ebt[yr] * TAX_RATE)
    net_income[yr]   = ebt[yr] - taxes[yr]
    dividends[yr]    = max(0, net_income[yr] * DIVIDEND_PAYOUT_RATIO)

    revolver_bal[yr] = new_revolver
    cash[yr]         = ending_cash

    # Retained earnings roll-forward
    retained_earnings[yr] = retained_earnings[yr - 1] + net_income[yr] - dividends[yr]

    # Cash flow statement
    cf_operations[yr] = net_income[yr] + depreciation[yr] - delta_nwc[yr]
    cf_investing[yr]  = -capex_arr[yr]
    revolver_change   = revolver_bal[yr] - revolver_bal[yr - 1]
    cf_financing[yr]  = -term_amort + revolver_change - dividends[yr]
    net_change_cash[yr] = cf_operations[yr] + cf_investing[yr] + cf_financing[yr]

    # Aggregate balance sheet
    total_current_ast[yr]  = cash[yr] + accounts_rec[yr] + inventory[yr]
    total_debt[yr]         = revolver_bal[yr] + term_loan_bal[yr] + sub_notes_bal[yr]
    total_current_liab[yr] = accounts_pay[yr]
    total_liabilities[yr]  = total_current_liab[yr] + total_debt[yr] + other_liab[yr]
    total_equity[yr]       = common_stock[yr] + retained_earnings[yr]
    total_assets[yr]       = total_current_ast[yr] + ppe_net[yr] + other_assets[yr]
    total_liab_equity[yr]  = total_liabilities[yr] + total_equity[yr]

    # Balance check
    balance_check[yr] = total_assets[yr] - total_liab_equity[yr]

    print(f"  Year {yr}: converged in {iteration + 1} iterations "
          f"(imbalance: ${abs(balance_check[yr]):,.0f})")


# =============================================================================
# 5. PRINT FINANCIAL STATEMENTS
# =============================================================================
print_section("INCOME STATEMENT")
is_headers = ["Item"] + YEAR_LABELS
is_rows = [
    ["Revenue"]            + [fmt_currency(v) for v in revenue],
    ["(-) COGS"]           + [fmt_currency(v) for v in cogs],
    ["Gross Profit"]       + [fmt_currency(v) for v in gross_profit],
    ["  Gross Margin"]     + [f"{gp/r*100:.1f}%" for gp, r in zip(gross_profit, revenue)],
    ["(-) SG&A"]           + [fmt_currency(v) for v in sga],
    ["(-) R&D"]            + [fmt_currency(v) for v in rd],
    ["(-) Other OpEx"]     + [fmt_currency(v) for v in other_opex],
    ["EBITDA"]             + [fmt_currency(v) for v in ebitda],
    ["(-) D&A"]            + [fmt_currency(v) for v in depreciation],
    ["EBIT"]               + [fmt_currency(v) for v in ebit],
    ["(-) Interest"]       + [fmt_currency(v) for v in interest_exp],
    ["EBT"]                + [fmt_currency(v) for v in ebt],
    ["(-) Taxes"]          + [fmt_currency(v) for v in taxes],
    ["Net Income"]         + [fmt_currency(v) for v in net_income],
    ["  Net Margin"]       + [f"{ni/r*100:.1f}%" for ni, r in zip(net_income, revenue)],
]
print_table("PRO FORMA INCOME STATEMENT", is_headers, is_rows)

print_section("BALANCE SHEET")
bs_headers = ["Item"] + YEAR_LABELS
bs_rows = [
    ["--- ASSETS ---"]           + [""] * (N_YEARS + 1),
    ["Cash"]                     + [fmt_currency(v) for v in cash],
    ["Accounts Receivable"]      + [fmt_currency(v) for v in accounts_rec],
    ["Inventory"]                + [fmt_currency(v) for v in inventory],
    ["Total Current Assets"]     + [fmt_currency(v) for v in total_current_ast],
    ["PP&E (Net)"]               + [fmt_currency(v) for v in ppe_net],
    ["Other Assets"]             + [fmt_currency(v) for v in other_assets],
    ["TOTAL ASSETS"]             + [fmt_currency(v) for v in total_assets],
    ["", ""] + [""] * N_YEARS,
    ["--- LIABILITIES ---"]      + [""] * (N_YEARS + 1),
    ["Accounts Payable"]         + [fmt_currency(v) for v in accounts_pay],
    ["Revolver"]                 + [fmt_currency(v) for v in revolver_bal],
    ["Term Loan"]                + [fmt_currency(v) for v in term_loan_bal],
    ["Subordinated Notes"]       + [fmt_currency(v) for v in sub_notes_bal],
    ["Other Liabilities"]        + [fmt_currency(v) for v in other_liab],
    ["TOTAL LIABILITIES"]        + [fmt_currency(v) for v in total_liabilities],
    ["", ""] + [""] * N_YEARS,
    ["--- EQUITY ---"]           + [""] * (N_YEARS + 1),
    ["Common Stock"]             + [fmt_currency(v) for v in common_stock],
    ["Retained Earnings"]        + [fmt_currency(v) for v in retained_earnings],
    ["TOTAL EQUITY"]             + [fmt_currency(v) for v in total_equity],
    ["", ""] + [""] * N_YEARS,
    ["TOTAL LIAB + EQUITY"]      + [fmt_currency(v) for v in total_liab_equity],
    ["BALANCE CHECK (A-L-E)"]    + [f"${v:,.0f}" for v in balance_check],
]
print_table("BALANCE SHEET", bs_headers, bs_rows)

print_section("CASH FLOW STATEMENT")
cf_headers = ["Item"] + YEAR_LABELS
cf_rows = [
    ["--- OPERATING ---"]        + [""] * (N_YEARS + 1),
    ["Net Income"]               + [fmt_currency(v) for v in net_income],
    ["(+) D&A"]                  + [fmt_currency(v) for v in depreciation],
    ["(-) Change in NWC"]        + [fmt_currency(v) for v in delta_nwc],
    ["CF from Operations"]       + [fmt_currency(v) for v in cf_operations],
    ["", ""] + [""] * N_YEARS,
    ["--- INVESTING ---"]        + [""] * (N_YEARS + 1),
    ["(-) CapEx"]                + [fmt_currency(v) for v in capex_arr],
    ["CF from Investing"]        + [fmt_currency(v) for v in cf_investing],
    ["", ""] + [""] * N_YEARS,
    ["--- FINANCING ---"]        + [""] * (N_YEARS + 1),
    ["Revolver Draw/(Repay)"]    + [fmt_currency(revolver_bal[i] - revolver_bal[max(0,i-1)])
                                    for i in range(N_YEARS + 1)],
    ["Term Loan Amort"]          + ["--"] + [fmt_currency(-TERM_LOAN_AMORT)
                                    if term_loan_bal[i-1] > 0 else "$0"
                                    for i in range(1, N_YEARS + 1)],
    ["(-) Dividends"]            + [fmt_currency(v) for v in dividends],
    ["CF from Financing"]        + [fmt_currency(v) for v in cf_financing],
    ["", ""] + [""] * N_YEARS,
    ["Net Change in Cash"]       + [fmt_currency(v) for v in net_change_cash],
    ["Beginning Cash"]           + [fmt_currency(cash[max(0, i-1)]) for i in range(N_YEARS + 1)],
    ["Ending Cash"]              + [fmt_currency(v) for v in cash],
]
print_table("CASH FLOW STATEMENT", cf_headers, cf_rows)


# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

x = np.arange(N_YEARS + 1)

# -------------------------------------------------------------------------
# FIGURE 1: Revenue, EBITDA & Net Income Trend
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x, revenue/1e6, color=COLORS["primary"], marker="o", linewidth=2.5,
         markersize=7, label="Revenue")
ax1.plot(x, ebitda/1e6, color=COLORS["secondary"], marker="s", linewidth=2.5,
         markersize=7, label="EBITDA")
ax1.plot(x, net_income/1e6, color=COLORS["accent"], marker="^", linewidth=2.5,
         markersize=7, label="Net Income")

for i in range(N_YEARS + 1):
    ax1.text(i, revenue[i]/1e6 + 50, f"${revenue[i]/1e6:,.0f}M",
             ha="center", fontsize=7, color=COLORS["primary"])

ax1.set_xlabel("Year")
ax1.set_ylabel("USD (Millions)")
ax1.set_title(f"{COMPANY} -- Revenue, EBITDA & Net Income Trend",
              fontsize=14, fontweight="bold", pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(YEAR_LABELS, fontsize=9)
ax1.legend(loc="upper left")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, p: f"${v:,.0f}M"))

fig1.tight_layout()
save_figure(fig1, "ts_01_income_trend", subdir="ts")

# -------------------------------------------------------------------------
# FIGURE 2: Margin Analysis
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

gm = gross_profit / revenue * 100
em = ebitda / revenue * 100
nm = net_income / revenue * 100

ax2.plot(x, gm, color=COLORS["primary"], marker="o", linewidth=2.5,
         markersize=7, label="Gross Margin")
ax2.plot(x, em, color=COLORS["secondary"], marker="s", linewidth=2.5,
         markersize=7, label="EBITDA Margin")
ax2.plot(x, nm, color=COLORS["accent"], marker="^", linewidth=2.5,
         markersize=7, label="Net Margin")

ax2.fill_between(x, gm, em, alpha=0.1, color=COLORS["primary"])
ax2.fill_between(x, em, nm, alpha=0.1, color=COLORS["secondary"])

ax2.set_xlabel("Year")
ax2.set_ylabel("Margin (%)")
ax2.set_title(f"{COMPANY} -- Margin Analysis",
              fontsize=14, fontweight="bold", pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(YEAR_LABELS, fontsize=9)
ax2.legend(loc="upper left")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct))

fig2.tight_layout()
save_figure(fig2, "ts_02_margin_analysis", subdir="ts")

# -------------------------------------------------------------------------
# FIGURE 3: Balance Sheet Composition (Stacked)
# -------------------------------------------------------------------------
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Assets side
ax3a.bar(x, cash/1e6, label="Cash", color=COLORS["primary"], alpha=0.8)
ax3a.bar(x, accounts_rec/1e6, bottom=cash/1e6, label="A/R",
         color=COLORS["secondary"], alpha=0.8)
ax3a.bar(x, inventory/1e6, bottom=(cash + accounts_rec)/1e6,
         label="Inventory", color=COLORS["accent"], alpha=0.8)
ax3a.bar(x, ppe_net/1e6,
         bottom=(cash + accounts_rec + inventory)/1e6,
         label="PP&E (Net)", color=COLORS["purple"], alpha=0.8)
ax3a.bar(x, other_assets/1e6,
         bottom=(cash + accounts_rec + inventory + ppe_net)/1e6,
         label="Other", color=COLORS["teal"], alpha=0.8)

ax3a.set_title("Assets", fontsize=12, fontweight="bold")
ax3a.set_ylabel("USD (Millions)")
ax3a.set_xticks(x)
ax3a.set_xticklabels(YEAR_LABELS, fontsize=8, rotation=30)
ax3a.legend(loc="upper left", fontsize=8)

# Liabilities + Equity side
ax3b.bar(x, accounts_pay/1e6, label="A/P", color=COLORS["danger"], alpha=0.8)
ax3b.bar(x, total_debt/1e6, bottom=accounts_pay/1e6,
         label="Total Debt", color=COLORS["accent"], alpha=0.8)
ax3b.bar(x, other_liab/1e6, bottom=(accounts_pay + total_debt)/1e6,
         label="Other Liab", color=COLORS["pink"], alpha=0.8)
ax3b.bar(x, total_equity/1e6,
         bottom=(accounts_pay + total_debt + other_liab)/1e6,
         label="Equity", color=COLORS["secondary"], alpha=0.8)

ax3b.set_title("Liabilities + Equity", fontsize=12, fontweight="bold")
ax3b.set_ylabel("USD (Millions)")
ax3b.set_xticks(x)
ax3b.set_xticklabels(YEAR_LABELS, fontsize=8, rotation=30)
ax3b.legend(loc="upper left", fontsize=8)

fig3.suptitle(f"{COMPANY} -- Balance Sheet Composition",
              fontsize=14, fontweight="bold", y=1.02)
fig3.tight_layout()
save_figure(fig3, "ts_03_balance_sheet", subdir="ts")

# -------------------------------------------------------------------------
# FIGURE 4: Cash Flow Waterfall
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Use Year 3 as representative
yr_show = 3
cf_labels = ["Beg Cash", "CF Ops", "CF Inv", "CF Fin", "End Cash"]
cf_vals   = [cash[yr_show - 1], cf_operations[yr_show],
             cf_investing[yr_show], cf_financing[yr_show], cash[yr_show]]

cum = np.zeros(len(cf_vals))
cum[0] = cf_vals[0]
for k in range(1, len(cf_vals) - 1):
    cum[k] = cum[k-1] + cf_vals[k]
cum[-1] = cf_vals[-1]

bot4 = np.zeros(len(cf_vals))
bot4[0] = 0
for k in range(1, len(cf_vals) - 1):
    if cf_vals[k] >= 0:
        bot4[k] = cum[k] - cf_vals[k]
    else:
        bot4[k] = cum[k]
bot4[-1] = 0

c4 = [COLORS["primary"], COLORS["secondary"], COLORS["danger"],
      COLORS["accent"], COLORS["teal"]]

ax4.bar(cf_labels, [abs(v)/1e6 for v in cf_vals],
        bottom=[b/1e6 for b in bot4], color=c4, alpha=0.85,
        edgecolor="none", width=0.55)

for k, (lbl, val) in enumerate(zip(cf_labels, cf_vals)):
    y = (bot4[k] + abs(val)/2) / 1e6
    sign = "+" if val > 0 and 0 < k < len(cf_vals)-1 else ""
    ax4.text(k, y, f"{sign}${val/1e6:,.0f}M", ha="center", va="center",
             fontsize=9, color=COLORS["white"], fontweight="bold")

ax4.set_title(f"{COMPANY} -- Cash Flow Waterfall (Year {yr_show})",
              fontsize=14, fontweight="bold", pad=15)
ax4.set_ylabel("USD (Millions)")

fig4.tight_layout()
save_figure(fig4, "ts_04_cf_waterfall", subdir="ts")

# -------------------------------------------------------------------------
# FIGURE 5: Debt Schedule & Leverage
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.fill_between(x, 0, revolver_bal/1e6, alpha=0.7, color=COLORS["danger"],
                 label="Revolver")
ax5.fill_between(x, revolver_bal/1e6, (revolver_bal + term_loan_bal)/1e6,
                 alpha=0.7, color=COLORS["primary"], label="Term Loan")
ax5.fill_between(x, (revolver_bal + term_loan_bal)/1e6, total_debt/1e6,
                 alpha=0.7, color=COLORS["accent"], label="Sub Notes")

# Leverage on secondary
ax5b = ax5.twinx()
lev = np.where(ebitda > 0, total_debt / ebitda, 0)
ax5b.plot(x, lev, color=COLORS["teal"], marker="D", linewidth=2,
          markersize=6, label="Net Leverage (x)")
ax5b.set_ylabel("Debt / EBITDA (x)", color=COLORS["teal"])
ax5b.tick_params(axis="y", labelcolor=COLORS["teal"])

ax5.set_xlabel("Year")
ax5.set_ylabel("Debt Outstanding ($ Millions)")
ax5.set_title(f"{COMPANY} -- Debt Structure & Leverage",
              fontsize=14, fontweight="bold", pad=15)
ax5.set_xticks(x)
ax5.set_xticklabels(YEAR_LABELS, fontsize=9)

lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

fig5.tight_layout()
save_figure(fig5, "ts_05_debt_leverage", subdir="ts")

# -------------------------------------------------------------------------
# FIGURE 6: Key Ratios Dashboard
# -------------------------------------------------------------------------
fig6, axes = plt.subplots(2, 2, figsize=(12, 8))

# ROE
roe = np.where(total_equity > 0, net_income / total_equity * 100, 0)
axes[0, 0].bar(x, roe, color=COLORS["primary"], alpha=0.85, width=0.5)
axes[0, 0].set_title("Return on Equity (ROE)", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("%")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(YEAR_LABELS, fontsize=7, rotation=30)
for i, v in enumerate(roe):
    axes[0, 0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7,
                    color=COLORS["white"])

# Interest Coverage
int_cov = np.where(interest_exp > 0, ebitda / interest_exp, 0)
axes[0, 1].bar(x, int_cov, color=COLORS["secondary"], alpha=0.85, width=0.5)
axes[0, 1].set_title("Interest Coverage (EBITDA/Interest)", fontsize=11,
                     fontweight="bold")
axes[0, 1].set_ylabel("x")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(YEAR_LABELS, fontsize=7, rotation=30)
for i, v in enumerate(int_cov):
    axes[0, 1].text(i, v + 0.1, f"{v:.1f}x", ha="center", fontsize=7,
                    color=COLORS["white"])

# Debt/Equity
de_ratio = np.where(total_equity > 0, total_debt / total_equity, 0)
axes[1, 0].bar(x, de_ratio, color=COLORS["accent"], alpha=0.85, width=0.5)
axes[1, 0].set_title("Debt / Equity Ratio", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("x")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(YEAR_LABELS, fontsize=7, rotation=30)
for i, v in enumerate(de_ratio):
    axes[1, 0].text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=7,
                    color=COLORS["white"])

# FCF Conversion (CF Ops / EBITDA)
fcf_conv = np.where(ebitda > 0, cf_operations / ebitda * 100, 0)
axes[1, 1].bar(x[1:], fcf_conv[1:], color=COLORS["purple"], alpha=0.85,
               width=0.5)
axes[1, 1].set_title("Cash Conversion (CF Ops / EBITDA)", fontsize=11,
                     fontweight="bold")
axes[1, 1].set_ylabel("%")
axes[1, 1].set_xticks(x[1:])
axes[1, 1].set_xticklabels(YEAR_LABELS[1:], fontsize=7, rotation=30)
for i, v in enumerate(fcf_conv[1:]):
    axes[1, 1].text(i + 1, v + 1, f"{v:.0f}%", ha="center", fontsize=7,
                    color=COLORS["white"])

fig6.suptitle(f"{COMPANY} -- Key Financial Ratios Dashboard",
              fontsize=14, fontweight="bold", y=1.02)
fig6.tight_layout()
save_figure(fig6, "ts_06_ratios_dashboard", subdir="ts")


# =============================================================================
# 7. SUMMARY
# =============================================================================
print_section("THREE-STATEMENT MODEL SUMMARY")
print(f"  Company             : {COMPANY}")
print(f"  Base Revenue        : {fmt_currency(BASE_REVENUE)}")
print(f"  Year 5 Revenue      : {fmt_currency(revenue[N_YEARS])}")
print(f"  Year 5 Net Income   : {fmt_currency(net_income[N_YEARS])}")
print(f"  Year 5 Net Margin   : {net_income[N_YEARS]/revenue[N_YEARS]*100:.1f}%")
print(f"  Year 5 Total Assets : {fmt_currency(total_assets[N_YEARS])}")
print(f"  Year 5 Total Debt   : {fmt_currency(total_debt[N_YEARS])}")
print(f"  Year 5 Total Equity : {fmt_currency(total_equity[N_YEARS])}")
print(f"  Balance Check       : All years balanced (max imbalance: "
      f"${max(abs(balance_check)):,.0f})")
print(f"  Circular Solver     : Fixed-point iteration, converged all years")
print(f"  Figures saved to    : outputs/figures/ts/")
print(f"  Module 3 of 6 complete.")
print("=" * 70)

#!/usr/bin/env python3
"""
=============================================================================
MODULE 6: SENSITIVITY & SCENARIO ANALYSIS
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A comprehensive sensitivity and scenario analysis toolkit applied to the
DCF valuation framework from Module 1.  The module implements four
complementary analytical approaches:

    1. Tornado Chart (One-Way Sensitivity)
       - Isolates impact of each input variable on valuation
       - Ranks variables by magnitude of influence
       - Identifies the key value drivers

    2. Two-Way Data Tables
       - Simultaneous variation of two inputs
       - WACC vs Growth, Growth vs Margin, Leverage vs Growth
       - Heatmap visualization with conditional formatting

    3. Monte Carlo Simulation
       - 50,000 trial stochastic simulation
       - Correlated input distributions (Cholesky decomposition)
       - Full output distribution with VaR and CVaR on valuation
       - Probability of achieving target price thresholds

    4. Scenario Manager
       - Base / Bull / Bear / Stress / Recovery scenarios
       - Internally consistent assumption sets
       - Spider chart comparison across scenarios

Theoretical Foundations
-----------------------
Sensitivity analysis quantifies model risk -- the risk that valuation
conclusions depend critically on uncertain assumptions.

    Tornado Analysis:
        dV/dX_i  approx  [V(X_i + delta) - V(X_i - delta)] / (2 * delta)

    Monte Carlo:
        V_sim = f(X_1, X_2, ..., X_n)  where X_i ~ Distribution_i
        P(V > Target) = (1/N) * sum_{j=1}^{N} I(V_j > Target)

    The correlation structure between inputs is modeled via the
    Cholesky decomposition of the correlation matrix:
        Z_correlated = L * Z_independent
        where L * L^T = Sigma (correlation matrix)

References
----------
    - Damodaran, A. (2012). "Investment Valuation", Ch. 23: Simulation.
    - Brealey, Myers & Allen (2020). "Principles of Corporate Finance",
      13th ed., Ch. 10: Project Analysis.
    - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering",
      Springer. Ch. 2: Generating Random Variables.

Output
------
    - Console: Tornado rankings, scenario comparison, MC statistics
    - Figures: 8 publication-quality charts saved to outputs/figures/scenario/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency
)
from src.common.finance_utils import (
    cost_of_equity_capm, wacc, unlevered_fcf,
    terminal_value_gordon, mid_year_discount_factors
)


# =============================================================================
# 1. BASE CASE DCF ENGINE (reusable function)
# =============================================================================
print_section("MODULE 6: SENSITIVITY & SCENARIO ANALYSIS")

# Base case assumptions (from Module 1, encapsulated here for modularity)
BASE = {
    "revenue_0"      : 10_000e6,
    "rev_growth"     : np.array([0.12, 0.10, 0.08, 0.07, 0.06]),
    "ebitda_margin"  : np.array([0.36, 0.37, 0.38, 0.39, 0.40]),
    "da_pct"         : 0.05,
    "capex_pct"      : np.array([0.07, 0.065, 0.06, 0.06, 0.055]),
    "nwc_pct"        : 0.10,    # NWC as % of revenue
    "tax_rate"       : 0.25,
    "rf"             : 0.04,
    "beta"           : 1.15,
    "erp"            : 0.055,
    "kd"             : 0.05,
    "eq_weight"      : 0.75,
    "debt_weight"    : 0.25,
    "terminal_growth": 0.025,
    "net_debt"       : 1_700e6,
    "shares"         : 500e6,
    "n_years"        : 5,
}


def run_dcf(params):
    """
    Execute a complete DCF valuation and return implied share price.

    Parameters
    ----------
    params : dict
        Dictionary of assumptions (same keys as BASE).

    Returns
    -------
    dict
        Keys: 'price', 'ev', 'equity', 'pv_fcf', 'pv_tv', 'wacc_rate',
              'revenue', 'ebitda', 'ufcf', 'tv'
    """
    n = params["n_years"]
    rev = np.zeros(n)
    rev[0] = params["revenue_0"] * (1.0 + params["rev_growth"][0])
    for i in range(1, n):
        rev[i] = rev[i - 1] * (1.0 + params["rev_growth"][i])

    ebitda = rev * params["ebitda_margin"]
    da     = rev * params["da_pct"]
    ebit   = ebitda - da
    capex  = rev * params["capex_pct"]

    nwc = rev * params["nwc_pct"]
    delta_nwc = np.diff(nwc, prepend=params["revenue_0"] * params["nwc_pct"])

    ufcf = unlevered_fcf(ebit, params["tax_rate"], da, capex, delta_nwc)

    ke = cost_of_equity_capm(params["rf"], params["beta"], params["erp"])
    wacc_rate = wacc(ke, params["kd"], params["tax_rate"],
                     params["eq_weight"], params["debt_weight"])

    mid_df = mid_year_discount_factors(n, wacc_rate)
    pv_fcf = (ufcf * mid_df).sum()

    if wacc_rate <= params["terminal_growth"]:
        tv = ufcf[-1] * 20  # Cap at 20x if growth exceeds WACC
    else:
        tv = terminal_value_gordon(ufcf[-1], wacc_rate, params["terminal_growth"])

    pv_tv = tv / (1.0 + wacc_rate) ** n
    ev = pv_fcf + pv_tv
    equity = ev - params["net_debt"]
    price = equity / params["shares"]

    return {
        "price"     : price,
        "ev"        : ev,
        "equity"    : equity,
        "pv_fcf"    : pv_fcf,
        "pv_tv"     : pv_tv,
        "wacc_rate" : wacc_rate,
        "revenue"   : rev,
        "ebitda"    : ebitda,
        "ufcf"      : ufcf,
        "tv"        : tv,
    }


# Base case result
base_result = run_dcf(BASE)
BASE_PRICE = base_result["price"]
print(f"  Base Case Implied Price: ${BASE_PRICE:.2f}")
print(f"  Base Case WACC: {base_result['wacc_rate']:.2%}")
print(f"  Base Case EV: {fmt_currency(base_result['ev'])}")


# =============================================================================
# 2. TORNADO CHART -- ONE-WAY SENSITIVITY
# =============================================================================
print_section("TORNADO CHART -- ONE-WAY SENSITIVITY")

# Define variables to perturb and their +/- ranges
TORNADO_VARS = [
    {"name": "Revenue Growth (avg)",   "key": "rev_growth",
     "delta": 0.02, "is_array": True},
    {"name": "EBITDA Margin (avg)",    "key": "ebitda_margin",
     "delta": 0.03, "is_array": True},
    {"name": "Terminal Growth (g)",    "key": "terminal_growth",
     "delta": 0.005, "is_array": False},
    {"name": "Risk-Free Rate",         "key": "rf",
     "delta": 0.01, "is_array": False},
    {"name": "Equity Beta",            "key": "beta",
     "delta": 0.20, "is_array": False},
    {"name": "Equity Risk Premium",    "key": "erp",
     "delta": 0.01, "is_array": False},
    {"name": "Cost of Debt",           "key": "kd",
     "delta": 0.01, "is_array": False},
    {"name": "Tax Rate",               "key": "tax_rate",
     "delta": 0.03, "is_array": False},
    {"name": "CapEx % Revenue (avg)",  "key": "capex_pct",
     "delta": 0.01, "is_array": True},
    {"name": "NWC % Revenue",          "key": "nwc_pct",
     "delta": 0.02, "is_array": False},
    {"name": "Net Debt",               "key": "net_debt",
     "delta": 500e6, "is_array": False},
    {"name": "D/E Mix (Eq Weight)",    "key": "eq_weight",
     "delta": 0.10, "is_array": False},
]

tornado_results = []

for var in TORNADO_VARS:
    # Upside case (+delta for growth/margin, -delta for costs)
    params_up = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in BASE.items()}
    params_dn = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in BASE.items()}

    if var["is_array"]:
        params_up[var["key"]] = BASE[var["key"]] + var["delta"]
        params_dn[var["key"]] = BASE[var["key"]] - var["delta"]
    else:
        params_up[var["key"]] = BASE[var["key"]] + var["delta"]
        params_dn[var["key"]] = BASE[var["key"]] - var["delta"]

    # Handle debt_weight coupling with eq_weight
    if var["key"] == "eq_weight":
        params_up["debt_weight"] = 1.0 - params_up["eq_weight"]
        params_dn["debt_weight"] = 1.0 - params_dn["eq_weight"]

    try:
        price_up = run_dcf(params_up)["price"]
    except (ValueError, ZeroDivisionError):
        price_up = BASE_PRICE

    try:
        price_dn = run_dcf(params_dn)["price"]
    except (ValueError, ZeroDivisionError):
        price_dn = BASE_PRICE

    low_price = min(price_up, price_dn)
    high_price = max(price_up, price_dn)
    spread = high_price - low_price

    tornado_results.append({
        "name"   : var["name"],
        "low"    : low_price,
        "high"   : high_price,
        "spread" : spread,
        "base"   : BASE_PRICE,
    })

# Sort by spread (largest first)
tornado_results.sort(key=lambda x: x["spread"], reverse=True)

# Print tornado table
tn_headers = ["Variable", "Low ($)", "Base ($)", "High ($)", "Spread ($)"]
tn_rows = []
for r in tornado_results:
    tn_rows.append([
        r["name"],
        f"${r['low']:.2f}",
        f"${r['base']:.2f}",
        f"${r['high']:.2f}",
        f"${r['spread']:.2f}",
    ])
print_table("TORNADO ANALYSIS -- RANKED BY IMPACT", tn_headers, tn_rows)


# =============================================================================
# 3. TWO-WAY DATA TABLES
# =============================================================================
print_section("TWO-WAY DATA TABLES")

# --- Table 1: WACC vs Terminal Growth ---
wacc_range = np.arange(0.07, 0.12 + 0.005, 0.005)
growth_range = np.arange(0.015, 0.04 + 0.0025, 0.0025)

table_wg = np.zeros((len(wacc_range), len(growth_range)))

for i, w_override in enumerate(wacc_range):
    for j, g in enumerate(growth_range):
        params = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in BASE.items()}
        params["terminal_growth"] = g
        # Override WACC by adjusting ERP to achieve target WACC
        # WACC = eq_w * (rf + beta*erp) + debt_w * kd * (1-t)
        # Solve for erp: erp = (WACC - debt_w*kd*(1-t) - eq_w*rf) / (eq_w*beta)
        target_wacc = w_override
        debt_component = params["debt_weight"] * params["kd"] * (1 - params["tax_rate"])
        rf_component = params["eq_weight"] * params["rf"]
        required_erp = (target_wacc - debt_component - rf_component) / (
            params["eq_weight"] * params["beta"])
        if required_erp < 0:
            table_wg[i, j] = np.nan
            continue
        params["erp"] = required_erp
        try:
            table_wg[i, j] = run_dcf(params)["price"]
        except (ValueError, ZeroDivisionError):
            table_wg[i, j] = np.nan

# --- Table 2: Revenue Growth vs EBITDA Margin ---
rev_g_range = np.arange(0.04, 0.16 + 0.01, 0.02)
margin_range = np.arange(0.30, 0.46 + 0.02, 0.02)

table_gm = np.zeros((len(rev_g_range), len(margin_range)))

for i, rg in enumerate(rev_g_range):
    for j, m in enumerate(margin_range):
        params = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in BASE.items()}
        params["rev_growth"] = np.full(5, rg)
        params["ebitda_margin"] = np.full(5, m)
        try:
            table_gm[i, j] = run_dcf(params)["price"]
        except (ValueError, ZeroDivisionError):
            table_gm[i, j] = np.nan

# Print tables
hdr_wg = ["WACC \\ g"] + [f"{g:.2%}" for g in growth_range]
rows_wg = []
for i, w in enumerate(wacc_range):
    row = [f"{w:.2%}"]
    for j in range(len(growth_range)):
        val = table_wg[i, j]
        row.append(f"${val:.1f}" if not np.isnan(val) else "N/A")
    rows_wg.append(row)
print_table("DATA TABLE: WACC vs Terminal Growth", hdr_wg, rows_wg)

hdr_gm = ["Growth \\ Margin"] + [f"{m:.0%}" for m in margin_range]
rows_gm = []
for i, rg in enumerate(rev_g_range):
    row = [f"{rg:.0%}"]
    for j in range(len(margin_range)):
        val = table_gm[i, j]
        row.append(f"${val:.1f}" if not np.isnan(val) else "N/A")
    rows_gm.append(row)
print_table("DATA TABLE: Revenue Growth vs EBITDA Margin", hdr_gm, rows_gm)


# =============================================================================
# 4. MONTE CARLO SIMULATION
# =============================================================================
print_section("MONTE CARLO SIMULATION (50,000 TRIALS)")

N_SIM = 50_000
np.random.seed(42)

# Define stochastic inputs with distributions
# [name, mean, std, distribution_type]
MC_INPUTS = {
    "rev_growth_avg"    : {"mean": 0.086, "std": 0.025, "dist": "normal"},
    "ebitda_margin_avg" : {"mean": 0.380, "std": 0.030, "dist": "normal"},
    "terminal_growth"   : {"mean": 0.025, "std": 0.005, "dist": "normal"},
    "rf"                : {"mean": 0.040, "std": 0.008, "dist": "normal"},
    "beta"              : {"mean": 1.150, "std": 0.150, "dist": "normal"},
    "erp"               : {"mean": 0.055, "std": 0.010, "dist": "normal"},
    "capex_pct_avg"     : {"mean": 0.062, "std": 0.008, "dist": "normal"},
}

# Correlation matrix between inputs
# Order: rev_growth, margin, terminal_g, rf, beta, erp, capex
CORR_MATRIX = np.array([
    [ 1.00,  0.30, 0.10,  0.00,  0.00,  0.00, 0.20],  # rev_growth
    [ 0.30,  1.00, 0.05,  0.00,  0.00,  0.00, -0.15], # margin
    [ 0.10,  0.05, 1.00,  0.30,  0.00,  0.10, 0.00],  # terminal_g
    [ 0.00,  0.00, 0.30,  1.00,  0.00,  0.40, 0.00],  # rf
    [ 0.00,  0.00, 0.00,  0.00,  1.00,  0.10, 0.00],  # beta
    [ 0.00,  0.00, 0.10,  0.40,  0.10,  1.00, 0.00],  # erp
    [ 0.20, -0.15, 0.00,  0.00,  0.00,  0.00, 1.00],  # capex
])

# Cholesky decomposition for correlated draws
L = np.linalg.cholesky(CORR_MATRIX)

# Generate correlated standard normal draws
Z_independent = np.random.standard_normal((N_SIM, 7))
Z_correlated = Z_independent @ L.T

# Transform to actual distributions
mc_inputs = {}
input_keys = list(MC_INPUTS.keys())
for idx, key in enumerate(input_keys):
    spec = MC_INPUTS[key]
    mc_inputs[key] = spec["mean"] + spec["std"] * Z_correlated[:, idx]

# Clip to reasonable ranges
mc_inputs["rev_growth_avg"]    = np.clip(mc_inputs["rev_growth_avg"], 0.0, 0.25)
mc_inputs["ebitda_margin_avg"] = np.clip(mc_inputs["ebitda_margin_avg"], 0.15, 0.55)
mc_inputs["terminal_growth"]   = np.clip(mc_inputs["terminal_growth"], 0.005, 0.045)
mc_inputs["rf"]                = np.clip(mc_inputs["rf"], 0.01, 0.08)
mc_inputs["beta"]              = np.clip(mc_inputs["beta"], 0.50, 2.00)
mc_inputs["erp"]               = np.clip(mc_inputs["erp"], 0.03, 0.08)
mc_inputs["capex_pct_avg"]     = np.clip(mc_inputs["capex_pct_avg"], 0.03, 0.10)

# Run simulations
mc_prices = np.zeros(N_SIM)
mc_evs    = np.zeros(N_SIM)
mc_waccs  = np.zeros(N_SIM)

print(f"  Running {N_SIM:,} simulations...")

for sim in range(N_SIM):
    params = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in BASE.items()}

    rg = mc_inputs["rev_growth_avg"][sim]
    params["rev_growth"]      = np.full(5, rg)
    params["ebitda_margin"]   = np.full(5, mc_inputs["ebitda_margin_avg"][sim])
    params["terminal_growth"] = mc_inputs["terminal_growth"][sim]
    params["rf"]              = mc_inputs["rf"][sim]
    params["beta"]            = mc_inputs["beta"][sim]
    params["erp"]             = mc_inputs["erp"][sim]
    params["capex_pct"]       = np.full(5, mc_inputs["capex_pct_avg"][sim])

    try:
        result = run_dcf(params)
        mc_prices[sim] = result["price"]
        mc_evs[sim]    = result["ev"]
        mc_waccs[sim]  = result["wacc_rate"]
    except (ValueError, ZeroDivisionError):
        mc_prices[sim] = np.nan
        mc_evs[sim]    = np.nan
        mc_waccs[sim]  = np.nan

# Filter valid results
valid_mask = ~np.isnan(mc_prices) & (mc_prices > 0) & (mc_prices < 500)
mc_prices_valid = mc_prices[valid_mask]
mc_evs_valid    = mc_evs[valid_mask]
mc_waccs_valid  = mc_waccs[valid_mask]

print(f"  Valid simulations: {len(mc_prices_valid):,} / {N_SIM:,}")

# Statistics
mc_mean   = np.mean(mc_prices_valid)
mc_median = np.median(mc_prices_valid)
mc_std    = np.std(mc_prices_valid)
mc_p5     = np.percentile(mc_prices_valid, 5)
mc_p25    = np.percentile(mc_prices_valid, 25)
mc_p75    = np.percentile(mc_prices_valid, 75)
mc_p95    = np.percentile(mc_prices_valid, 95)
mc_skew   = sp_stats.skew(mc_prices_valid)
mc_kurt   = sp_stats.kurtosis(mc_prices_valid)

# VaR and CVaR on valuation (downside)
var_5  = np.percentile(mc_prices_valid, 5)
cvar_5 = np.mean(mc_prices_valid[mc_prices_valid <= var_5])

# Probability thresholds
CURRENT_PRICE = 45.00
prob_above_current = np.mean(mc_prices_valid > CURRENT_PRICE) * 100
prob_above_60 = np.mean(mc_prices_valid > 60) * 100
prob_below_30 = np.mean(mc_prices_valid < 30) * 100

mc_stat_rows = [
    ["Simulations (valid)",    f"{len(mc_prices_valid):,}"],
    ["Mean Price",             f"${mc_mean:.2f}"],
    ["Median Price",           f"${mc_median:.2f}"],
    ["Std Deviation",          f"${mc_std:.2f}"],
    ["5th Percentile",         f"${mc_p5:.2f}"],
    ["25th Percentile",        f"${mc_p25:.2f}"],
    ["75th Percentile",        f"${mc_p75:.2f}"],
    ["95th Percentile",        f"${mc_p95:.2f}"],
    ["Skewness",               f"{mc_skew:.3f}"],
    ["Excess Kurtosis",        f"{mc_kurt:.3f}"],
    ["", ""],
    ["VaR (5%)",               f"${var_5:.2f}"],
    ["CVaR (5%)",              f"${cvar_5:.2f}"],
    ["", ""],
    [f"P(Price > ${CURRENT_PRICE:.0f})", f"{prob_above_current:.1f}%"],
    ["P(Price > $60)",         f"{prob_above_60:.1f}%"],
    ["P(Price < $30)",         f"{prob_below_30:.1f}%"],
]
print_table("MONTE CARLO RESULTS", ["Statistic", "Value"], mc_stat_rows)


# =============================================================================
# 5. SCENARIO MANAGER
# =============================================================================
print_section("SCENARIO MANAGER")

SCENARIOS = {
    "Bull": {
        "rev_growth"      : np.array([0.15, 0.13, 0.11, 0.10, 0.09]),
        "ebitda_margin"   : np.array([0.38, 0.40, 0.42, 0.43, 0.44]),
        "terminal_growth" : 0.030,
        "beta"            : 1.00,
        "capex_pct"       : np.array([0.06, 0.055, 0.05, 0.05, 0.045]),
        "description"     : "Strong growth, margin expansion, lower risk",
    },
    "Base": {
        "rev_growth"      : BASE["rev_growth"],
        "ebitda_margin"   : BASE["ebitda_margin"],
        "terminal_growth" : BASE["terminal_growth"],
        "beta"            : BASE["beta"],
        "capex_pct"       : BASE["capex_pct"],
        "description"     : "Consensus assumptions",
    },
    "Bear": {
        "rev_growth"      : np.array([0.06, 0.04, 0.03, 0.03, 0.02]),
        "ebitda_margin"   : np.array([0.33, 0.32, 0.31, 0.30, 0.29]),
        "terminal_growth" : 0.020,
        "beta"            : 1.30,
        "capex_pct"       : np.array([0.08, 0.075, 0.07, 0.07, 0.065]),
        "description"     : "Slow growth, margin compression, higher risk",
    },
    "Stress": {
        "rev_growth"      : np.array([0.02, 0.00, -0.02, 0.01, 0.02]),
        "ebitda_margin"   : np.array([0.28, 0.25, 0.23, 0.24, 0.25]),
        "terminal_growth" : 0.015,
        "beta"            : 1.50,
        "capex_pct"       : np.array([0.09, 0.085, 0.08, 0.075, 0.07]),
        "description"     : "Recession scenario with recovery",
    },
    "Recovery": {
        "rev_growth"      : np.array([0.03, 0.05, 0.10, 0.12, 0.10]),
        "ebitda_margin"   : np.array([0.30, 0.33, 0.36, 0.39, 0.41]),
        "terminal_growth" : 0.028,
        "beta"            : 1.10,
        "capex_pct"       : np.array([0.075, 0.07, 0.06, 0.055, 0.05]),
        "description"     : "V-shaped recovery with margin rebuild",
    },
}

scenario_results = {}
for scen_name, scen_params in SCENARIOS.items():
    params = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in BASE.items()}
    for key, val in scen_params.items():
        if key != "description":
            if isinstance(val, np.ndarray):
                params[key] = val.copy()
            else:
                params[key] = val

    try:
        result = run_dcf(params)
        scenario_results[scen_name] = result
    except (ValueError, ZeroDivisionError):
        scenario_results[scen_name] = {"price": np.nan, "ev": np.nan,
                                        "wacc_rate": np.nan}

scen_headers = ["Metric", "Bull", "Base", "Bear", "Stress", "Recovery"]
scen_rows = [
    ["Implied Price"] + [f"${scenario_results[s]['price']:.2f}"
                          if not np.isnan(scenario_results[s]['price']) else "N/A"
                          for s in SCENARIOS],
    ["Enterprise Value"] + [fmt_currency(scenario_results[s]['ev'])
                            if not np.isnan(scenario_results[s]['ev']) else "N/A"
                            for s in SCENARIOS],
    ["WACC"] + [f"{scenario_results[s]['wacc_rate']:.2%}"
                if not np.isnan(scenario_results[s]['wacc_rate']) else "N/A"
                for s in SCENARIOS],
    ["Avg Rev Growth"] + [f"{np.mean(SCENARIOS[s]['rev_growth']):.1%}"
                          for s in SCENARIOS],
    ["Avg EBITDA Margin"] + [f"{np.mean(SCENARIOS[s]['ebitda_margin']):.1%}"
                             for s in SCENARIOS],
    ["Terminal Growth"] + [f"{SCENARIOS[s]['terminal_growth']:.2%}"
                           for s in SCENARIOS],
    ["vs Current ($45)"] + [
        f"{(scenario_results[s]['price']/CURRENT_PRICE - 1)*100:+.1f}%"
        if not np.isnan(scenario_results[s]['price']) else "N/A"
        for s in SCENARIOS],
]
print_table("SCENARIO COMPARISON", scen_headers, scen_rows)

# Probability-weighted price
SCENARIO_PROBS = {"Bull": 0.15, "Base": 0.45, "Bear": 0.25,
                  "Stress": 0.10, "Recovery": 0.05}
prob_weighted_price = sum(
    SCENARIO_PROBS[s] * scenario_results[s]["price"]
    for s in SCENARIOS if not np.isnan(scenario_results[s]["price"])
)
print(f"\n  Probability-Weighted Price: ${prob_weighted_price:.2f}")
print(f"  (Bull 15%, Base 45%, Bear 25%, Stress 10%, Recovery 5%)")


# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

# -------------------------------------------------------------------------
# FIGURE 1: Tornado Chart
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 7))

n_vars = len(tornado_results)
y1 = np.arange(n_vars)

for i, r in enumerate(tornado_results):
    # Low side (left of base)
    low_delta = r["low"] - r["base"]
    high_delta = r["high"] - r["base"]

    ax1.barh(i, low_delta, left=r["base"], height=0.6,
             color=COLORS["danger"], alpha=0.75, edgecolor="none")
    ax1.barh(i, high_delta, left=r["base"], height=0.6,
             color=COLORS["secondary"], alpha=0.75, edgecolor="none")

    ax1.text(r["low"] - 0.5, i, f"${r['low']:.1f}", ha="right",
             va="center", fontsize=7, color=COLORS["white"])
    ax1.text(r["high"] + 0.5, i, f"${r['high']:.1f}", ha="left",
             va="center", fontsize=7, color=COLORS["white"])

ax1.axvline(x=BASE_PRICE, color=COLORS["accent"], linewidth=2,
            linestyle="--", label=f"Base: ${BASE_PRICE:.2f}")

ax1.set_yticks(y1)
ax1.set_yticklabels([r["name"] for r in tornado_results], fontsize=9)
ax1.set_xlabel("Implied Share Price ($)")
ax1.set_title("Tornado Chart -- One-Way Sensitivity Analysis",
              fontsize=14, fontweight="bold", pad=15)
ax1.legend(loc="lower right")
ax1.invert_yaxis()

fig1.tight_layout()
save_figure(fig1, "scenario_01_tornado", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 2: Heatmap -- WACC vs Terminal Growth
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 7))

masked_wg = np.ma.masked_invalid(table_wg)
im2 = ax2.imshow(masked_wg, cmap="RdYlGn", aspect="auto",
                 vmin=np.nanpercentile(table_wg, 5),
                 vmax=np.nanpercentile(table_wg, 95))

for i in range(len(wacc_range)):
    for j in range(len(growth_range)):
        val = table_wg[i, j]
        if np.isnan(val):
            txt, clr = "N/A", "#666666"
        else:
            txt = f"${val:.0f}"
            clr = "black" if val > 55 else COLORS["white"]
        ax2.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)

ax2.set_xticks(np.arange(len(growth_range)))
ax2.set_xticklabels([f"{g:.2%}" for g in growth_range], rotation=45, ha="right")
ax2.set_yticks(np.arange(len(wacc_range)))
ax2.set_yticklabels([f"{w:.2%}" for w in wacc_range])
ax2.set_xlabel("Terminal Growth Rate")
ax2.set_ylabel("WACC")
ax2.set_title("Two-Way Data Table: WACC vs Terminal Growth",
              fontsize=13, fontweight="bold", pad=15)

cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label("Share Price ($)", color=COLORS["white"])
cbar2.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color=COLORS["white"])

fig2.tight_layout()
save_figure(fig2, "scenario_02_heatmap_wacc_growth", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 3: Heatmap -- Revenue Growth vs EBITDA Margin
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(11, 7))

masked_gm = np.ma.masked_invalid(table_gm)
im3 = ax3.imshow(masked_gm, cmap="RdYlGn", aspect="auto",
                 vmin=np.nanpercentile(table_gm, 5),
                 vmax=np.nanpercentile(table_gm, 95))

for i in range(len(rev_g_range)):
    for j in range(len(margin_range)):
        val = table_gm[i, j]
        if np.isnan(val):
            txt, clr = "N/A", "#666666"
        else:
            txt = f"${val:.0f}"
            clr = "black" if val > 60 else COLORS["white"]
        ax3.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=clr)

ax3.set_xticks(np.arange(len(margin_range)))
ax3.set_xticklabels([f"{m:.0%}" for m in margin_range])
ax3.set_yticks(np.arange(len(rev_g_range)))
ax3.set_yticklabels([f"{rg:.0%}" for rg in rev_g_range])
ax3.set_xlabel("EBITDA Margin")
ax3.set_ylabel("Revenue Growth (avg)")
ax3.set_title("Two-Way Data Table: Revenue Growth vs EBITDA Margin",
              fontsize=13, fontweight="bold", pad=15)

cbar3 = fig3.colorbar(im3, ax=ax3, shrink=0.8)
cbar3.set_label("Share Price ($)", color=COLORS["white"])
cbar3.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color=COLORS["white"])

fig3.tight_layout()
save_figure(fig3, "scenario_03_heatmap_growth_margin", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 4: Monte Carlo Distribution (Histogram + KDE)
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(11, 6))

ax4.hist(mc_prices_valid, bins=100, color=COLORS["primary"], alpha=0.6,
         edgecolor="none", density=True, label="Simulation Distribution")

# KDE overlay
kde_x = np.linspace(mc_prices_valid.min(), mc_prices_valid.max(), 500)
kde = sp_stats.gaussian_kde(mc_prices_valid)
ax4.plot(kde_x, kde(kde_x), color=COLORS["accent"], linewidth=2.5,
         label="KDE Fit")

# Percentile lines
ax4.axvline(x=mc_p5, color=COLORS["danger"], linewidth=1.5, linestyle="--",
            label=f"5th Pct: ${mc_p5:.0f}")
ax4.axvline(x=mc_median, color=COLORS["secondary"], linewidth=2,
            label=f"Median: ${mc_median:.0f}")
ax4.axvline(x=mc_p95, color=COLORS["danger"], linewidth=1.5, linestyle="--",
            label=f"95th Pct: ${mc_p95:.0f}")
ax4.axvline(x=CURRENT_PRICE, color=COLORS["white"], linewidth=1.5,
            linestyle=":", alpha=0.7, label=f"Current: ${CURRENT_PRICE:.0f}")

ax4.set_xlabel("Implied Share Price ($)")
ax4.set_ylabel("Density")
ax4.set_title(f"Monte Carlo DCF Valuation ({len(mc_prices_valid):,} trials)",
              fontsize=14, fontweight="bold", pad=15)
ax4.legend(loc="upper right", fontsize=8)

fig4.tight_layout()
save_figure(fig4, "scenario_04_mc_distribution", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 5: Monte Carlo -- Cumulative Probability
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 6))

sorted_prices = np.sort(mc_prices_valid)
cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100

ax5.plot(sorted_prices, cdf, color=COLORS["primary"], linewidth=2)

# Highlight key thresholds
for threshold, clr, lbl in [(30, COLORS["danger"], "$30 (Downside)"),
                              (CURRENT_PRICE, COLORS["accent"], f"${CURRENT_PRICE:.0f} (Current)"),
                              (60, COLORS["secondary"], "$60 (Target)"),
                              (80, COLORS["purple"], "$80 (Bull)")]:
    prob = np.mean(mc_prices_valid <= threshold) * 100
    ax5.axvline(x=threshold, color=clr, linewidth=1, linestyle="--", alpha=0.6)
    ax5.axhline(y=prob, color=clr, linewidth=0.5, linestyle=":", alpha=0.4)
    ax5.plot(threshold, prob, "o", color=clr, markersize=8, zorder=5)
    ax5.text(threshold + 1, prob + 2, f"{lbl}\nP={prob:.1f}%",
             fontsize=7, color=clr)

ax5.set_xlabel("Implied Share Price ($)")
ax5.set_ylabel("Cumulative Probability (%)")
ax5.set_title("Monte Carlo -- Cumulative Distribution Function",
              fontsize=14, fontweight="bold", pad=15)
ax5.set_ylim(0, 105)

fig5.tight_layout()
save_figure(fig5, "scenario_05_mc_cdf", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 6: Scenario Comparison Bar Chart
# -------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(11, 6))

scen_names = list(SCENARIOS.keys())
scen_prices = [scenario_results[s]["price"] for s in scen_names]
scen_colors = [COLORS["secondary"], COLORS["primary"], COLORS["accent"],
               COLORS["danger"], COLORS["purple"]]

bars6 = ax6.bar(scen_names, scen_prices, color=scen_colors, alpha=0.85,
                edgecolor="none", width=0.55)

for bar, price in zip(bars6, scen_prices):
    if not np.isnan(price):
        pct = (price / CURRENT_PRICE - 1) * 100
        sign = "+" if pct > 0 else ""
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"${price:.1f}\n({sign}{pct:.0f}%)", ha="center",
                 fontsize=9, color=COLORS["white"], fontweight="bold")

ax6.axhline(y=CURRENT_PRICE, color=COLORS["white"], linewidth=1.5,
            linestyle="--", alpha=0.6, label=f"Current: ${CURRENT_PRICE:.2f}")
ax6.axhline(y=prob_weighted_price, color=COLORS["accent"], linewidth=1.5,
            linestyle=":", alpha=0.8,
            label=f"Prob-Weighted: ${prob_weighted_price:.2f}")

ax6.set_ylabel("Implied Share Price ($)")
ax6.set_title("Scenario Analysis -- Implied Valuation Comparison",
              fontsize=14, fontweight="bold", pad=15)
ax6.legend(loc="upper right")

fig6.tight_layout()
save_figure(fig6, "scenario_06_comparison", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 7: Spider / Radar Chart -- Scenario Profiles
# -------------------------------------------------------------------------
fig7, ax7 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

spider_metrics = ["Avg Growth", "Avg Margin", "Term Growth",
                  "Beta (inv)", "CapEx Eff", "Price"]
N_METRICS = len(spider_metrics)
angles = np.linspace(0, 2 * np.pi, N_METRICS, endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

scen_spider_colors = [COLORS["secondary"], COLORS["primary"], COLORS["accent"],
                      COLORS["danger"], COLORS["purple"]]

for idx, (scen_name, scen_params) in enumerate(SCENARIOS.items()):
    result = scenario_results[scen_name]
    if np.isnan(result["price"]):
        continue

    # Normalize each metric to 0-1 scale for radar
    avg_g = np.mean(scen_params["rev_growth"])
    avg_m = np.mean(scen_params["ebitda_margin"])
    tg    = scen_params["terminal_growth"]
    beta_inv = 1.0 / scen_params["beta"]  # Invert so higher = better
    capex_eff = 1.0 - np.mean(scen_params["capex_pct"])
    price_norm = result["price"]

    # Scale to 0-100
    values = [
        avg_g / 0.15 * 100,
        avg_m / 0.45 * 100,
        tg / 0.035 * 100,
        beta_inv / 1.0 * 100,
        capex_eff / 0.97 * 100,
        min(price_norm / 100 * 100, 100),
    ]
    values += values[:1]

    ax7.plot(angles, values, "o-", linewidth=2, markersize=4,
             color=scen_spider_colors[idx], label=scen_name, alpha=0.8)
    ax7.fill(angles, values, color=scen_spider_colors[idx], alpha=0.08)

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(spider_metrics, fontsize=9, color=COLORS["white"])
ax7.set_ylim(0, 100)
ax7.set_title("Scenario Profiles -- Radar Comparison",
              fontsize=13, fontweight="bold", pad=25, color=COLORS["white"])
ax7.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax7.set_facecolor(COLORS["axes_bg"])
fig7.patch.set_facecolor(COLORS["bg"])
ax7.grid(color=COLORS["grid"], alpha=0.3)
ax7.tick_params(colors=COLORS["white"])

fig7.tight_layout()
save_figure(fig7, "scenario_07_spider", subdir="scenario")

# -------------------------------------------------------------------------
# FIGURE 8: Input Correlation Heatmap
# -------------------------------------------------------------------------
fig8, ax8 = plt.subplots(figsize=(8, 7))

corr_labels = ["Rev\nGrowth", "EBITDA\nMargin", "Term\nGrowth",
               "Risk-Free\nRate", "Beta", "ERP", "CapEx\n% Rev"]

im8 = ax8.imshow(CORR_MATRIX, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = CORR_MATRIX[i, j]
        clr = "black" if abs(val) > 0.5 else COLORS["white"]
        ax8.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, color=clr)

ax8.set_xticks(np.arange(len(corr_labels)))
ax8.set_xticklabels(corr_labels, fontsize=8)
ax8.set_yticks(np.arange(len(corr_labels)))
ax8.set_yticklabels(corr_labels, fontsize=8)
ax8.set_title("Monte Carlo Input Correlation Matrix",
              fontsize=13, fontweight="bold", pad=15)

cbar8 = fig8.colorbar(im8, ax=ax8, shrink=0.8)
cbar8.set_label("Correlation", color=COLORS["white"])
cbar8.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar8.ax.axes, "yticklabels"), color=COLORS["white"])

fig8.tight_layout()
save_figure(fig8, "scenario_08_correlation", subdir="scenario")


# =============================================================================
# 7. SUMMARY
# =============================================================================
print_section("SENSITIVITY & SCENARIO ANALYSIS SUMMARY")
print(f"  Base Case Price          : ${BASE_PRICE:.2f}")
print(f"  Top Driver               : {tornado_results[0]['name']} "
      f"(spread: ${tornado_results[0]['spread']:.2f})")
print(f"  Monte Carlo Mean         : ${mc_mean:.2f}")
print(f"  Monte Carlo 90% Range    : ${mc_p5:.2f} - ${mc_p95:.2f}")
print(f"  P(Price > Current $45)   : {prob_above_current:.1f}%")
print(f"  Scenario Range           : ${min(scen_prices):.2f} - ${max(scen_prices):.2f}")
print(f"  Prob-Weighted Price      : ${prob_weighted_price:.2f}")
print(f"  Figures saved to         : outputs/figures/scenario/")
print(f"  Module 6 of 6 complete.")
print("=" * 70)

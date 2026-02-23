#!/usr/bin/env python3
"""
=============================================================================
FINANCIAL MODELING UTILITIES
=============================================================================
Core financial calculations shared across all modeling templates:
    - WACC computation
    - Free Cash Flow projection
    - Terminal value (Gordon Growth & Exit Multiple)
    - Debt schedule mechanics
    - Depreciation schedules
    - Working capital modeling
    - Tax shield calculations
    - IRR / MOIC computations

All functions are pure (no side effects) and fully documented.
"""

import numpy as np
from scipy.optimize import brentq


# =============================================================================
# COST OF CAPITAL
# =============================================================================
def cost_of_equity_capm(rf, beta, erp):
    """
    Capital Asset Pricing Model.

    Parameters
    ----------
    rf : float
        Risk-free rate (decimal).
    beta : float
        Equity beta (levered).
    erp : float
        Equity risk premium (decimal).

    Returns
    -------
    float
        Cost of equity (decimal).

    Notes
    -----
    Ke = Rf + Beta * ERP

    Reference: Sharpe (1964), "Capital Asset Prices: A Theory of Market
    Equilibrium under Conditions of Risk", Journal of Finance.
    """
    return rf + beta * erp


def wacc(ke, kd, tax_rate, equity_weight, debt_weight):
    """
    Weighted Average Cost of Capital.

    Parameters
    ----------
    ke : float
        Cost of equity (decimal).
    kd : float
        Pre-tax cost of debt (decimal).
    tax_rate : float
        Marginal corporate tax rate (decimal).
    equity_weight : float
        E / (E + D), market value weights.
    debt_weight : float
        D / (E + D), market value weights.

    Returns
    -------
    float
        WACC (decimal).

    Notes
    -----
    WACC = (E/(E+D)) * Ke + (D/(E+D)) * Kd * (1 - t)

    The tax shield on debt reduces the effective cost of debt financing.
    Market value weights are preferred over book value weights for
    theoretical consistency with Modigliani-Miller (1958).
    """
    return equity_weight * ke + debt_weight * kd * (1.0 - tax_rate)


def unlever_beta(levered_beta, tax_rate, debt_equity_ratio):
    """
    Hamada equation: unlever beta to remove capital structure effects.

    Parameters
    ----------
    levered_beta : float
    tax_rate : float
    debt_equity_ratio : float
        D/E ratio.

    Returns
    -------
    float
        Unlevered (asset) beta.

    Notes
    -----
    Beta_u = Beta_l / [1 + (1 - t) * (D/E)]

    Reference: Hamada, R.S. (1972), "The Effect of the Firm's Capital
    Structure on the Systematic Risk of Common Stocks", Journal of Finance.
    """
    return levered_beta / (1.0 + (1.0 - tax_rate) * debt_equity_ratio)


def relever_beta(unlevered_beta, tax_rate, debt_equity_ratio):
    """
    Hamada equation: relever beta for target capital structure.

    Parameters
    ----------
    unlevered_beta : float
    tax_rate : float
    debt_equity_ratio : float
        Target D/E ratio.

    Returns
    -------
    float
        Relevered beta.
    """
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * debt_equity_ratio)


# =============================================================================
# FREE CASH FLOW
# =============================================================================
def unlevered_fcf(ebit, tax_rate, da, capex, delta_nwc):
    """
    Unlevered Free Cash Flow (FCFF).

    Parameters
    ----------
    ebit : float or np.ndarray
        Earnings Before Interest and Taxes.
    tax_rate : float
        Marginal tax rate.
    da : float or np.ndarray
        Depreciation & Amortization.
    capex : float or np.ndarray
        Capital Expenditures (positive = outflow).
    delta_nwc : float or np.ndarray
        Change in Net Working Capital (positive = cash outflow).

    Returns
    -------
    float or np.ndarray
        Unlevered FCF.

    Notes
    -----
    UFCF = EBIT * (1 - t) + D&A - CapEx - Delta NWC

    This is the cash flow available to all capital providers (debt + equity)
    before any financing cash flows.  It is the correct cash flow to
    discount at WACC in a DCF valuation.
    """
    nopat = ebit * (1.0 - tax_rate)
    return nopat + da - capex - delta_nwc


def levered_fcf(net_income, da, capex, delta_nwc, net_borrowing):
    """
    Levered Free Cash Flow (FCFE).

    Parameters
    ----------
    net_income : float or np.ndarray
    da : float or np.ndarray
    capex : float or np.ndarray
    delta_nwc : float or np.ndarray
    net_borrowing : float or np.ndarray
        New debt issued minus debt repaid.

    Returns
    -------
    float or np.ndarray
        Levered FCF (to equity holders).
    """
    return net_income + da - capex - delta_nwc + net_borrowing


# =============================================================================
# TERMINAL VALUE
# =============================================================================
def terminal_value_gordon(fcf_terminal, wacc_rate, growth_rate):
    """
    Gordon Growth Model terminal value.

    Parameters
    ----------
    fcf_terminal : float
        Final projected year FCF (or FCF * (1+g) for next year).
    wacc_rate : float
        Discount rate (decimal).
    growth_rate : float
        Perpetuity growth rate (decimal).

    Returns
    -------
    float
        Terminal value.

    Notes
    -----
    TV = FCF_n * (1 + g) / (WACC - g)

    The growth rate should not exceed the long-term nominal GDP growth
    rate of the economy (typically 2-3% for developed markets).

    Reference: Gordon, M.J. (1959), "Dividends, Earnings, and Stock Prices",
    Review of Economics and Statistics.
    """
    if wacc_rate <= growth_rate:
        raise ValueError(
            f"WACC ({wacc_rate:.4f}) must exceed growth rate ({growth_rate:.4f}) "
            "for Gordon Growth Model to produce a finite value."
        )
    return fcf_terminal * (1.0 + growth_rate) / (wacc_rate - growth_rate)


def terminal_value_exit_multiple(metric_terminal, multiple):
    """
    Exit Multiple terminal value.

    Parameters
    ----------
    metric_terminal : float
        Terminal year metric (EBITDA, EBIT, Revenue, etc.).
    multiple : float
        Exit multiple (e.g., EV/EBITDA = 10x).

    Returns
    -------
    float
        Terminal value.

    Notes
    -----
    TV = Metric_n * Multiple

    Common multiples: EV/EBITDA (most common), EV/EBIT, EV/Revenue.
    The exit multiple should reflect the long-term equilibrium valuation
    for the industry, not current market conditions.
    """
    return metric_terminal * multiple


# =============================================================================
# DISCOUNTING
# =============================================================================
def discount_cashflows(cashflows, rate, periods=None):
    """
    Discount a series of cash flows to present value.

    Parameters
    ----------
    cashflows : array-like
        Cash flows by period (index 0 = period 1).
    rate : float
        Discount rate per period (decimal).
    periods : array-like, optional
        Custom period numbers. Defaults to 1, 2, ..., n.

    Returns
    -------
    np.ndarray
        Array of present values.
    float
        Sum of present values (total PV).
    """
    cf = np.asarray(cashflows, dtype=float)
    if periods is None:
        periods = np.arange(1, len(cf) + 1, dtype=float)
    else:
        periods = np.asarray(periods, dtype=float)
    discount_factors = 1.0 / (1.0 + rate) ** periods
    pv = cf * discount_factors
    return pv, pv.sum()


def mid_year_discount_factors(n_periods, rate):
    """
    Mid-year convention discount factors.

    Parameters
    ----------
    n_periods : int
    rate : float

    Returns
    -------
    np.ndarray
        Discount factors at mid-year points (0.5, 1.5, ...).

    Notes
    -----
    The mid-year convention assumes cash flows arrive at the midpoint
    of each year rather than at year-end, reflecting a more realistic
    distribution of cash generation throughout the year.
    """
    periods = np.arange(1, n_periods + 1) - 0.5
    return 1.0 / (1.0 + rate) ** periods


# =============================================================================
# IRR & MOIC
# =============================================================================
def compute_irr(cashflows, guess=0.10):
    """
    Internal Rate of Return via Brent's method.

    Parameters
    ----------
    cashflows : array-like
        Cash flows starting at t=0 (initial investment as negative).
    guess : float
        Initial guess for root-finding.

    Returns
    -------
    float
        IRR (decimal), or np.nan if no solution found.

    Notes
    -----
    IRR is the discount rate that makes NPV = 0:
        sum_{t=0}^{N} CF_t / (1 + IRR)^t = 0

    Uses scipy.optimize.brentq for robust root-finding on [-0.99, 10.0].
    """
    cf = np.asarray(cashflows, dtype=float)

    def npv_func(r):
        periods = np.arange(len(cf))
        return np.sum(cf / (1.0 + r) ** periods)

    try:
        return brentq(npv_func, -0.50, 10.0, maxiter=1000)
    except (ValueError, RuntimeError):
        return np.nan


def compute_moic(total_distributions, total_invested):
    """
    Multiple on Invested Capital.

    Parameters
    ----------
    total_distributions : float
        Total cash returned to investors.
    total_invested : float
        Total cash invested (positive).

    Returns
    -------
    float
        MOIC.
    """
    if total_invested == 0:
        return np.inf
    return total_distributions / total_invested


# =============================================================================
# DEPRECIATION SCHEDULES
# =============================================================================
def straight_line_depreciation(cost, salvage, useful_life):
    """
    Straight-line depreciation schedule.

    Parameters
    ----------
    cost : float
        Asset cost.
    salvage : float
        Salvage value at end of useful life.
    useful_life : int
        Useful life in years.

    Returns
    -------
    np.ndarray
        Annual depreciation amounts.
    np.ndarray
        Cumulative depreciation by year-end.
    np.ndarray
        Net book value by year-end.
    """
    annual_dep = (cost - salvage) / useful_life
    dep_schedule = np.full(useful_life, annual_dep)
    cum_dep = np.cumsum(dep_schedule)
    nbv = cost - cum_dep
    return dep_schedule, cum_dep, nbv


# =============================================================================
# WORKING CAPITAL
# =============================================================================
def project_working_capital(revenue, cogs, dso, dio, dpo):
    """
    Project Net Working Capital from operating cycle assumptions.

    Parameters
    ----------
    revenue : np.ndarray
        Revenue by period.
    cogs : np.ndarray
        Cost of Goods Sold by period.
    dso : float
        Days Sales Outstanding.
    dio : float
        Days Inventory Outstanding.
    dpo : float
        Days Payable Outstanding.

    Returns
    -------
    np.ndarray
        Accounts Receivable by period.
    np.ndarray
        Inventory by period.
    np.ndarray
        Accounts Payable by period.
    np.ndarray
        Net Working Capital by period.
    np.ndarray
        Change in NWC by period (positive = cash outflow).

    Notes
    -----
    AR = Revenue * (DSO / 365)
    Inventory = COGS * (DIO / 365)
    AP = COGS * (DPO / 365)
    NWC = AR + Inventory - AP
    Delta NWC = NWC_t - NWC_{t-1}
    """
    ar = revenue * (dso / 365.0)
    inv = cogs * (dio / 365.0)
    ap = cogs * (dpo / 365.0)
    nwc = ar + inv - ap
    delta_nwc = np.diff(nwc, prepend=0)
    return ar, inv, ap, nwc, delta_nwc

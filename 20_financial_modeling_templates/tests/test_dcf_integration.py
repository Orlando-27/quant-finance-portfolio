#!/usr/bin/env python3
"""
=============================================================================
INTEGRATION TEST -- DCF VALUATION PIPELINE
=============================================================================
Verifies the end-to-end DCF pipeline produces reasonable outputs.

Run:
    pytest tests/test_dcf_integration.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.finance_utils import (
    cost_of_equity_capm, wacc, unlevered_fcf,
    terminal_value_gordon, mid_year_discount_factors,
)


def run_dcf_simple(revenue_0, growth, margin, wacc_rate, term_g, net_debt, shares):
    """Simplified DCF for integration testing."""
    n = len(growth)
    rev = np.zeros(n)
    rev[0] = revenue_0 * (1 + growth[0])
    for i in range(1, n):
        rev[i] = rev[i-1] * (1 + growth[i])

    ebitda = rev * margin
    da = rev * 0.05
    ebit = ebitda - da
    capex = rev * 0.06
    nwc = rev * 0.10
    dnwc = np.diff(nwc, prepend=revenue_0 * 0.10)

    ufcf = unlevered_fcf(ebit, 0.25, da, capex, dnwc)
    mid_df = mid_year_discount_factors(n, wacc_rate)
    pv_fcf = (ufcf * mid_df).sum()
    tv = terminal_value_gordon(ufcf[-1], wacc_rate, term_g)
    pv_tv = tv / (1 + wacc_rate) ** n
    ev = pv_fcf + pv_tv
    return (ev - net_debt) / shares


class TestDCFIntegration:
    """Integration tests for DCF valuation pipeline."""

    def test_positive_valuation(self):
        """A profitable company should have positive equity value."""
        price = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.09, 0.025, 1_500e6, 500e6
        )
        assert price > 0

    def test_higher_growth_higher_value(self):
        """Higher growth should lead to higher valuation."""
        low = run_dcf_simple(
            10_000e6, np.full(5, 0.05), 0.35, 0.09, 0.025, 1_500e6, 500e6
        )
        high = run_dcf_simple(
            10_000e6, np.full(5, 0.15), 0.35, 0.09, 0.025, 1_500e6, 500e6
        )
        assert high > low

    def test_higher_wacc_lower_value(self):
        """Higher WACC should reduce valuation."""
        low_wacc = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.08, 0.025, 1_500e6, 500e6
        )
        high_wacc = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.12, 0.025, 1_500e6, 500e6
        )
        assert low_wacc > high_wacc

    def test_higher_margin_higher_value(self):
        """Higher EBITDA margin should increase valuation."""
        low_m = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.25, 0.09, 0.025, 1_500e6, 500e6
        )
        high_m = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.45, 0.09, 0.025, 1_500e6, 500e6
        )
        assert high_m > low_m

    def test_more_debt_lower_equity(self):
        """More net debt reduces equity value per share."""
        low_debt = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.09, 0.025, 500e6, 500e6
        )
        high_debt = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.09, 0.025, 5_000e6, 500e6
        )
        assert low_debt > high_debt

    def test_reasonable_range(self):
        """Typical assumptions should produce price in $10-$200 range."""
        price = run_dcf_simple(
            10_000e6, np.full(5, 0.08), 0.35, 0.09, 0.025, 1_700e6, 500e6
        )
        assert 10 < price < 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

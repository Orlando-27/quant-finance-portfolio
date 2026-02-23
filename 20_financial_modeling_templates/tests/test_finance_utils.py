#!/usr/bin/env python3
"""
=============================================================================
UNIT TESTS -- FINANCIAL MODELING UTILITIES
=============================================================================
Tests for src/common/finance_utils.py covering:
    - WACC computation
    - Beta levering / unlevering
    - Free cash flow calculations
    - Terminal value models
    - Discounting mechanics
    - IRR and MOIC
    - Working capital projections
    - Depreciation schedules

Run:
    cd 20_financial_modeling_templates
    pytest tests/test_finance_utils.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.finance_utils import (
    cost_of_equity_capm,
    wacc,
    unlever_beta,
    relever_beta,
    unlevered_fcf,
    levered_fcf,
    terminal_value_gordon,
    terminal_value_exit_multiple,
    discount_cashflows,
    mid_year_discount_factors,
    compute_irr,
    compute_moic,
    straight_line_depreciation,
    project_working_capital,
)


# =========================================================================
# COST OF EQUITY (CAPM)
# =========================================================================
class TestCAPM:
    """Test Cost of Equity via Capital Asset Pricing Model."""

    def test_basic_capm(self):
        """Ke = 4% + 1.2 * 5.5% = 10.6%."""
        ke = cost_of_equity_capm(0.04, 1.2, 0.055)
        assert abs(ke - 0.106) < 1e-10

    def test_zero_beta(self):
        """Beta = 0 implies Ke = Rf."""
        ke = cost_of_equity_capm(0.04, 0.0, 0.06)
        assert abs(ke - 0.04) < 1e-10

    def test_beta_one(self):
        """Beta = 1 implies Ke = Rf + ERP."""
        ke = cost_of_equity_capm(0.03, 1.0, 0.05)
        assert abs(ke - 0.08) < 1e-10


# =========================================================================
# WACC
# =========================================================================
class TestWACC:
    """Test Weighted Average Cost of Capital."""

    def test_all_equity(self):
        """100% equity implies WACC = Ke."""
        result = wacc(0.10, 0.05, 0.25, 1.0, 0.0)
        assert abs(result - 0.10) < 1e-10

    def test_all_debt(self):
        """100% debt implies WACC = Kd * (1-t)."""
        result = wacc(0.10, 0.06, 0.25, 0.0, 1.0)
        assert abs(result - 0.045) < 1e-10

    def test_mixed_capital(self):
        """WACC = 0.7 * 10% + 0.3 * 5% * (1-25%) = 8.125%."""
        result = wacc(0.10, 0.05, 0.25, 0.70, 0.30)
        expected = 0.70 * 0.10 + 0.30 * 0.05 * 0.75
        assert abs(result - expected) < 1e-10


# =========================================================================
# BETA LEVER / UNLEVER
# =========================================================================
class TestBeta:
    """Test Hamada beta levering and unlevering."""

    def test_roundtrip(self):
        """Unlever then relever should return original."""
        levered = 1.20
        tax = 0.25
        de = 0.50
        unlevered = unlever_beta(levered, tax, de)
        relevered = relever_beta(unlevered, tax, de)
        assert abs(relevered - levered) < 1e-10

    def test_no_debt(self):
        """D/E = 0 means unlevered = levered."""
        result = unlever_beta(1.30, 0.25, 0.0)
        assert abs(result - 1.30) < 1e-10

    def test_positive_debt(self):
        """Unlevered beta < levered beta when D/E > 0."""
        unlevered = unlever_beta(1.50, 0.30, 0.60)
        assert unlevered < 1.50


# =========================================================================
# FREE CASH FLOW
# =========================================================================
class TestFCF:
    """Test Free Cash Flow computations."""

    def test_ufcf_scalar(self):
        """UFCF = EBIT*(1-t) + D&A - CapEx - dNWC."""
        result = unlevered_fcf(
            ebit=1000, tax_rate=0.25, da=200, capex=300, delta_nwc=50
        )
        expected = 1000 * 0.75 + 200 - 300 - 50  # 600
        assert abs(result - expected) < 1e-10

    def test_ufcf_array(self):
        """UFCF works with numpy arrays."""
        ebit = np.array([1000, 1100])
        da   = np.array([200, 210])
        capex = np.array([300, 310])
        dnwc = np.array([50, 55])
        result = unlevered_fcf(ebit, 0.25, da, capex, dnwc)
        assert result.shape == (2,)
        assert abs(result[0] - 600) < 1e-10

    def test_levered_fcf(self):
        """LFCF = NI + D&A - CapEx - dNWC + Net Borrowing."""
        result = levered_fcf(500, 200, 300, 50, 100)
        expected = 500 + 200 - 300 - 50 + 100  # 450
        assert abs(result - expected) < 1e-10


# =========================================================================
# TERMINAL VALUE
# =========================================================================
class TestTerminalValue:
    """Test terminal value calculations."""

    def test_gordon_basic(self):
        """TV = 100 * (1.025) / (0.10 - 0.025) = 1366.67."""
        tv = terminal_value_gordon(100, 0.10, 0.025)
        expected = 100 * 1.025 / 0.075
        assert abs(tv - expected) < 0.01

    def test_gordon_raises_on_invalid(self):
        """WACC <= growth should raise ValueError."""
        with pytest.raises(ValueError):
            terminal_value_gordon(100, 0.03, 0.04)

    def test_exit_multiple(self):
        """TV = EBITDA * Multiple."""
        tv = terminal_value_exit_multiple(500, 10.0)
        assert abs(tv - 5000) < 1e-10


# =========================================================================
# DISCOUNTING
# =========================================================================
class TestDiscounting:
    """Test present value calculations."""

    def test_single_cashflow(self):
        """PV of $100 in 1 year at 10% = $90.91."""
        pv_arr, total_pv = discount_cashflows([100], 0.10)
        assert abs(total_pv - 90.909090909) < 0.01

    def test_mid_year_convention(self):
        """Mid-year factors should be higher than year-end."""
        mid = mid_year_discount_factors(3, 0.10)
        year_end = 1.0 / (1.10 ** np.arange(1, 4))
        assert all(mid > year_end)

    def test_zero_rate(self):
        """Zero discount rate means PV = CF."""
        pv_arr, total_pv = discount_cashflows([100, 200, 300], 0.0)
        assert abs(total_pv - 600) < 1e-10


# =========================================================================
# IRR & MOIC
# =========================================================================
class TestIRR:
    """Test Internal Rate of Return computation."""

    def test_simple_irr(self):
        """Invest $100, receive $121 in 2 years => IRR = 10%."""
        irr = compute_irr([-100, 0, 121])
        assert abs(irr - 0.10) < 0.001

    def test_moic(self):
        """MOIC = 300 / 100 = 3.0x."""
        assert abs(compute_moic(300, 100) - 3.0) < 1e-10

    def test_irr_nan_on_failure(self):
        """All positive cashflows should return NaN."""
        irr = compute_irr([100, 100, 100])
        assert np.isnan(irr)


# =========================================================================
# DEPRECIATION
# =========================================================================
class TestDepreciation:
    """Test depreciation schedule generation."""

    def test_straight_line(self):
        """$1000 cost, $100 salvage, 9 years => $100/year."""
        dep, cum, nbv = straight_line_depreciation(1000, 100, 9)
        assert abs(dep[0] - 100) < 1e-10
        assert abs(cum[-1] - 900) < 1e-10
        assert abs(nbv[-1] - 100) < 1e-10
        assert len(dep) == 9


# =========================================================================
# WORKING CAPITAL
# =========================================================================
class TestWorkingCapital:
    """Test working capital projection."""

    def test_dimensions(self):
        """Output arrays should match input length."""
        rev = np.array([1000, 1100, 1200])
        cogs = np.array([500, 550, 600])
        ar, inv, ap, nwc, dnwc = project_working_capital(rev, cogs, 45, 30, 40)
        assert ar.shape == (3,)
        assert dnwc.shape == (3,)

    def test_nwc_positive(self):
        """NWC = AR + Inv - AP should be positive for typical assumptions."""
        rev = np.array([1000])
        cogs = np.array([500])
        ar, inv, ap, nwc, dnwc = project_working_capital(rev, cogs, 60, 45, 30)
        assert nwc[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

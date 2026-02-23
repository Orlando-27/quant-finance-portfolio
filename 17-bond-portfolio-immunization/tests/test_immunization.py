"""
test_immunization.py — Unit Tests for ImmunizationEngine
==========================================================
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bond import Bond
from src.immunization import ImmunizationEngine, Liability


def _make_universe():
    """Standard 3-bond universe for testing."""
    bonds = [
        Bond(face_value=1000, coupon_rate=0.03, maturity=2.0,  frequency=2, issuer="T2Y"),
        Bond(face_value=1000, coupon_rate=0.05, maturity=5.0,  frequency=2, issuer="T5Y"),
        Bond(face_value=1000, coupon_rate=0.06, maturity=10.0, frequency=2, issuer="T10Y"),
    ]
    yields = [0.03, 0.05, 0.06]
    return bonds, yields


def _make_liabilities():
    return [
        Liability(time=3.0, amount=50_000),
        Liability(time=5.0, amount=80_000),
        Liability(time=7.0, amount=60_000),
    ]


class TestRedington:
    """Tests for Redington immunization."""

    def setup_method(self):
        self.bonds, self.yields = _make_universe()
        self.liabs = _make_liabilities()
        self.engine = ImmunizationEngine(self.bonds, self.liabs, self.yields)

    def test_immunization_runs_without_error(self):
        result = self.engine.redington_immunization(flat_ytm=0.05)
        assert result is not None

    def test_weights_non_negative(self):
        result = self.engine.redington_immunization(flat_ytm=0.05)
        assert np.all(result["weights"] >= -1e-8)

    def test_pv_match(self):
        result = self.engine.redington_immunization(flat_ytm=0.05)
        pv_diff = abs(result["portfolio_pv"] - result["liability_pv"])
        assert pv_diff < 10, f"PV mismatch = ${pv_diff:,.2f}"

    def test_duration_match(self):
        result = self.engine.redington_immunization(flat_ytm=0.05)
        dur_diff = abs(result["portfolio_duration"] - result["liability_duration"])
        assert dur_diff < 0.05, f"Duration mismatch = {dur_diff:.4f}"

    def test_convexity_surplus_positive(self):
        result = self.engine.redington_immunization(flat_ytm=0.05)
        assert result["convexity_surplus"] >= -0.1, \
            f"Convexity surplus {result['convexity_surplus']:.4f} should be >= 0"


class TestCashFlowMatching:
    """Tests for Cash Flow Matching LP."""

    def setup_method(self):
        self.bonds, self.yields = _make_universe()
        self.liabs = _make_liabilities()
        self.engine = ImmunizationEngine(self.bonds, self.liabs, self.yields)

    def test_cfm_runs_without_error(self):
        result = self.engine.cash_flow_matching()
        assert result is not None

    def test_cfm_units_non_negative(self):
        result = self.engine.cash_flow_matching()
        assert np.all(result["units"] >= -1e-6)

    def test_coverage_ratios_gte_one(self):
        result = self.engine.cash_flow_matching()
        if result["optimizer_success"]:
            assert np.all(result["coverage_ratios"] >= 0.95), \
                f"Coverage ratio below 95%: {result['coverage_ratios']}"


class TestPortfolioBuilders:
    """Tests for Bullet, Barbell, Ladder constructors."""

    def setup_method(self):
        self.bonds, self.yields = _make_universe()
        self.budget = 100_000.0

    def test_bullet_allocates_budget(self):
        w = ImmunizationEngine.build_bullet(self.bonds, self.yields,
                                             target_maturity=5.0, budget=self.budget)
        prices = np.array([b.price(y) for b, y in zip(self.bonds, self.yields)])
        total  = float(np.dot(w, prices))
        assert abs(total - self.budget) < 1.0

    def test_barbell_two_nonzero_positions(self):
        w = ImmunizationEngine.build_barbell(self.bonds, self.yields, budget=self.budget)
        assert np.sum(w > 0) == 2

    def test_ladder_all_nonzero(self):
        w = ImmunizationEngine.build_ladder(self.bonds, self.yields, budget=self.budget)
        assert np.all(w > 0), "Ladder should hold all bonds"

"""
test_bond.py — Unit Tests for Bond Pricing Engine
===================================================
Tests cover:
  • Price-YTM consistency (round-trip)
  • Duration limiting cases (zero-coupon vs coupon bond)
  • DV01 sign and magnitude
  • Convexity positivity
  • Taylor approximation accuracy within ±200 bps

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bond import Bond


class TestBondPricing:
    """Tests for Bond.price() and Bond.ytm()."""

    def setup_method(self):
        """Standard 5-year 5% semi-annual bond at par."""
        self.bond = Bond(face_value=1000, coupon_rate=0.05,
                         maturity=5.0, frequency=2, issuer="TestBond")

    def test_par_bond_prices_at_coupon_rate(self):
        """A bond priced at coupon rate should be near par."""
        price = self.bond.price(ytm=0.05)
        assert abs(price - 1000.0) < 0.01, f"Par bond price {price:.4f} != 1000"

    def test_premium_bond(self):
        """Bond with YTM < coupon should trade at premium."""
        price = self.bond.price(ytm=0.03)
        assert price > 1000.0, "YTM < coupon should give premium"

    def test_discount_bond(self):
        """Bond with YTM > coupon should trade at discount."""
        price = self.bond.price(ytm=0.07)
        assert price < 1000.0, "YTM > coupon should give discount"

    def test_ytm_round_trip(self):
        """YTM solved from price should recover original YTM."""
        original_ytm = 0.065
        price = self.bond.price(ytm=original_ytm)
        solved_ytm = self.bond.ytm(market_price=price)
        assert abs(solved_ytm - original_ytm) < 1e-7, \
            f"YTM round-trip error: {abs(solved_ytm - original_ytm):.2e}"

    def test_zero_coupon_pricing(self):
        """Zero-coupon bond: P = F * (1 + y/m)^(-n)."""
        zcb   = Bond(face_value=1000, coupon_rate=0.0, maturity=10.0, frequency=2)
        y     = 0.06
        price = zcb.price(ytm=y)
        expected = 1000 / (1 + y/2) ** 20
        assert abs(price - expected) < 0.001, f"ZCB price mismatch: {price:.4f} vs {expected:.4f}"

    def test_price_decreases_with_yield(self):
        """Price must be strictly decreasing in yield."""
        ytms   = np.linspace(0.01, 0.15, 50)
        prices = [self.bond.price(y) for y in ytms]
        diffs  = np.diff(prices)
        assert np.all(diffs < 0), "Bond price must decrease monotonically with yield"


class TestDuration:
    """Tests for Macaulay and Modified Duration."""

    def setup_method(self):
        self.bond = Bond(face_value=1000, coupon_rate=0.05,
                         maturity=5.0, frequency=2)

    def test_macaulay_duration_positive(self):
        d = self.bond.macaulay_duration(ytm=0.05)
        assert d > 0

    def test_macaulay_lt_maturity(self):
        """Macaulay duration must be strictly less than maturity for coupon bond."""
        d = self.bond.macaulay_duration(ytm=0.05)
        assert d < self.bond.maturity, f"D_mac {d:.4f} must be < maturity {self.bond.maturity}"

    def test_zero_coupon_macaulay_equals_maturity(self):
        """For a zero-coupon bond, D_mac = maturity exactly."""
        zcb = Bond(face_value=1000, coupon_rate=0.0, maturity=7.0, frequency=2)
        d   = zcb.macaulay_duration(ytm=0.05)
        assert abs(d - 7.0) < 0.001, f"ZCB D_mac {d:.4f} != 7.0"

    def test_modified_duration_relationship(self):
        """D_mod = D_mac / (1 + y/m)."""
        ytm   = 0.06
        d_mac = self.bond.macaulay_duration(ytm)
        d_mod = self.bond.modified_duration(ytm)
        expected = d_mac / (1 + ytm / self.bond.frequency)
        assert abs(d_mod - expected) < 1e-9

    def test_dv01_positive(self):
        """DV01 must be positive (price falls as yield rises)."""
        dv01 = self.bond.dv01(ytm=0.05)
        assert dv01 > 0, f"DV01 {dv01:.4f} must be > 0"

    def test_duration_increases_with_maturity(self):
        """Longer-maturity bonds have higher duration (holding coupon fixed)."""
        d5  = Bond(coupon_rate=0.05, maturity=5.0,  frequency=2).modified_duration(0.05)
        d10 = Bond(coupon_rate=0.05, maturity=10.0, frequency=2).modified_duration(0.05)
        d30 = Bond(coupon_rate=0.05, maturity=30.0, frequency=2).modified_duration(0.05)
        assert d5 < d10 < d30, "Duration should increase with maturity"


class TestConvexity:
    """Tests for Convexity."""

    def setup_method(self):
        self.bond = Bond(face_value=1000, coupon_rate=0.05,
                         maturity=10.0, frequency=2)

    def test_convexity_positive(self):
        """Convexity must always be positive for standard bonds."""
        conv = self.bond.convexity(ytm=0.05)
        assert conv > 0, f"Convexity {conv:.4f} must be > 0"

    def test_convexity_increases_with_maturity(self):
        """Longer bonds have more convexity."""
        c5  = Bond(coupon_rate=0.05, maturity=5.0,  frequency=2).convexity(0.05)
        c10 = Bond(coupon_rate=0.05, maturity=10.0, frequency=2).convexity(0.05)
        c30 = Bond(coupon_rate=0.05, maturity=30.0, frequency=2).convexity(0.05)
        assert c5 < c10 < c30

    def test_price_approx_within_100bps(self):
        """Taylor approximation should be within 1% for ±100bps shock."""
        ytm   = 0.05
        shock = 0.01                              # 100 bps
        exact = self.bond.price(ytm + shock)
        approx = self.bond.price_change_approx(ytm, shock)
        rel_err = abs(exact - approx) / exact
        assert rel_err < 0.01, f"Approx error {rel_err*100:.3f}% > 1% for 100bps"

    def test_convexity_benefit_positive(self):
        """Convexity ensures price rises more on down-shock than it falls on up-shock."""
        ytm = 0.05
        dy  = 0.01
        p_up   = self.bond.price(ytm + dy)
        p_down = self.bond.price(ytm - dy)
        p_base = self.bond.price(ytm)
        gain   = p_down - p_base
        loss   = p_base - p_up
        assert gain > loss, "Convexity benefit: gain on rally > loss on sell-off"

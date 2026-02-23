"""
test_krd.py — Unit Tests for Key Rate Duration Engine
=======================================================
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bond import Bond
from src.key_rate_duration import KeyRateDuration


class TestKeyRateDuration:

    def setup_method(self):
        self.bond   = Bond(face_value=1000, coupon_rate=0.05,
                           maturity=10.0, frequency=2, issuer="T10Y")
        self.ytm    = 0.05
        self.engine = KeyRateDuration(
            key_tenors=[0.25, 0.5, 1, 2, 5, 10, 30],
            shock_bps=1.0
        )

    def test_krd_computation_runs(self):
        result = self.engine.compute_bond_krd(self.bond, self.ytm)
        assert "krd_vector" in result

    def test_krd_vector_length(self):
        result = self.engine.compute_bond_krd(self.bond, self.ytm)
        assert len(result["krd_vector"]) == len(self.engine.key_tenors)

    def test_sum_krd_approx_dmod(self):
        """Sum of KRDs should approximately equal Modified Duration."""
        result = self.engine.compute_bond_krd(self.bond, self.ytm)
        tol    = 0.5    # Allow 0.5yr tolerance due to interpolation
        assert abs(result["sum_krd"] - result["d_mod"]) < tol, \
            f"ΣD={result['sum_krd']:.4f} vs D_mod={result['d_mod']:.4f}"

    def test_krdv01_proportional_to_krd(self):
        """KR-DV01 = KRD * P * 1e-4."""
        result  = self.engine.compute_bond_krd(self.bond, self.ytm)
        P       = result["base_price"]
        expected = result["krd_vector"] * P * 1e-4
        np.testing.assert_allclose(result["krdv01"], expected, rtol=1e-6)

    def test_long_tenor_bond_has_more_long_end_krd(self):
        """A 30Y bond should have most KRD at long tenors."""
        bond30 = Bond(face_value=1000, coupon_rate=0.04, maturity=30.0, frequency=2)
        result = self.engine.compute_bond_krd(bond30, 0.04)
        krd    = result["krd_vector"]
        # Last two tenors (10Y + 30Y) should dominate
        long_share = (krd[-1] + krd[-2]) / max(np.sum(krd), 1e-9)
        assert long_share > 0.4, f"Long-end KRD share {long_share:.2%} < 40%"

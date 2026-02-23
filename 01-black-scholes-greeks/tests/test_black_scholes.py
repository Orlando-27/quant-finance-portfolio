"""
Unit Tests for Black-Scholes Pricing Engine
=============================================

Validates pricing, Greeks, and boundary conditions.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.black_scholes import BlackScholesEngine, OptionParameters, OptionType


@pytest.fixture
def engine():
    return BlackScholesEngine()

@pytest.fixture
def atm_params():
    return OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)


class TestPricing:
    def test_call_known_value(self, engine, atm_params):
        price = engine.price(atm_params, OptionType.CALL)
        assert abs(price - 10.4506) < 0.01

    def test_put_call_parity(self, engine, atm_params):
        result = engine.put_call_parity_check(atm_params)
        assert result["parity_holds"]

    def test_deep_itm_call(self, engine):
        p = OptionParameters(S=200, K=100, T=0.01, r=0.05, sigma=0.20)
        price = engine.price(p, OptionType.CALL)
        assert abs(price - (200 - 100 * np.exp(-0.05 * 0.01))) < 0.5

    def test_deep_otm_put(self, engine):
        p = OptionParameters(S=200, K=100, T=0.25, r=0.05, sigma=0.20)
        assert engine.price(p, OptionType.PUT) < 0.01


class TestGreeks:
    def test_delta_finite_diff(self, engine, atm_params):
        dS = 0.01
        p_up = OptionParameters(S=100.01, K=100, T=1, r=0.05, sigma=0.20)
        p_dn = OptionParameters(S=99.99, K=100, T=1, r=0.05, sigma=0.20)
        fd = (engine.price(p_up, OptionType.CALL)
              - engine.price(p_dn, OptionType.CALL)) / (2 * dS)
        analytical = engine.delta(atm_params, OptionType.CALL)
        assert abs(fd - analytical) < 1e-4

    def test_gamma_positive(self, engine, atm_params):
        assert engine.gamma(atm_params) > 0

    def test_call_delta_bounds(self, engine, atm_params):
        assert 0 <= engine.delta(atm_params, OptionType.CALL) <= 1

    def test_put_delta_bounds(self, engine, atm_params):
        assert -1 <= engine.delta(atm_params, OptionType.PUT) <= 0


class TestValidation:
    def test_negative_spot(self):
        with pytest.raises(ValueError):
            OptionParameters(S=-100, K=100, T=1, r=0.05, sigma=0.20)

    def test_negative_vol(self):
        with pytest.raises(ValueError):
            OptionParameters(S=100, K=100, T=1, r=0.05, sigma=-0.20)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

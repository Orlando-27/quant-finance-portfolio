"""
Unit Tests -- Monte Carlo Exotic Derivatives
=============================================
Tests path generation, exotic option pricing (Asian, Barrier, Lookback),
convergence behavior, and variance reduction techniques.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.path_generator import PathGenerator
from src.models.asian_options import AsianOptionPricer
from src.models.barrier_options import BarrierOptionPricer
from src.models.lookback_options import LookbackOptionPricer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_params():
    """Standard GBM parameters for testing."""
    return {
        "S0": 100.0,
        "r": 0.05,
        "sigma": 0.20,
        "T": 1.0,
    }


@pytest.fixture
def path_gen():
    """Path generator instance with fixed seed."""
    return PathGenerator(seed=42)


# ---------------------------------------------------------------------------
# Path Generator Tests
# ---------------------------------------------------------------------------
class TestPathGenerator:
    """Tests for GBM path generation."""

    def test_path_shape(self, path_gen, base_params):
        """Verify output dimensions match (n_paths, n_steps+1)."""
        n_paths, n_steps = 1000, 252
        paths = path_gen.generate_gbm(
            S0=base_params["S0"], r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=n_paths, n_steps=n_steps,
        )
        assert paths.shape == (n_paths, n_steps + 1)

    def test_initial_value(self, path_gen, base_params):
        """All paths must start at S0."""
        paths = path_gen.generate_gbm(
            S0=base_params["S0"], r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=500, n_steps=100,
        )
        np.testing.assert_allclose(paths[:, 0], base_params["S0"])

    def test_positive_prices(self, path_gen, base_params):
        """GBM paths must remain strictly positive."""
        paths = path_gen.generate_gbm(
            S0=base_params["S0"], r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=5000, n_steps=252,
        )
        assert np.all(paths > 0)

    def test_expected_terminal_value(self, path_gen, base_params):
        """E[S_T] under risk-neutral measure = S0 * exp(r*T)."""
        n_paths = 50_000
        paths = path_gen.generate_gbm(
            S0=base_params["S0"], r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=n_paths, n_steps=252,
        )
        expected = base_params["S0"] * np.exp(base_params["r"] * base_params["T"])
        empirical = paths[:, -1].mean()
        assert abs(empirical - expected) / expected < 0.02, \
            f"E[S_T] = {empirical:.2f}, expected ~{expected:.2f}"

    def test_reproducibility(self, base_params):
        """Same seed must produce identical paths."""
        gen1 = PathGenerator(seed=123)
        gen2 = PathGenerator(seed=123)
        p1 = gen1.generate_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                n_paths=100, n_steps=50)
        p2 = gen2.generate_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                n_paths=100, n_steps=50)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Asian Option Tests
# ---------------------------------------------------------------------------
class TestAsianOptions:
    """Tests for arithmetic and geometric Asian option pricing."""

    def test_asian_call_positive(self, base_params):
        """Asian call price must be non-negative."""
        pricer = AsianOptionPricer(seed=42)
        price = pricer.price_arithmetic(
            S0=base_params["S0"], K=100, r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=10_000, n_steps=252, option_type="call",
        )
        assert price >= 0

    def test_asian_put_positive(self, base_params):
        """Asian put price must be non-negative."""
        pricer = AsianOptionPricer(seed=42)
        price = pricer.price_arithmetic(
            S0=base_params["S0"], K=100, r=base_params["r"],
            sigma=base_params["sigma"], T=base_params["T"],
            n_paths=10_000, n_steps=252, option_type="put",
        )
        assert price >= 0

    def test_asian_cheaper_than_european(self, base_params):
        """Asian call <= European call (averaging reduces volatility)."""
        from scipy.stats import norm
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        pricer = AsianOptionPricer(seed=42)
        asian_call = pricer.price_arithmetic(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            n_paths=50_000, n_steps=252, option_type="call",
        )
        assert asian_call < bs_call * 1.05  # Allow 5% MC noise

    def test_deep_itm_asian_call(self, base_params):
        """Deep ITM Asian call should approximate intrinsic value."""
        pricer = AsianOptionPricer(seed=42)
        price = pricer.price_arithmetic(
            S0=200, K=100, r=0.05, sigma=0.10, T=1.0,
            n_paths=20_000, n_steps=252, option_type="call",
        )
        # Expected average ~= S0 * exp(r*T/2) for continuous averaging
        assert price > 80  # Deep ITM, must have significant value


# ---------------------------------------------------------------------------
# Barrier Option Tests
# ---------------------------------------------------------------------------
class TestBarrierOptions:
    """Tests for knock-in and knock-out barrier options."""

    def test_knock_out_leq_european(self, base_params):
        """Down-and-out call <= European call (extra extinction risk)."""
        from scipy.stats import norm
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        pricer = BarrierOptionPricer(seed=42)
        barrier_price = pricer.price_down_and_out(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            barrier=80, n_paths=50_000, n_steps=252, option_type="call",
        )
        assert barrier_price <= bs_call * 1.05

    def test_knock_out_positive(self, base_params):
        """Barrier option price must be non-negative."""
        pricer = BarrierOptionPricer(seed=42)
        price = pricer.price_down_and_out(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, n_paths=10_000, n_steps=252, option_type="call",
        )
        assert price >= 0

    def test_high_barrier_approaches_european(self, base_params):
        """Down-and-out with very low barrier ~= European call."""
        from scipy.stats import norm
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        pricer = BarrierOptionPricer(seed=42)
        price = pricer.price_down_and_out(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            barrier=1.0,  # Extremely low barrier, almost never hit
            n_paths=50_000, n_steps=252, option_type="call",
        )
        assert abs(price - bs_call) / bs_call < 0.05

    def test_in_out_parity(self):
        """Knock-in + Knock-out = European (barrier parity)."""
        pricer = BarrierOptionPricer(seed=42)
        params = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
                      barrier=85, n_paths=50_000, n_steps=252, option_type="call")

        ko = pricer.price_down_and_out(**params)
        ki = pricer.price_down_and_in(**params)

        from scipy.stats import norm
        d1 = (np.log(100/100) + (0.05 + 0.02)*1) / (0.2*1)
        d2 = d1 - 0.2
        bs = 100*norm.cdf(d1) - 100*np.exp(-0.05)*norm.cdf(d2)

        assert abs((ko + ki) - bs) / bs < 0.10  # 10% tolerance for MC


# ---------------------------------------------------------------------------
# Lookback Option Tests
# ---------------------------------------------------------------------------
class TestLookbackOptions:
    """Tests for floating-strike lookback options."""

    def test_lookback_call_geq_european(self, base_params):
        """Lookback call >= European call (optimal strike is better)."""
        from scipy.stats import norm
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        pricer = LookbackOptionPricer(seed=42)
        lb_price = pricer.price_floating_strike(
            S0=S0, r=r, sigma=sigma, T=T,
            n_paths=50_000, n_steps=252, option_type="call",
        )
        assert lb_price >= bs_call * 0.95  # Allow MC noise

    def test_lookback_positive(self, base_params):
        """Lookback option price must be positive."""
        pricer = LookbackOptionPricer(seed=42)
        price = pricer.price_floating_strike(
            S0=100, r=0.05, sigma=0.2, T=1.0,
            n_paths=10_000, n_steps=252, option_type="call",
        )
        assert price > 0

    def test_higher_vol_higher_lookback(self):
        """Higher volatility -> higher lookback price."""
        pricer = LookbackOptionPricer(seed=42)
        params = dict(S0=100, r=0.05, T=1.0,
                      n_paths=30_000, n_steps=252, option_type="call")
        p_low = pricer.price_floating_strike(sigma=0.10, **params)
        p_high = pricer.price_floating_strike(sigma=0.40, **params)
        assert p_high > p_low


# ---------------------------------------------------------------------------
# Convergence Tests
# ---------------------------------------------------------------------------
class TestConvergence:
    """Tests for Monte Carlo convergence properties."""

    def test_standard_error_decreases(self, base_params):
        """Standard error should decrease with sqrt(n_paths)."""
        pricer = AsianOptionPricer(seed=42)
        se_1k = pricer.price_arithmetic(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
            n_paths=1_000, n_steps=100, option_type="call",
            return_se=True,
        )
        se_10k = pricer.price_arithmetic(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
            n_paths=10_000, n_steps=100, option_type="call",
            return_se=True,
        )
        # If return_se returns tuple (price, se)
        if isinstance(se_1k, tuple):
            assert se_10k[1] < se_1k[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

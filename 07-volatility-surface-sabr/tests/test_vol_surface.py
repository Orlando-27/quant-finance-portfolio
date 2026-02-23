"""
================================================================================
UNIT TESTS: VOLATILITY SURFACE & SABR CALIBRATION
================================================================================
Run: pytest tests/test_vol_surface.py -v

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import pytest
import numpy as np


@pytest.fixture
def atm_params():
    return {"F": 100.0, "K": 100.0, "T": 0.5, "sigma": 0.20, "df": 1.0}


@pytest.fixture
def market_smile():
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    vols = np.array([0.30, 0.27, 0.24, 0.21, 0.20, 0.195, 0.20, 0.21, 0.23])
    return {"strikes": strikes, "vols": vols, "forward": 100.0, "expiry": 0.5}


@pytest.fixture
def multi_expiry_data():
    from src.surface import SliceData
    slices = []
    base_strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    for T, atm_v in [(0.083, 0.22), (0.25, 0.21), (0.5, 0.20), (1.0, 0.19)]:
        skew = 0.15 * np.exp(-0.5 * T)
        k = np.log(base_strikes / 100.0)
        vols = atm_v - skew * k + 0.05 * k ** 2
        slices.append(SliceData(
            expiry=T, forward=100.0, strikes=base_strikes,
            market_vols=np.maximum(vols, 0.05)
        ))
    return slices


class TestImpliedVol:
    def test_newton_roundtrip(self, atm_params):
        from src.implied_vol import black_price, implied_vol_newton
        price = black_price(atm_params["F"], atm_params["K"], atm_params["T"],
                             atm_params["sigma"], "call", atm_params["df"])
        result = implied_vol_newton(price, atm_params["F"], atm_params["K"],
                                     atm_params["T"], "call", atm_params["df"])
        assert result.converged
        assert abs(result.iv - atm_params["sigma"]) < 1e-8

    def test_brent_roundtrip(self, atm_params):
        from src.implied_vol import black_price, implied_vol_brent
        price = black_price(atm_params["F"], atm_params["K"], atm_params["T"],
                             atm_params["sigma"], "call", atm_params["df"])
        result = implied_vol_brent(price, atm_params["F"], atm_params["K"],
                                    atm_params["T"], "call", atm_params["df"])
        assert result.converged
        assert abs(result.iv - atm_params["sigma"]) < 1e-8

    def test_put_call_parity_iv(self, atm_params):
        from src.implied_vol import black_price, implied_vol_newton
        call_price = black_price(atm_params["F"], atm_params["K"], atm_params["T"],
                                  atm_params["sigma"], "call", atm_params["df"])
        put_price = black_price(atm_params["F"], atm_params["K"], atm_params["T"],
                                 atm_params["sigma"], "put", atm_params["df"])
        iv_call = implied_vol_newton(call_price, atm_params["F"], atm_params["K"],
                                      atm_params["T"], "call")
        iv_put = implied_vol_newton(put_price, atm_params["F"], atm_params["K"],
                                     atm_params["T"], "put")
        assert abs(iv_call.iv - iv_put.iv) < 1e-6

    def test_deep_otm(self):
        from src.implied_vol import black_price, implied_vol_newton
        price = black_price(100, 60, 0.25, 0.30, "call", 1.0)
        result = implied_vol_newton(price, 100, 60, 0.25, "call")
        assert result.converged
        assert abs(result.iv - 0.30) < 1e-4


class TestSABR:
    def test_atm_limit(self):
        from src.sabr import SABRModel
        sabr = SABRModel(beta=0.5)
        vol = sabr.implied_vol(np.array([100.0]), 100.0, 0.5,
                                alpha=0.2, rho=-0.3, nu=0.4)
        assert vol[0] > 0
        assert vol[0] < 1.0

    def test_calibration_roundtrip(self, market_smile):
        from src.sabr import SABRModel
        sabr = SABRModel(beta=0.5)
        result = sabr.calibrate(
            forward=market_smile["forward"],
            strikes=market_smile["strikes"],
            market_vols=market_smile["vols"],
            expiry=market_smile["expiry"]
        )
        assert result.converged
        assert result.rmse < 0.005

    def test_smile_shape(self, market_smile):
        from src.sabr import SABRModel
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(market_smile["forward"], market_smile["strikes"],
                        market_smile["vols"], market_smile["expiry"])
        strikes_fine = np.linspace(80, 120, 200)
        vols = sabr.implied_vol(strikes_fine, 100.0, 0.5)
        atm_idx = np.argmin(np.abs(strikes_fine - 100))
        assert vols[0] > vols[atm_idx]
        assert vols[-1] > vols[atm_idx]

    def test_obloj_vs_hagan(self, market_smile):
        from src.sabr import SABRModel
        hagan = SABRModel(beta=0.5, formula="hagan")
        obloj = SABRModel(beta=0.5, formula="obloj")
        hagan.calibrate(market_smile["forward"], market_smile["strikes"],
                         market_smile["vols"], market_smile["expiry"])
        obloj.calibrate(market_smile["forward"], market_smile["strikes"],
                         market_smile["vols"], market_smile["expiry"])
        assert hagan.params.alpha > 0
        assert obloj.params.alpha > 0


class TestSVI:
    def test_fit_quality(self):
        from src.svi import SVIModel
        k = np.linspace(-0.3, 0.3, 15)
        w_market = 0.04 + 0.02 * k ** 2
        svi = SVIModel()
        result = svi.fit(k, w_market)
        assert result.rmse < 0.001

    def test_positive_variance(self):
        from src.svi import SVIModel
        k = np.linspace(-0.2, 0.2, 20)
        w_market = 0.05 + 0.1 * np.abs(k)
        svi = SVIModel()
        svi.fit(k, w_market)
        w_model = svi.total_variance(np.linspace(-0.5, 0.5, 100))
        assert np.all(w_model > -1e-6)

    def test_arbitrage_check(self):
        from src.svi import SVIModel
        k = np.linspace(-0.3, 0.3, 20)
        w_market = 0.04 + 0.01 * (k ** 2)
        svi = SVIModel()
        svi.fit(k, w_market)
        arb = svi.check_arbitrage()
        assert "butterfly_free" in arb
        assert "positive_variance" in arb


class TestSurface:
    def test_sabr_surface_fit(self, multi_expiry_data):
        from src.surface import VolSurface
        surf = VolSurface(method="sabr", beta=0.5)
        diag = surf.fit(multi_expiry_data)
        assert diag.total_rmse < 0.01
        assert len(surf.expiries) == 4

    def test_interpolation(self, multi_expiry_data):
        from src.surface import VolSurface
        surf = VolSurface(method="sabr", beta=0.5)
        surf.fit(multi_expiry_data)
        vol = surf.get_vol(100.0, 0.375)
        assert 0.05 < vol < 0.50

    def test_atm_term_structure(self, multi_expiry_data):
        from src.surface import VolSurface
        surf = VolSurface(method="sabr", beta=0.5)
        surf.fit(multi_expiry_data)
        T, vols = surf.atm_term_structure()
        assert len(T) == 4
        assert all(v > 0 for v in vols)


class TestVannaVolga:
    def test_pillar_recovery(self):
        from src.vanna_volga import VannaVolga, VannaVolgaQuotes
        q = VannaVolgaQuotes(
            K_put=90, K_atm=100, K_call=110,
            vol_put=0.25, vol_atm=0.20, vol_call=0.22,
            forward=100.0, expiry=0.25
        )
        vv = VannaVolga(q)
        quality = vv.fit_quality()
        assert quality["max_error"] < 0.01

    def test_smile_shape(self):
        from src.vanna_volga import VannaVolga, VannaVolgaQuotes
        q = VannaVolgaQuotes(
            K_put=90, K_atm=100, K_call=110,
            vol_put=0.25, vol_atm=0.20, vol_call=0.22,
            forward=100.0, expiry=0.25
        )
        vv = VannaVolga(q)
        strikes = np.linspace(85, 115, 50)
        vols = vv.smile(strikes)
        assert all(v > 0 for v in vols)


class TestArbitrage:
    def test_clean_surface(self, multi_expiry_data):
        from src.surface import VolSurface
        from src.arbitrage import ArbitrageChecker
        surf = VolSurface(method="sabr", beta=0.5)
        surf.fit(multi_expiry_data)
        checker = ArbitrageChecker(surf)
        diag = checker.full_check(forward=100.0)
        assert diag.n_violations < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

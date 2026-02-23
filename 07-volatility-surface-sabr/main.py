"""
Volatility Surface & SABR Calibration - Main Entry Point
=========================================================
Demonstrates the complete workflow: market data generation,
implied volatility extraction, SABR calibration, SVI fitting,
local volatility computation, and arbitrage diagnostics.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from src.implied_vol import ImpliedVolSolver
from src.sabr import SABRModel, SABRCalibrator
from src.svi import SVIParametrization
from src.surface import VolatilitySurface
from src.local_vol import DupireLocalVol
from src.arbitrage import ArbitrageDiagnostics
from src.vanna_volga import VannaVolga
from src.visualization import VolSurfaceVisualizer


def generate_synthetic_market_data():
    """Generate synthetic option market data for demonstration."""
    np.random.seed(42)
    S0 = 100.0
    r = 0.05

    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    expiries = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00])

    # Base SABR parameters for generating realistic smiles
    alpha, beta, rho, nu = 0.25, 0.7, -0.3, 0.4
    sabr = SABRModel(alpha=alpha, beta=beta, rho=rho, nu=nu)

    market_vols = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        F = S0 * np.exp(r * T)
        for j, K in enumerate(strikes):
            vol = sabr.implied_vol_hagan(F, K, T)
            # Add noise to simulate real market
            market_vols[i, j] = vol + np.random.normal(0, 0.005)

    return S0, r, strikes, expiries, market_vols


def main():
    """Run the complete volatility surface analysis pipeline."""
    print("=" * 70)
    print("VOLATILITY SURFACE & SABR CALIBRATION FRAMEWORK")
    print("=" * 70)

    # --- Step 1: Market Data ---
    print("\n[1/6] Generating synthetic market data...")
    S0, r, strikes, expiries, market_vols = generate_synthetic_market_data()
    print(f"  Spot: {S0} | Rate: {r}")
    print(f"  Strikes: {strikes}")
    print(f"  Expiries: {expiries}")

    # --- Step 2: SABR Calibration ---
    print("\n[2/6] Calibrating SABR model per expiry...")
    calibrator = SABRCalibrator(beta=0.7)
    for i, T in enumerate(expiries):
        F = S0 * np.exp(r * T)
        params = calibrator.calibrate(F, strikes, market_vols[i, :], T)
        rmse = np.sqrt(np.mean((
            np.array([SABRModel(**params).implied_vol_hagan(F, K, T) for K in strikes])
            - market_vols[i, :]
        ) ** 2))
        print(f"  T={T:.2f}y | alpha={params['alpha']:.4f} "
              f"rho={params['rho']:.4f} nu={params['nu']:.4f} | RMSE={rmse:.6f}")

    # --- Step 3: SVI Parametrization ---
    print("\n[3/6] Fitting SVI parametrization...")
    svi = SVIParametrization()
    F_1y = S0 * np.exp(r * 1.0)
    log_moneyness = np.log(strikes / F_1y)
    svi_params = svi.fit(log_moneyness, market_vols[3, :] ** 2)  # T=1y slice
    print(f"  SVI params (T=1y): {svi_params}")

    # --- Step 4: Surface Construction ---
    print("\n[4/6] Constructing volatility surface...")
    surface = VolatilitySurface(strikes, expiries, market_vols)
    atm_vols = surface.get_atm_term_structure(S0, r)
    print("  ATM term structure:")
    for T, vol in zip(expiries, atm_vols):
        print(f"    T={T:.2f}y | ATM vol={vol:.4f}")

    # --- Step 5: Arbitrage Diagnostics ---
    print("\n[5/6] Running arbitrage diagnostics...")
    diagnostics = ArbitrageDiagnostics()
    for i, T in enumerate(expiries):
        F = S0 * np.exp(r * T)
        cal_free = diagnostics.check_calendar_arbitrage(
            market_vols[max(0, i-1), :], market_vols[i, :],
            max(expiries[max(0, i-1)], 0.01), T
        )
        bfly_free = diagnostics.check_butterfly_arbitrage(
            strikes, market_vols[i, :], F, T
        )
        print(f"  T={T:.2f}y | Calendar: {'PASS' if cal_free else 'FAIL'} "
              f"| Butterfly: {'PASS' if bfly_free else 'FAIL'}")

    # --- Step 6: Summary ---
    print("\n[6/6] Pipeline complete.")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Market data: {len(expiries)} expiries x {len(strikes)} strikes")
    print(f"  SABR calibration: {len(expiries)} slices calibrated")
    print(f"  Vol range: [{market_vols.min():.4f}, {market_vols.max():.4f}]")
    print("  All visualizations available via VolSurfaceVisualizer class")
    print("=" * 70)


if __name__ == "__main__":
    main()

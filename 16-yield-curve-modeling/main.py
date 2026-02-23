"""
Yield Curve Modeling and Forecasting — Main Pipeline
======================================================
Full analysis: NS/NSS fitting, PCA decomposition, Diebold-Li VAR
forecasting, bootstrap confidence bands, publication-quality charts.

Usage:
    python main.py                          # synthetic US
    python main.py --mode tes               # Colombian TES
    python main.py --mode live              # try yfinance live data
    python main.py --bootstrap-n 100        # faster bootstrap
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from models.nelson_siegel  import NelsonSiegel, NelsonSiegelSvensson, diebold_li_fit_panel
from models.pca_factors    import YieldCurvePCA
from models.var_forecast   import DieboldLiVAR
from utils.data_loader     import SyntheticYieldCurve, USTreasuryLoader
from utils.helpers         import (cubic_spline_curve, curve_fit_metrics,
                                   yield_change_decomposition)
from visualization.charts  import (plot_ns_nss_fit, plot_factor_dynamics,
                                   plot_pca_analysis, plot_var_forecast,
                                   plot_bootstrap_bands)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yield Curve Modeling Pipeline")
    p.add_argument("--mode",        default="synthetic",
                   choices=["synthetic", "tes", "live"])
    p.add_argument("--n-periods",   type=int, default=120)
    p.add_argument("--forecast-h",  type=int, default=12)
    p.add_argument("--bootstrap-n", type=int, default=300)
    p.add_argument("--output-dir",  default="outputs/figures")
    return p.parse_args()


def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------
def run_ns_nss_fitting(tenors, yields, label, out_dir):
    _banner(f"NS / NSS Fitting — {label}")

    ns  = NelsonSiegel()
    nss = NelsonSiegelSvensson()
    ns_p  = ns.fit(tenors, yields)
    nss_p = nss.fit(tenors, yields)

    tau_fine    = np.linspace(0.1, max(tenors), 200)
    ns_fine     = ns.predict(tau_fine)
    nss_fine    = nss.predict(tau_fine)
    spline_fine = cubic_spline_curve(tenors, yields, tau_fine)
    ns_at_obs   = ns.predict(tenors)
    nss_at_obs  = nss.predict(tenors)

    ns_diag  = curve_fit_metrics(yields, ns_at_obs,  tenors)
    nss_diag = curve_fit_metrics(yields, nss_at_obs, tenors)

    print(f"\nNelson-Siegel fit:")
    for k, v in ns_diag.items():
        if k != "per_tenor":
            print(f"  {k:<22}: {v}")
    print(f"  β₀(Level) = {ns_p.beta0*100:.4f}%  "
          f"β₁(Slope) = {ns_p.beta1*100:.4f}%  "
          f"β₂(Curv) = {ns_p.beta2*100:.4f}%  "
          f"λ = {ns_p.lam:.4f}")

    print(f"\nNelson-Siegel-Svensson fit:")
    for k, v in nss_diag.items():
        if k != "per_tenor":
            print(f"  {k:<22}: {v}")

    fwd = nss.forward_curve(tau_fine)
    print(f"\nNSS Forward curve: [{fwd.min()*100:.2f}%, {fwd.max()*100:.2f}%]")

    plot_ns_nss_fit(
        tenors, yields, ns_at_obs, nss_at_obs,
        tau_fine, ns_fine, nss_fine, spline_fine,
        ns_p, nss_p, title=label,
        save_path=f"{out_dir}/01_ns_nss_fit.png",
    )
    return ns, nss, tau_fine


def run_factor_dynamics(panel, tenors, label, out_dir):
    _banner(f"Factor Dynamics — {label}")

    factors = diebold_li_fit_panel(tenors, panel, lam=0.0609)
    print(f"\nFactor summary statistics (%):")
    print((factors * 100).describe().round(4).to_string())

    decomp = yield_change_decomposition(panel)
    print(f"\nYield change decomposition (mean/month, bps):")
    print((decomp.mean() * 1e4).round(3).to_string())

    plot_factor_dynamics(
        factors, title=label,
        save_path=f"{out_dir}/02_factor_dynamics.png",
    )
    return factors


def run_pca_analysis(panel, out_dir):
    _banner("PCA Factor Analysis")

    pca    = YieldCurvePCA(n_components=3, scale=True)
    pca.fit(panel)
    scores = pca.transform(panel)

    print("\nPCA Summary:")
    print(pca.summary().to_string(index=False))
    print(f"\nLoadings:\n{pca.loadings_.round(4).to_string()}")

    recon  = pca.reconstruct(panel)
    common = panel.index.intersection(recon.index)
    rmse   = np.sqrt(((panel.loc[common] - recon.loc[common])**2).mean().mean()) * 1e4
    print(f"\nAvg reconstruction RMSE (3 PCs): {rmse:.3f} bps")

    plot_pca_analysis(pca, scores, panel,
                      save_path=f"{out_dir}/03_pca_analysis.png")
    return pca


def run_var_forecast(factors, panel, tenors, h, label, out_dir):
    _banner(f"Diebold-Li VAR Forecast — {label}")

    dl = DieboldLiVAR(max_lags=6, lam=0.0609)
    dl.fit(factors)

    summ = dl.summary()
    print(f"\n  Optimal lag : {summ['optimal_lag']}")
    print(f"  AIC         : {summ['aic']:.4f}")
    print(f"  BIC         : {summ['bic']:.4f}")
    print(f"  Observations: {summ['n_obs']}")

    print(f"\nADF stationarity tests:")
    for col, res in summ["adf_tests"].items():
        status = "Stationary ✓" if res["p_value"] < 0.05 else "Non-stationary"
        print(f"  {col:<22}: ADF={res['adf_stat']:.3f}  "
              f"p={res['p_value']:.4f}  {status}")

    fc_curves = dl.forecast_curves(tenors, h=h)
    print(f"\n  t+1  : {(fc_curves.iloc[0].values*100).round(3)}")
    print(f"  t+{h} : {(fc_curves.iloc[-1].values*100).round(3)}")

    historical_last = panel.dropna().iloc[-1].values[:len(tenors)]
    plot_var_forecast(
        tenors, historical_last, fc_curves, panel,
        title=label,
        save_path=f"{out_dir}/04_var_forecast.png",
    )
    return fc_curves


def run_bootstrap(ns, tenors, yields, tau_fine, n_iter, label, out_dir):
    _banner(f"Bootstrap Confidence Bands — {label}")

    boot    = ns.bootstrap_bands(tenors, yields, tau_fine,
                                 n_iter=n_iter, ci=0.95, seed=42)
    ns_fine = ns.predict(tau_fine)
    bw      = (boot["upper"] - boot["lower"]) * 1e4

    print(f"\n  Bootstrap n       : {n_iter}")
    print(f"  Avg 95% CI width  : {bw.mean():.2f} bps")
    print(f"  Max CI width      : {bw.max():.2f} bps "
          f"(at τ={tau_fine[np.argmax(bw)]:.1f}y)")

    plot_bootstrap_bands(
        tenors, yields, tau_fine, boot, ns_fine,
        title=label,
        save_path=f"{out_dir}/05_bootstrap_bands.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _args()
    os.makedirs(args.output_dir, exist_ok=True)

    _banner("YIELD CURVE MODELING & FORECASTING")
    print(f"  Mode       : {args.mode}")
    print(f"  Periods    : {args.n_periods} months")
    print(f"  Forecast   : {args.forecast_h} months ahead")
    print(f"  Bootstrap  : {args.bootstrap_n} iterations")
    print(f"  Output dir : {args.output_dir}")

    # --- Load data ---
    if args.mode == "live":
        loader = USTreasuryLoader(start="2015-01-01", end="2024-12-31")
        panel  = loader.fetch()
        if panel.empty or len(panel) < 12:
            print("  [WARNING] Live data unavailable — falling back to synthetic")
            args.mode = "synthetic"

    if args.mode in ("synthetic", "tes"):
        mode_map = {"synthetic": "us", "tes": "tes"}
        gen      = SyntheticYieldCurve(mode=mode_map[args.mode],
                                        n_periods=args.n_periods, seed=42)
        panel    = gen.generate()

    label  = ("Colombian TES (Synthetic)" if args.mode == "tes"
               else "US Treasuries (Synthetic)")
    tenors = panel.columns.astype(float).values

    print(f"\n  Panel shape : {panel.shape}")
    snapshot = panel.dropna().iloc[-1].values
    print("  Latest yields (%):")
    print("  " + "  ".join(f"{t:.1f}y={y*100:.2f}%"
                             for t, y in zip(tenors, snapshot)))

    # --- Run pipeline ---
    ns, nss, tau_fine = run_ns_nss_fitting(tenors, snapshot, label, args.output_dir)
    factors           = run_factor_dynamics(panel, tenors, label, args.output_dir)
    _                 = run_pca_analysis(panel, args.output_dir)
    _                 = run_var_forecast(factors, panel, tenors,
                                          h=args.forecast_h,
                                          label=label, out_dir=args.output_dir)
    run_bootstrap(ns, tenors, snapshot, tau_fine,
                  n_iter=args.bootstrap_n, label=label, out_dir=args.output_dir)

    _banner("PIPELINE COMPLETE")
    print(f"  5 charts saved to: {args.output_dir}/")
    print("  Try: python main.py --mode tes")


if __name__ == "__main__":
    main()

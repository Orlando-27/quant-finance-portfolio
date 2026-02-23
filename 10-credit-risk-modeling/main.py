"""
Credit Risk Modeling -- Main Entry Point
==========================================
Demonstrates the complete credit risk pipeline: Merton structural
model, hazard rate bootstrapping, Vasicek portfolio model, CDS
pricing, CreditMetrics simulation, and Credit VaR.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from src.models.merton import MertonModel
from src.models.reduced_form import HazardRateModel
from src.models.vasicek import VasicekPortfolioModel
from src.models.creditmetrics import CreditMetricsEngine
from src.models.cds_pricing import CDSPricer
from src.credit_var import CreditVaREngine


def main():
    print("=" * 70)
    print("CREDIT RISK MODELING FRAMEWORK")
    print("=" * 70)

    # --- 1. Merton Structural Model ---
    print("\n[1/6] Merton Structural Model")
    print("-" * 40)
    merton = MertonModel(risk_free_rate=0.05)
    result = merton.calibrate_from_equity(
        E_market=50.0, sigma_E=0.40, D=80.0, T=1.0
    )
    print(f"  Asset value:       {result.asset_value:.2f}")
    print(f"  Asset volatility:  {result.asset_vol:.4f}")
    print(f"  Distance to Def:   {result.distance_to_default:.4f}")
    print(f"  PD (risk-neutral): {result.default_prob_rn:.4%}")
    print(f"  Credit spread:     {result.credit_spread:.1f} bps")
    print(f"  Recovery rate:     {result.recovery_rate:.4f}")

    # --- 2. Reduced-Form Model ---
    print("\n[2/6] Hazard Rate Bootstrapping from CDS Spreads")
    print("-" * 40)
    hazard_model = HazardRateModel(recovery_rate=0.40)
    cds_tenors = np.array([1, 2, 3, 5, 7, 10], dtype=float)
    cds_spreads = np.array([0.0080, 0.0100, 0.0115, 0.0130, 0.0140, 0.0150])
    rf_rates = np.array([0.04, 0.042, 0.043, 0.045, 0.046, 0.047])

    surv_curve = hazard_model.bootstrap_hazard_rates(cds_spreads, cds_tenors, rf_rates)
    for i, t in enumerate(cds_tenors):
        print(f"  T={t:.0f}y | lambda={surv_curve.hazard_rates[i]:.4f} "
              f"| Q(0,T)={surv_curve.survival_probs[i]:.4f} "
              f"| PD={1-surv_curve.survival_probs[i]:.4%}")

    # --- 3. Vasicek Portfolio Model ---
    print("\n[3/6] Vasicek Single-Factor Model (Basel IRB)")
    print("-" * 40)
    pd, lgd = 0.02, 0.45
    rho = VasicekPortfolioModel.basel_correlation(pd, "corporate")
    vasicek = VasicekPortfolioModel(pd=pd, lgd=lgd, rho=rho)
    vresult = vasicek.full_analysis()
    print(f"  PD={pd:.2%} | LGD={lgd:.0%} | rho={rho:.4f}")
    print(f"  Expected Loss:     {vresult.expected_loss:.4%}")
    print(f"  Credit VaR 99.9%:  {vresult.credit_var:.4%}")
    print(f"  Economic Capital:  {vresult.economic_capital:.4%}")
    print(f"  Basel IRB Capital: {vasicek.basel_irb_capital():.4%}")

    # --- 4. CDS Pricing ---
    print("\n[4/6] CDS Pricing")
    print("-" * 40)
    cds = CDSPricer(recovery_rate=0.40)
    fair_s = cds.fair_spread(surv_curve.survival_probs, rf_rates, cds_tenors)
    print(f"  Fair CDS spread: {fair_s:.1f} bps")
    valuation = cds.mark_to_market(
        contract_spread_bps=150, survival_probs=surv_curve.survival_probs,
        rf_rates=rf_rates, tenors=cds_tenors, notional=10_000_000,
    )
    print(f"  MTM (buyer): ${valuation.mtm_value:,.0f}")
    print(f"  DV01: ${cds.cds_dv01(surv_curve.survival_probs, rf_rates, cds_tenors):,.0f}")

    # --- 5. CreditMetrics ---
    print("\n[5/6] CreditMetrics Migration Simulation")
    print("-" * 40)
    np.random.seed(42)
    n_ob = 100
    cm = CreditMetricsEngine(seed=42)
    exposures = np.random.uniform(1e6, 20e6, n_ob)
    ratings = np.random.choice([2, 3, 4, 5], size=n_ob, p=[0.3, 0.3, 0.25, 0.15])
    lgds = np.full(n_ob, 0.45)
    sectors = np.random.choice(5, size=n_ob)
    corr = cm.build_correlation_matrix(n_ob, 0.30, 0.10, sectors.tolist())

    cm_result = cm.simulate(exposures, ratings, lgds, corr, n_simulations=20_000)
    print(f"  Portfolio: {n_ob} obligors | ${exposures.sum()/1e6:.0f}M total exposure")
    print(f"  Expected Loss:    ${cm_result.expected_loss:,.0f}")
    print(f"  Credit VaR 95%:   ${cm_result.credit_var_95:,.0f}")
    print(f"  Credit VaR 99%:   ${cm_result.credit_var_99:,.0f}")
    print(f"  Credit VaR 99.9%: ${cm_result.credit_var_999:,.0f}")

    # --- 6. Credit VaR Engine ---
    print("\n[6/6] Credit VaR with Stress Testing")
    print("-" * 40)
    engine = CreditVaREngine(seed=42)
    pds = np.random.uniform(0.005, 0.05, n_ob)
    rhos = np.full(n_ob, 0.20)

    stress = {
        "mild_recession": {"pd_mult": 2.0, "lgd_add": 0.05},
        "severe_recession": {"pd_mult": 4.0, "lgd_add": 0.15, "rho_mult": 1.3},
    }
    stress_results = engine.stress_test(exposures, pds, lgds, rhos, stress, 30_000)
    for name, res in stress_results.items():
        print(f"  {name:20s} | EL=${res.expected_loss:>12,.0f} "
              f"| CVaR99.9=${res.credit_var_999:>12,.0f} "
              f"| EC=${res.economic_capital:>12,.0f}")

    conc = engine.concentration_risk(exposures, pds, lgds)
    print(f"\n  Concentration: HHI={conc['hhi']:.4f} | "
          f"Eff.N={conc['effective_n']:.0f} | Top5={conc['top5_share']:.1%}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

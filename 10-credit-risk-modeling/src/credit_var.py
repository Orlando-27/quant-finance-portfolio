"""
Credit VaR Monte Carlo Engine
==============================
Monte Carlo simulation for portfolio Credit VaR with heterogeneous
obligors, sector correlations, and importance sampling for tail
estimation.

Credit VaR = Quantile_alpha(Loss) - Expected_Loss

This captures unexpected loss: the loss beyond what is already
provisioned for (expected loss).

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CreditVaRResult:
    """Container for Credit VaR simulation results."""
    expected_loss: float
    credit_var_95: float
    credit_var_99: float
    credit_var_999: float
    economic_capital: float  # CVaR_999 - EL
    loss_distribution: np.ndarray
    n_defaults_distribution: np.ndarray
    tail_losses: np.ndarray  # Losses beyond VaR_99


class CreditVaREngine:
    """
    Monte Carlo engine for heterogeneous portfolio Credit VaR.

    Unlike Vasicek (homogeneous large portfolio), this handles:
        - Different PDs per obligor
        - Different LGDs and exposures
        - Sector-based correlation structure
        - Concentration risk

    Uses the single-factor Gaussian copula:
        X_i = sqrt(rho_i) * Z + sqrt(1 - rho_i) * epsilon_i

    Default if X_i < N_inv(PD_i)
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def simulate(self, exposures: np.ndarray,
                 pds: np.ndarray,
                 lgds: np.ndarray,
                 rhos: np.ndarray,
                 n_simulations: int = 100_000,
                 sectors: np.ndarray = None,
                 sector_correlations: np.ndarray = None) -> CreditVaRResult:
        """
        Run Credit VaR Monte Carlo simulation.

        Args:
            exposures:   (N,) exposure at default for each obligor.
            pds:         (N,) probability of default for each obligor.
            lgds:        (N,) loss given default for each obligor.
            rhos:        (N,) asset correlation to systematic factor.
            n_simulations: Number of Monte Carlo paths.
            sectors:     (N,) sector index for each obligor (optional).
            sector_correlations: (S, S) matrix if multi-sector model.

        Returns:
            CreditVaRResult with full distribution.
        """
        n_obligors = len(exposures)
        default_thresholds = norm.ppf(pds)

        losses = np.zeros(n_simulations)
        n_defaults = np.zeros(n_simulations, dtype=int)

        # Multi-sector or single-factor
        if sectors is not None and sector_correlations is not None:
            n_sectors = sector_correlations.shape[0]
            L_sec = np.linalg.cholesky(sector_correlations)
        else:
            sectors = np.zeros(n_obligors, dtype=int)
            n_sectors = 1
            L_sec = np.array([[1.0]])

        for sim in range(n_simulations):
            # Draw sector factors
            z_raw = self.rng.standard_normal(n_sectors)
            z_sectors = L_sec @ z_raw

            sim_loss = 0.0
            sim_defaults = 0

            for i in range(n_obligors):
                z_sys = z_sectors[sectors[i]]
                eps_i = self.rng.standard_normal()
                x_i = np.sqrt(rhos[i]) * z_sys + np.sqrt(1 - rhos[i]) * eps_i

                if x_i < default_thresholds[i]:
                    sim_loss += exposures[i] * lgds[i]
                    sim_defaults += 1

            losses[sim] = sim_loss
            n_defaults[sim] = sim_defaults

        el = losses.mean()
        var_999 = np.percentile(losses, 99.9)

        return CreditVaRResult(
            expected_loss=el,
            credit_var_95=np.percentile(losses, 95),
            credit_var_99=np.percentile(losses, 99),
            credit_var_999=var_999,
            economic_capital=var_999 - el,
            loss_distribution=losses,
            n_defaults_distribution=n_defaults,
            tail_losses=losses[losses > np.percentile(losses, 99)],
        )

    def concentration_risk(self, exposures: np.ndarray,
                           pds: np.ndarray,
                           lgds: np.ndarray) -> Dict[str, float]:
        """
        Analyze portfolio concentration risk.

        Metrics:
            - HHI (Herfindahl-Hirschman Index) of exposures
            - Top-N exposure share
            - Granularity adjustment (Gordy & Lutkebohmert)
        """
        total = exposures.sum()
        shares = exposures / total

        hhi = np.sum(shares ** 2)
        top5 = np.sort(shares)[-5:].sum()
        top10 = np.sort(shares)[-10:].sum()

        # Effective number of exposures
        eff_n = 1.0 / hhi

        # Granularity adjustment (simplified)
        # GA = (1/2) * sum(s_i^2 * PD_i * LGD_i^2 * (1-PD_i))
        ga = 0.5 * np.sum(shares**2 * pds * lgds**2 * (1 - pds))

        return {
            "hhi": hhi,
            "effective_n": eff_n,
            "top5_share": top5,
            "top10_share": top10,
            "granularity_adjustment": ga,
        }

    def stress_test(self, exposures: np.ndarray,
                    pds: np.ndarray,
                    lgds: np.ndarray,
                    rhos: np.ndarray,
                    stress_scenarios: Dict[str, Dict],
                    n_simulations: int = 50_000) -> Dict[str, CreditVaRResult]:
        """
        Run Credit VaR under multiple stress scenarios.

        Args:
            stress_scenarios: Dict mapping scenario names to parameter
                overrides. E.g., {'recession': {'pd_mult': 3.0, 'lgd_add': 0.1}}

        Returns:
            Dict mapping scenario name to CreditVaRResult.
        """
        results = {}

        # Base case
        results["base"] = self.simulate(exposures, pds, lgds, rhos, n_simulations)

        for name, params in stress_scenarios.items():
            s_pds = np.minimum(pds * params.get("pd_mult", 1.0), 0.99)
            s_lgds = np.minimum(lgds + params.get("lgd_add", 0.0), 1.0)
            s_rhos = np.minimum(rhos * params.get("rho_mult", 1.0), 0.99)

            results[name] = self.simulate(
                exposures, s_pds, s_lgds, s_rhos, n_simulations
            )

        return results

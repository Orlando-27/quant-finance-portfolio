"""
Credit Default Swap (CDS) Pricing
===================================
Fair spread computation, mark-to-market valuation, risky annuity
(DV01), and CDS basis analysis.

A CDS is an insurance contract against default:
    - Protection buyer pays periodic premium (spread * notional)
    - Protection seller pays (1-R) * notional upon default

References:
    Hull, J. & White, A. (2000). Valuing Credit Default Swaps.
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CDSValuation:
    """Container for CDS valuation results."""
    fair_spread: float       # Par spread in basis points
    mtm_value: float         # Mark-to-market value (buyer perspective)
    premium_leg_pv: float    # PV of premium payments
    protection_leg_pv: float # PV of protection payment
    risky_annuity: float     # DV01 = risky PV01
    accrued_premium: float   # Accrued if default mid-period


class CDSPricer:
    """
    CDS pricing engine using hazard rate framework.

    Fair spread equates premium leg to protection leg:
        s* = Protection Leg PV / Risky Annuity

    Premium Leg PV:
        PL = s * sum_{i=1}^{N} delta_i * D(0,t_i) * Q(0,t_i)

    Protection Leg PV:
        PtL = (1-R) * sum_{i=1}^{N} D(0,t_i) * [Q(0,t_{i-1}) - Q(0,t_i)]

    where D = risk-free discount factor, Q = survival probability.
    """

    def __init__(self, recovery_rate: float = 0.40):
        self.R = recovery_rate

    def fair_spread(self, survival_probs: np.ndarray,
                    rf_rates: np.ndarray,
                    tenors: np.ndarray) -> float:
        """
        Compute fair CDS spread (par spread).

        Args:
            survival_probs: Q(0, t_i) for each payment date.
            rf_rates:       Risk-free zero rates for each tenor.
            tenors:         Payment times in years.

        Returns:
            Fair spread in basis points.
        """
        risky_ann = self._risky_annuity(survival_probs, rf_rates, tenors)
        prot_leg = self._protection_leg(survival_probs, rf_rates, tenors)
        return (prot_leg / risky_ann) * 10_000  # Convert to bps

    def mark_to_market(self, contract_spread_bps: float,
                       survival_probs: np.ndarray,
                       rf_rates: np.ndarray,
                       tenors: np.ndarray,
                       notional: float = 1_000_000) -> CDSValuation:
        """
        Mark-to-market valuation of an existing CDS position.

        MTM (buyer) = Protection Leg PV - Premium Leg PV
                    = (s* - s_contract) * Risky Annuity * Notional

        Args:
            contract_spread_bps: Contracted spread in basis points.
            survival_probs:      Current survival probabilities.
            rf_rates:            Current risk-free zero rates.
            tenors:              Remaining payment dates.
            notional:            Contract notional.

        Returns:
            CDSValuation with all components.
        """
        s_contract = contract_spread_bps / 10_000
        risky_ann = self._risky_annuity(survival_probs, rf_rates, tenors)
        prot_pv = self._protection_leg(survival_probs, rf_rates, tenors)
        prem_pv = s_contract * risky_ann

        fair_s_bps = (prot_pv / risky_ann) * 10_000
        mtm = (prot_pv - prem_pv) * notional

        return CDSValuation(
            fair_spread=fair_s_bps,
            mtm_value=mtm,
            premium_leg_pv=prem_pv * notional,
            protection_leg_pv=prot_pv * notional,
            risky_annuity=risky_ann,
            accrued_premium=self._accrued(s_contract, tenors),
        )

    def cds_dv01(self, survival_probs: np.ndarray,
                 rf_rates: np.ndarray,
                 tenors: np.ndarray,
                 notional: float = 1_000_000) -> float:
        """
        CDS DV01 (risky PV01): change in MTM for 1 bp spread change.

            DV01 = Risky Annuity * Notional / 10,000
        """
        risky_ann = self._risky_annuity(survival_probs, rf_rates, tenors)
        return risky_ann * notional / 10_000

    def _risky_annuity(self, Q: np.ndarray, rf: np.ndarray,
                       tenors: np.ndarray) -> float:
        """Premium leg per unit spread = sum(delta_i * df_i * Q_i)."""
        ann = 0.0
        for i in range(len(tenors)):
            dt = tenors[i] - (tenors[i-1] if i > 0 else 0.0)
            df = np.exp(-rf[i] * tenors[i])
            ann += dt * df * Q[i]
        return ann

    def _protection_leg(self, Q: np.ndarray, rf: np.ndarray,
                        tenors: np.ndarray) -> float:
        """Protection leg PV = (1-R) * sum(df_i * (Q_{i-1} - Q_i))."""
        prot = 0.0
        for i in range(len(tenors)):
            Q_prev = Q[i-1] if i > 0 else 1.0
            df = np.exp(-rf[i] * tenors[i])
            prot += df * (Q_prev - Q[i])
        return prot * (1.0 - self.R)

    def _accrued(self, spread: float, tenors: np.ndarray) -> float:
        """Approximate accrued premium (half of last period)."""
        if len(tenors) < 2:
            return 0.0
        last_period = tenors[-1] - tenors[-2]
        return spread * last_period * 0.5

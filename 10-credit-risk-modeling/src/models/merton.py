"""
Merton Structural Credit Risk Model (1974)
===========================================
Treats equity as a European call on firm assets. Computes probability
of default, distance to default, credit spreads, and recovery rates.
Includes iterative calibration from equity market data.

References:
    Merton, R. (1974). On the Pricing of Corporate Debt.
    Hull, J. (2018). Options, Futures, and Other Derivatives, Ch. 24.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class MertonResult:
    """Container for Merton model outputs."""
    asset_value: float
    asset_vol: float
    equity_value: float
    debt_value: float
    default_prob_rn: float      # Risk-neutral PD = N(-d2)
    default_prob_physical: float # Physical PD = N(-DD)
    distance_to_default: float  # DD
    credit_spread: float        # In basis points
    recovery_rate: float        # Expected recovery = debt_value / (D * exp(-r*T))
    d1: float
    d2: float


class MertonModel:
    """
    Merton (1974) structural model for credit risk.

    The firm's total value V follows GBM under the physical measure:
        dV = mu * V * dt + sigma_V * V * dW

    Equity E is a call option on V with strike equal to face value of debt D:
        E = V * N(d1) - D * exp(-r*T) * N(d2)

    Key insight: equity holders have limited liability. If V < D at maturity,
    they walk away and bondholders receive V (< D).
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Continuously compounded risk-free rate.
        """
        self.r = risk_free_rate

    def price_equity(self, V: float, sigma_V: float,
                     D: float, T: float) -> float:
        """
        Price equity as a European call on firm assets.

        Args:
            V:       Current firm asset value.
            sigma_V: Asset volatility (annualized).
            D:       Face value of zero-coupon debt.
            T:       Time to debt maturity (years).

        Returns:
            Equity value E.
        """
        d1, d2 = self._d1_d2(V, sigma_V, D, T)
        return V * norm.cdf(d1) - D * np.exp(-self.r * T) * norm.cdf(d2)

    def price_debt(self, V: float, sigma_V: float,
                   D: float, T: float) -> float:
        """
        Price risky debt. Since V = E + B:
            B = V - E = D*exp(-r*T)*N(d2) + V*N(-d1)

        Alternatively, risky debt = risk-free debt - put on V:
            B = D*exp(-r*T) - Put(V, D)
        """
        E = self.price_equity(V, sigma_V, D, T)
        return V - E

    def default_probability(self, V: float, sigma_V: float,
                           D: float, T: float,
                           mu: Optional[float] = None) -> Dict[str, float]:
        """
        Compute default probabilities under both measures.

        Risk-neutral PD:  N(-d2) where d2 uses r
        Physical PD:      N(-DD) where DD uses mu (drift)

        Args:
            V, sigma_V, D, T: Model parameters.
            mu: Physical drift of assets. If None, uses r (risk-neutral only).

        Returns:
            Dictionary with 'risk_neutral' and 'physical' PDs.
        """
        d1, d2 = self._d1_d2(V, sigma_V, D, T)
        pd_rn = norm.cdf(-d2)

        pd_phys = np.nan
        if mu is not None:
            dd = (np.log(V / D) + (mu - 0.5 * sigma_V**2) * T) / \
                 (sigma_V * np.sqrt(T))
            pd_phys = norm.cdf(-dd)

        return {"risk_neutral": pd_rn, "physical": pd_phys}

    def distance_to_default(self, V: float, sigma_V: float,
                           D: float, T: float,
                           mu: float = None) -> float:
        """
        Distance to Default (DD): number of standard deviations
        the firm is from the default point.

            DD = [ln(V/D) + (mu - sigma_V^2/2) * T] / (sigma_V * sqrt(T))

        Moody's KMV uses this as the primary credit metric.
        """
        drift = mu if mu is not None else self.r
        return (np.log(V / D) + (drift - 0.5 * sigma_V**2) * T) / \
               (sigma_V * np.sqrt(T))

    def credit_spread(self, V: float, sigma_V: float,
                     D: float, T: float) -> float:
        """
        Credit spread in basis points.

        The yield on risky debt y_risky satisfies:
            B = D * exp(-y_risky * T)
        So:
            spread = y_risky - r = -ln(B / D) / T - r

        Returns spread in basis points (multiply by 10,000).
        """
        B = self.price_debt(V, sigma_V, D, T)
        y_risky = -np.log(B / D) / T
        return (y_risky - self.r) * 10_000

    def calibrate_from_equity(self, E_market: float, sigma_E: float,
                              D: float, T: float,
                              V0: float = None,
                              sigma_V0: float = None) -> MertonResult:
        """
        Iterative calibration of (V, sigma_V) from observed equity
        value and equity volatility.

        System of two equations:
            1. E = V*N(d1) - D*exp(-r*T)*N(d2)         [BSM pricing]
            2. sigma_E = (V/E) * N(d1) * sigma_V        [Ito's lemma]

        Uses scipy.fsolve for simultaneous root finding.

        Args:
            E_market: Observed equity market cap.
            sigma_E:  Observed equity volatility (annualized).
            D:        Face value of debt.
            T:        Time to maturity.
            V0:       Initial guess for V (default: E + D).
            sigma_V0: Initial guess for sigma_V (default: sigma_E * E/(E+D)).

        Returns:
            MertonResult with all computed quantities.
        """
        if V0 is None:
            V0 = E_market + D * np.exp(-self.r * T)
        if sigma_V0 is None:
            sigma_V0 = sigma_E * E_market / V0

        def equations(x):
            V, sig_V = x
            d1, d2 = self._d1_d2(V, sig_V, D, T)
            eq1 = V * norm.cdf(d1) - D * np.exp(-self.r * T) * norm.cdf(d2) - E_market
            eq2 = (V / E_market) * norm.cdf(d1) * sig_V - sigma_E
            return [eq1, eq2]

        V_sol, sigma_V_sol = fsolve(equations, [V0, sigma_V0], full_output=False)
        V_sol = abs(V_sol)
        sigma_V_sol = abs(sigma_V_sol)

        d1, d2 = self._d1_d2(V_sol, sigma_V_sol, D, T)
        E_model = self.price_equity(V_sol, sigma_V_sol, D, T)
        B_model = V_sol - E_model
        dd = self.distance_to_default(V_sol, sigma_V_sol, D, T)
        spread = self.credit_spread(V_sol, sigma_V_sol, D, T)
        recovery = B_model / (D * np.exp(-self.r * T))

        return MertonResult(
            asset_value=V_sol,
            asset_vol=sigma_V_sol,
            equity_value=E_model,
            debt_value=B_model,
            default_prob_rn=norm.cdf(-d2),
            default_prob_physical=norm.cdf(-dd),
            distance_to_default=dd,
            credit_spread=spread,
            recovery_rate=recovery,
            d1=d1, d2=d2,
        )

    def sensitivity_analysis(self, V: float, sigma_V: float,
                            D: float, T: float,
                            param: str = "leverage",
                            n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Analyze sensitivity of PD and spread to key parameters.

        Args:
            param: One of 'leverage', 'volatility', 'maturity'.
            n_points: Number of points in the grid.

        Returns:
            Dict with 'x', 'pd', 'spread', 'dd' arrays.
        """
        leverage = D / V

        if param == "leverage":
            x = np.linspace(0.1, 0.95, n_points)
            D_grid = x * V
            results = [self._compute_metrics(V, sigma_V, d, T) for d in D_grid]
        elif param == "volatility":
            x = np.linspace(0.05, 0.80, n_points)
            results = [self._compute_metrics(V, s, D, T) for s in x]
        elif param == "maturity":
            x = np.linspace(0.25, 10.0, n_points)
            results = [self._compute_metrics(V, sigma_V, D, t) for t in x]
        else:
            raise ValueError(f"Unknown param: {param}")

        return {
            "x": x,
            "pd": np.array([r[0] for r in results]),
            "spread": np.array([r[1] for r in results]),
            "dd": np.array([r[2] for r in results]),
        }

    def _d1_d2(self, V: float, sigma_V: float,
               D: float, T: float) -> Tuple[float, float]:
        """Compute d1 and d2 for the Merton model."""
        d1 = (np.log(V / D) + (self.r + 0.5 * sigma_V**2) * T) / \
             (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        return d1, d2

    def _compute_metrics(self, V, sigma_V, D, T):
        """Helper: compute (PD, spread, DD) for a parameter set."""
        d1, d2 = self._d1_d2(V, sigma_V, D, T)
        pd = norm.cdf(-d2)
        spread = self.credit_spread(V, sigma_V, D, T)
        dd = self.distance_to_default(V, sigma_V, D, T)
        return pd, spread, dd

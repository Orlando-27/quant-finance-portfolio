"""
Vasicek Single-Factor Portfolio Credit Risk Model
===================================================
Analytical framework for portfolio loss distribution under the
asymptotic single-risk-factor (ASRF) model. Foundation of the
Basel II/III Internal Ratings-Based (IRB) approach.

References:
    Vasicek, O. (2002). The Distribution of Loan Portfolio Value. Risk.
    Basel Committee (2006). Basel II IRB Approach.
    Gordy, M. (2003). A Risk-Factor Model Foundation for
    Ratings-Based Bank Capital Rules.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VasicekResult:
    """Container for Vasicek model outputs."""
    expected_loss: float
    unexpected_loss: float
    credit_var: float        # VaR at given confidence
    economic_capital: float  # = Credit VaR - EL
    loss_quantiles: Dict[str, float]
    conditional_pd_at_var: float


class VasicekPortfolioModel:
    """
    Vasicek single-factor model for large homogeneous portfolios.

    Each obligor i has asset return:
        X_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i

    where:
        Z ~ N(0,1)         : systematic (market) factor
        epsilon_i ~ N(0,1)  : idiosyncratic risk
        rho                 : asset correlation (0 < rho < 1)

    Default occurs when X_i < N_inv(PD), i.e., the asset return
    falls below the default threshold.

    Conditional on Z, the portfolio loss fraction converges to:
        L(Z) = LGD * N( (N_inv(PD) - sqrt(rho)*Z) / sqrt(1-rho) )

    The unconditional loss distribution has CDF:
        P(L <= x) = N( (sqrt(1-rho)*N_inv(x/LGD) - N_inv(PD)) / sqrt(rho) )
    """

    def __init__(self, pd: float, lgd: float, rho: float):
        """
        Args:
            pd:  Probability of default (through-the-cycle).
            lgd: Loss given default (0 to 1).
            rho: Asset correlation parameter (0 to 1).
        """
        self.pd = pd
        self.lgd = lgd
        self.rho = rho
        self._validate()

    def _validate(self):
        assert 0 < self.pd < 1, "PD must be in (0, 1)"
        assert 0 < self.lgd <= 1, "LGD must be in (0, 1]"
        assert 0 < self.rho < 1, "rho must be in (0, 1)"

    def conditional_pd(self, z: float) -> float:
        """
        Conditional PD given systematic factor realization Z = z.

            PD(z) = N( (N_inv(PD) - sqrt(rho) * z) / sqrt(1-rho) )

        When z is very negative (economic downturn), PD(z) >> PD.
        """
        return norm.cdf(
            (norm.ppf(self.pd) - np.sqrt(self.rho) * z) /
            np.sqrt(1 - self.rho)
        )

    def conditional_loss(self, z: float) -> float:
        """Expected portfolio loss fraction conditional on Z = z."""
        return self.lgd * self.conditional_pd(z)

    def expected_loss(self) -> float:
        """Unconditional expected loss = PD * LGD."""
        return self.pd * self.lgd

    def loss_quantile(self, alpha: float) -> float:
        """
        Analytical loss quantile from Vasicek formula.

        The alpha-quantile of the loss distribution:
            VaR_alpha = LGD * N( (N_inv(PD) + sqrt(rho)*N_inv(alpha)) / sqrt(1-rho) )

        This is the Basel II/III IRB capital formula.

        Args:
            alpha: Confidence level (e.g., 0.999 for 99.9%).

        Returns:
            Loss quantile as fraction of portfolio.
        """
        return self.lgd * norm.cdf(
            (norm.ppf(self.pd) + np.sqrt(self.rho) * norm.ppf(alpha)) /
            np.sqrt(1 - self.rho)
        )

    def credit_var(self, alpha: float = 0.999) -> float:
        """Credit VaR = loss quantile - expected loss."""
        return self.loss_quantile(alpha) - self.expected_loss()

    def economic_capital(self, alpha: float = 0.999) -> float:
        """Economic capital = Credit VaR (unexpected loss buffer)."""
        return self.credit_var(alpha)

    def loss_distribution_cdf(self, x: float) -> float:
        """
        CDF of the portfolio loss distribution.

            P(L <= x) = N( (sqrt(1-rho)*N_inv(x/LGD) - N_inv(PD)) / sqrt(rho) )
        """
        if x <= 0:
            return 0.0
        if x >= self.lgd:
            return 1.0
        return norm.cdf(
            (np.sqrt(1 - self.rho) * norm.ppf(x / self.lgd) - norm.ppf(self.pd)) /
            np.sqrt(self.rho)
        )

    def loss_distribution_pdf(self, x: float, dx: float = 1e-5) -> float:
        """Numerical PDF via finite differences of the CDF."""
        return (self.loss_distribution_cdf(x + dx) -
                self.loss_distribution_cdf(x - dx)) / (2 * dx)

    def full_analysis(self, confidence_levels: list = None) -> VasicekResult:
        """
        Complete portfolio analysis at multiple confidence levels.

        Args:
            confidence_levels: List of confidence levels.
                Default: [0.95, 0.99, 0.999, 0.9999]
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.999, 0.9999]

        el = self.expected_loss()
        quantiles = {f"{a:.4f}": self.loss_quantile(a) for a in confidence_levels}
        var_999 = self.loss_quantile(0.999)
        ul = var_999 - el

        # Z value at VaR level
        z_var = norm.ppf(0.999)
        cpd = self.conditional_pd(-z_var)  # Negative Z = downturn

        return VasicekResult(
            expected_loss=el,
            unexpected_loss=ul,
            credit_var=var_999,
            economic_capital=ul,
            loss_quantiles=quantiles,
            conditional_pd_at_var=cpd,
        )

    def basel_irb_capital(self, maturity: float = 2.5) -> float:
        """
        Basel II/III IRB capital requirement.

        K = [LGD*N((N_inv(PD)+sqrt(rho)*N_inv(0.999))/sqrt(1-rho)) - PD*LGD]
            * (1 + (M-2.5)*b) / (1 - 1.5*b)

        where b = (0.11852 - 0.05478*ln(PD))^2 is the maturity adjustment
        and M is the effective maturity.

        Args:
            maturity: Effective maturity in years.

        Returns:
            Capital requirement as fraction of exposure.
        """
        b = (0.11852 - 0.05478 * np.log(self.pd)) ** 2
        k_base = self.loss_quantile(0.999) - self.expected_loss()
        maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        return k_base * maturity_adj

    @staticmethod
    def basel_correlation(pd: float, asset_class: str = "corporate") -> float:
        """
        Basel II prescribed asset correlation as function of PD.

        For corporates:
            rho = 0.12*(1-exp(-50*PD))/(1-exp(-50)) + 0.24*(1-(1-exp(-50*PD))/(1-exp(-50)))

        This creates a decreasing rho(PD): higher PD -> lower correlation.
        Rationale: high-PD firms default more from idiosyncratic reasons.

        Args:
            pd: Probability of default.
            asset_class: 'corporate', 'retail_mortgage', 'retail_revolving', 'sme'.
        """
        if asset_class == "corporate":
            factor = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
            return 0.12 * factor + 0.24 * (1 - factor)
        elif asset_class == "retail_mortgage":
            return 0.15
        elif asset_class == "retail_revolving":
            return 0.04
        elif asset_class == "sme":
            factor = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
            return 0.12 * factor + 0.24 * (1 - factor) - \
                   0.04 * (1 - min(50.0, 5.0) / 50.0)  # Size adjustment
        else:
            raise ValueError(f"Unknown asset class: {asset_class}")

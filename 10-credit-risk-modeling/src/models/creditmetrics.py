"""
CreditMetrics Portfolio Credit Risk Framework
===============================================
Full-simulation approach to portfolio credit risk using correlated
rating migrations. Implements the JP Morgan CreditMetrics (1997)
methodology with Cholesky-correlated asset returns.

References:
    Gupton, G., Finger, C., & Bhatia, M. (1997). CreditMetrics
    Technical Document. JP Morgan.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Standard S&P transition matrix (1-year, %)
# Rows: current rating, Columns: AAA AA A BBB BB B CCC Default
DEFAULT_TRANSITION_MATRIX = np.array([
    [90.81, 8.33, 0.68, 0.06, 0.12, 0.00, 0.00, 0.00],  # AAA
    [0.70, 90.65, 7.79, 0.64, 0.06, 0.14, 0.02, 0.00],   # AA
    [0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06],   # A
    [0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18],   # BBB
    [0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06],   # BB
    [0.00, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20],   # B
    [0.22, 0.00, 0.22, 1.30, 2.38, 11.24, 64.86, 19.79], # CCC
]) / 100.0

RATING_LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
RATING_INDEX = {label: i for i, label in enumerate(RATING_LABELS)}


@dataclass
class CreditMetricsResult:
    """Container for CreditMetrics simulation output."""
    portfolio_losses: np.ndarray   # Simulated loss distribution
    expected_loss: float
    loss_std: float
    credit_var_95: float
    credit_var_99: float
    credit_var_999: float
    migration_counts: Dict[str, Dict[str, int]]  # Per-obligor migrations


class CreditMetricsEngine:
    """
    CreditMetrics simulation engine for portfolio credit risk.

    Algorithm:
        1. Define obligors with ratings, exposures, LGDs
        2. Build Cholesky matrix from asset correlation structure
        3. For each simulation:
           a. Draw correlated standard normals via Cholesky
           b. Map each normal to a new rating using transition thresholds
           c. Revalue each position under the new rating
           d. Compute portfolio loss
        4. Aggregate simulation results for risk metrics
    """

    def __init__(self, transition_matrix: np.ndarray = None,
                 seed: int = None):
        """
        Args:
            transition_matrix: (K x K+1) matrix where K = number of non-default
                states and last column is default probability.
                Default: Standard S&P 1-year transition matrix.
            seed: Random seed for reproducibility.
        """
        self.tm = transition_matrix if transition_matrix is not None \
            else DEFAULT_TRANSITION_MATRIX
        self.rng = np.random.default_rng(seed)
        self.n_ratings = self.tm.shape[0]

        # Compute migration thresholds from transition probabilities
        self.thresholds = self._compute_thresholds()

    def _compute_thresholds(self) -> np.ndarray:
        """
        Convert transition probabilities to normal distribution thresholds.

        For rating i, the thresholds z_{i,j} satisfy:
            P(migrate to rating j) = N(z_{i,j}) - N(z_{i,j-1})

        Working from default (worst) to best rating, cumulate probabilities
        and invert the normal CDF.
        """
        n_states = self.tm.shape[1]  # Including default
        thresholds = np.full((self.n_ratings, n_states + 1), -np.inf)
        thresholds[:, -1] = np.inf

        for i in range(self.n_ratings):
            cum_prob = 0.0
            # From default (worst) to best, reversed
            for j in range(n_states - 1, -1, -1):
                cum_prob += self.tm[i, j]
                cum_prob = min(cum_prob, 1.0 - 1e-10)
                if j > 0:
                    thresholds[i, j] = norm.ppf(cum_prob)

        return thresholds

    def simulate(self, exposures: np.ndarray,
                 ratings: np.ndarray,
                 lgds: np.ndarray,
                 correlation_matrix: np.ndarray,
                 n_simulations: int = 50_000,
                 spread_curves: Dict[str, np.ndarray] = None) -> CreditMetricsResult:
        """
        Run Monte Carlo simulation for portfolio credit risk.

        Args:
            exposures:          (N,) array of exposure amounts.
            ratings:            (N,) array of current rating indices (0=AAA,...,6=CCC).
            lgds:               (N,) array of loss given default.
            correlation_matrix: (N, N) asset correlation matrix.
            n_simulations:      Number of Monte Carlo paths.
            spread_curves:      Optional spread curves for mark-to-market.
                                If None, only default losses are computed.

        Returns:
            CreditMetricsResult with full loss distribution.
        """
        n_obligors = len(exposures)
        assert correlation_matrix.shape == (n_obligors, n_obligors)

        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(correlation_matrix)

        # Pre-allocate
        losses = np.zeros(n_simulations)
        migration_tracker = {i: {r: 0 for r in RATING_LABELS}
                            for i in range(n_obligors)}

        for sim in range(n_simulations):
            # Draw correlated normals
            z_indep = self.rng.standard_normal(n_obligors)
            z_corr = L @ z_indep

            sim_loss = 0.0
            for i in range(n_obligors):
                # Determine new rating based on threshold mapping
                new_rating = self._map_to_rating(ratings[i], z_corr[i])
                new_label = RATING_LABELS[new_rating]
                migration_tracker[i][new_label] += 1

                if new_rating == self.tm.shape[1] - 1:  # Default
                    sim_loss += exposures[i] * lgds[i]
                elif spread_curves is not None and new_rating != ratings[i]:
                    # Mark-to-market loss from migration
                    mtm_loss = self._mtm_loss(
                        exposures[i], ratings[i], new_rating, spread_curves
                    )
                    sim_loss += mtm_loss

            losses[sim] = sim_loss

        el = losses.mean()
        return CreditMetricsResult(
            portfolio_losses=losses,
            expected_loss=el,
            loss_std=losses.std(),
            credit_var_95=np.percentile(losses, 95),
            credit_var_99=np.percentile(losses, 99),
            credit_var_999=np.percentile(losses, 99.9),
            migration_counts=migration_tracker,
        )

    def _map_to_rating(self, current_rating: int, z: float) -> int:
        """Map a standard normal draw to a new rating via thresholds."""
        thresholds = self.thresholds[current_rating]
        for j in range(len(thresholds) - 1):
            if thresholds[j] <= z < thresholds[j + 1]:
                return j
        return len(thresholds) - 2  # Default

    def _mtm_loss(self, exposure: float, old_rating: int,
                  new_rating: int,
                  spread_curves: Dict[str, np.ndarray]) -> float:
        """
        Mark-to-market loss from rating migration.

        Approximation: loss = exposure * (spread_new - spread_old) * duration
        """
        old_label = RATING_LABELS[old_rating]
        new_label = RATING_LABELS[new_rating]
        if old_label in spread_curves and new_label in spread_curves:
            spread_diff = spread_curves[new_label].mean() - \
                         spread_curves[old_label].mean()
            duration = 4.0  # Simplified average duration
            return exposure * spread_diff * duration
        return 0.0

    def build_correlation_matrix(self, n_obligors: int,
                                intra_sector_rho: float = 0.30,
                                inter_sector_rho: float = 0.10,
                                sectors: List[int] = None) -> np.ndarray:
        """
        Build block-diagonal asset correlation matrix.

        Obligors in the same sector have higher correlation (intra_sector_rho)
        than obligors in different sectors (inter_sector_rho).

        Args:
            n_obligors:        Number of obligors.
            intra_sector_rho:  Correlation within same sector.
            inter_sector_rho:  Correlation across sectors.
            sectors:           Sector assignment for each obligor.

        Returns:
            (N, N) positive semi-definite correlation matrix.
        """
        if sectors is None:
            sectors = [0] * n_obligors  # All same sector

        corr = np.full((n_obligors, n_obligors), inter_sector_rho)
        for i in range(n_obligors):
            corr[i, i] = 1.0
            for j in range(i + 1, n_obligors):
                if sectors[i] == sectors[j]:
                    corr[i, j] = intra_sector_rho
                    corr[j, i] = intra_sector_rho

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(corr)
        if np.min(eigvals) < 0:
            corr += (-np.min(eigvals) + 1e-6) * np.eye(n_obligors)

        return corr

"""
Utilities for Credit Risk Modeling
====================================
Rating data, transition matrix tools, and visualization helpers.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

RATING_LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}
SPECULATIVE_GRADE = {"BB", "B", "CCC"}

# Typical credit spreads by rating (bps) for 5-year horizon
TYPICAL_SPREADS = {
    "AAA": 20, "AA": 40, "A": 70, "BBB": 130,
    "BB": 250, "B": 450, "CCC": 800,
}

# Typical recovery rates by seniority
RECOVERY_RATES = {
    "senior_secured": 0.55,
    "senior_unsecured": 0.40,
    "subordinated": 0.25,
    "junior_subordinated": 0.15,
}


def validate_transition_matrix(tm: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Validate a transition matrix.

    Requirements:
        1. All entries non-negative
        2. Each row sums to 1.0
        3. Default is absorbing (last column row sums handled separately)
    """
    if np.any(tm < -tol):
        return False
    row_sums = tm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False
    return True


def multi_period_transition(tm: np.ndarray, n_years: int) -> np.ndarray:
    """
    Compute n-year transition matrix by matrix exponentiation.

    T(n) = T(1)^n

    Note: This assumes time-homogeneous (constant) transition probabilities.
    """
    result = np.eye(tm.shape[0])
    for _ in range(n_years):
        result = result @ tm
    return result


def cumulative_default_rates(tm: np.ndarray,
                             max_years: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute cumulative default rates for each rating over time.

    Returns dict mapping rating label to array of cumulative PDs.
    """
    n_ratings = tm.shape[0]
    default_col = tm.shape[1] - 1
    cdr = {}

    for i in range(n_ratings):
        cum_defaults = np.zeros(max_years)
        tm_n = np.eye(tm.shape[0])
        # Extend tm to square if needed
        tm_ext = np.zeros((tm.shape[1], tm.shape[1]))
        tm_ext[:n_ratings, :] = tm
        tm_ext[default_col, default_col] = 1.0  # Default is absorbing

        power = np.eye(tm_ext.shape[0])
        for yr in range(max_years):
            power = power @ tm_ext
            cum_defaults[yr] = power[i, default_col]

        cdr[RATING_LABELS[i]] = cum_defaults

    return cdr


def plot_loss_distribution(losses: np.ndarray,
                          var_levels: Dict[str, float] = None,
                          title: str = "Portfolio Loss Distribution",
                          figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot portfolio loss distribution with VaR markers.

    Args:
        losses:     Array of simulated losses.
        var_levels: Dict mapping label to VaR value (e.g., {'VaR 99%': 5e6}).
        title:      Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(losses, bins=200, density=True, alpha=0.7, color="steelblue",
             edgecolor="none")
    ax1.set_xlabel("Portfolio Loss")
    ax1.set_ylabel("Density")
    ax1.set_title(title)

    if var_levels:
        colors = ["orange", "red", "darkred", "black"]
        for idx, (label, val) in enumerate(var_levels.items()):
            c = colors[idx % len(colors)]
            ax1.axvline(val, color=c, linestyle="--", linewidth=1.5, label=label)
        ax1.legend(fontsize=8)

    # ECDF
    sorted_l = np.sort(losses)
    ecdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
    ax2.plot(sorted_l, ecdf, color="steelblue", linewidth=0.8)
    ax2.set_xlabel("Portfolio Loss")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Empirical CDF")
    ax2.grid(True, alpha=0.3)

    if var_levels:
        for idx, (label, val) in enumerate(var_levels.items()):
            c = colors[idx % len(colors)]
            ax2.axvline(val, color=c, linestyle="--", linewidth=1.5, label=label)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig

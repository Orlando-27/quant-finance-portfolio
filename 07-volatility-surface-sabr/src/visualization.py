"""
================================================================================
VISUALIZATION: VOLATILITY SURFACES, SMILES & DIAGNOSTICS
================================================================================
Professional plotting functions for implied and local volatility surfaces,
smile comparison, term structure, and calibration diagnostics.

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple

from src.surface import VolSurface


COLORS = {
    "primary": "#003366", "secondary": "#0066CC", "accent": "#FF6600",
    "positive": "#2E8B57", "negative": "#DC143C", "neutral": "#666666",
    "grid": "#E0E0E0",
}
LINE_COLORS = ["#003366", "#0066CC", "#FF6600", "#2E8B57", "#9B59B6", "#E74C3C"]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "grid.color": COLORS["grid"],
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
    })


def plot_surface_3d(surface, strikes, expiries, title="Implied Volatility Surface",
                     save_path=None):
    """3D surface plot of implied volatility."""
    setup_style()
    vol_grid = surface.get_vol_grid(strikes, expiries)
    K_mesh, T_mesh = np.meshgrid(strikes, expiries, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, vol_grid * 100,
                            cmap=cm.coolwarm, alpha=0.85,
                            linewidth=0.3, edgecolor="grey")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry (years)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="IV (%)")
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_smile(strikes, vols, market_strikes=None, market_vols=None,
               title="Volatility Smile", forward=None, save_path=None):
    """Plot a single volatility smile with optional market data overlay."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strikes, vols * 100, color=COLORS["primary"], linewidth=2, label="Model")

    if market_strikes is not None and market_vols is not None:
        ax.scatter(market_strikes, np.array(market_vols) * 100,
                    color=COLORS["accent"], s=60, zorder=5,
                    edgecolors="white", linewidth=0.5, label="Market")
    if forward is not None:
        ax.axvline(forward, color=COLORS["neutral"], linestyle="--",
                    alpha=0.5, label=f"Forward = {forward:.1f}")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_smile_comparison(strikes, vol_dict, market_strikes=None,
                           market_vols=None, title="Model Comparison",
                           save_path=None):
    """Compare multiple model smiles on the same plot."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, vols) in enumerate(vol_dict.items()):
        ax.plot(strikes, vols * 100, color=LINE_COLORS[i % len(LINE_COLORS)],
                linewidth=1.8, label=name)

    if market_strikes is not None and market_vols is not None:
        ax.scatter(market_strikes, np.array(market_vols) * 100,
                    color="black", s=50, marker="x", zorder=5,
                    linewidth=1.5, label="Market")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_term_structure(expiries, atm_vols, skews=None,
                         title="ATM Term Structure", save_path=None):
    """Plot ATM volatility and skew term structures."""
    setup_style()
    n_plots = 2 if skews is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(expiries, atm_vols * 100, color=COLORS["primary"],
                  linewidth=2, marker="o", markersize=5)
    axes[0].set_ylabel("ATM Vol (%)")
    axes[0].set_title(title)

    if skews is not None:
        colors = [COLORS["negative"] if s < 0 else COLORS["positive"] for s in skews]
        axes[1].bar(expiries, skews, width=expiries.min() * 0.3,
                     color=colors, edgecolor="white")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_ylabel("ATM Skew")
        axes[1].set_title("Skew Term Structure")

    axes[-1].set_xlabel("Expiry (years)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_local_vs_implied(strikes, implied_vol, local_vol,
                           title="Local Vol vs Implied Vol", save_path=None):
    """Compare local volatility to implied volatility."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strikes, implied_vol * 100, color=COLORS["primary"],
            linewidth=2, label="Implied Vol")
    ax.plot(strikes, local_vol * 100, color=COLORS["accent"],
            linewidth=2, linestyle="--", label="Local Vol")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Volatility (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_calibration_errors(results, title="Calibration Errors by Expiry",
                             save_path=None):
    """Plot calibration errors per expiry slice."""
    setup_style()
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (T, (strikes, errors)) in enumerate(sorted(results.items())):
        ax = axes[i]
        colors = [COLORS["negative"] if e < 0 else COLORS["positive"] for e in errors]
        ax.bar(strikes, errors * 10000, color=colors, edgecolor="white",
               width=(strikes.max() - strikes.min()) / len(strikes) * 0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"T = {T:.2f}y")
        ax.set_xlabel("Strike")

    axes[0].set_ylabel("Error (bps)")
    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

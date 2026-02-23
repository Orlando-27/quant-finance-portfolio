"""
Publication-quality visualizations for Black-Scholes & Greeks.

Figures generated:
    01_volatility_smile.png        - Implied vol smile across strikes
    02_greeks_surface.png          - 3D Greeks as f(S, T)
    03_delta_gamma_profile.png     - Delta and Gamma vs moneyness
    04_theta_time_decay.png        - Theta acceleration near expiry
    05_pnl_decomposition.png       - P&L attribution via Greeks
    06_implied_vs_realized.png     - IV vs HV comparison
    07_put_call_parity.png         - Parity verification scatter

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
NAVY   = "#1a1a2e"
TEAL   = "#16697a"
CORAL  = "#db6400"
GOLD   = "#c5a880"
SLATE  = "#4a4e69"
COLORS = [NAVY, TEAL, CORAL, GOLD, SLATE, "#2d6a4f", "#e07a5f", "#3d405b"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})


def _watermark(fig):
    fig.text(0.99, 0.01, "J. Bobadilla | CQF", fontsize=7,
             color="gray", alpha=0.5, ha="right", va="bottom")


def _save(fig, proj, name):
    path = os.path.join(proj, "outputs", "figures", name)
    fig.savefig(path)
    plt.close(fig)
    return path


def _bs_price(S, K, T, r, sigma, option="call"):
    """Black-Scholes closed-form price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_greeks(S, K, T, r, sigma):
    """Return dict of Greeks for a call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "rho": rho, "d1": d1, "d2": d2}


# ---------------------------------------------------------------------------
# Figure 01: Volatility smile
# ---------------------------------------------------------------------------
def plot_volatility_smile(proj=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    K_range = np.linspace(70, 130, 200)
    S0 = 100
    for i, T in enumerate([0.1, 0.25, 0.5, 1.0]):
        moneyness = K_range / S0
        # Synthetic smile: base vol + skew + convexity
        iv = 0.20 + 0.10 * (moneyness - 1.0)**2 + 0.05 * np.exp(-T) * (moneyness - 1.0)**2
        ax.plot(K_range, iv * 100, color=COLORS[i], lw=2, label=f"T = {T:.2f}y")
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Implied Volatility Smile Across Maturities")
    ax.legend(frameon=True, fancybox=True)
    ax.axvline(100, color="gray", ls=":", alpha=0.5, label="ATM")
    _watermark(fig)
    return _save(fig, proj, "01_volatility_smile.png")


# ---------------------------------------------------------------------------
# Figure 02: Greeks 3D surface (Delta as f(S,T))
# ---------------------------------------------------------------------------
def plot_greeks_surface(proj=None):
    S = np.linspace(60, 140, 80)
    T = np.linspace(0.05, 2.0, 80)
    SS, TT = np.meshgrid(S, T)
    K, r, sigma = 100, 0.05, 0.20
    d1 = (np.log(SS / K) + (r + 0.5 * sigma**2) * TT) / (sigma * np.sqrt(TT))
    delta = norm.cdf(d1)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(SS, TT, delta, cmap="RdYlBu_r", alpha=0.85,
                           edgecolor="none", antialiased=True)
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Time to Expiry (T)")
    ax.set_zlabel("Delta")
    ax.set_title("Call Delta Surface: f(S, T)")
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.view_init(elev=25, azim=-45)
    _watermark(fig)
    return _save(fig, proj, "02_greeks_surface.png")


# ---------------------------------------------------------------------------
# Figure 03: Delta & Gamma profiles
# ---------------------------------------------------------------------------
def plot_delta_gamma(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    S = np.linspace(60, 140, 300)
    K, T, r, sigma = 100, 0.5, 0.05, 0.20

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    ax1.plot(S, delta_call, color=TEAL, lw=2.5, label="Call Delta")
    ax1.plot(S, delta_put, color=CORAL, lw=2.5, label="Put Delta")
    ax1.axhline(0, color="gray", ls=":")
    ax1.axvline(K, color="gray", ls=":", alpha=0.4)
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Delta")
    ax1.set_title("Delta Profile")
    ax1.legend()

    ax2.plot(S, gamma, color=NAVY, lw=2.5)
    ax2.fill_between(S, gamma, alpha=0.15, color=NAVY)
    ax2.axvline(K, color="gray", ls=":", alpha=0.4)
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Gamma")
    ax2.set_title("Gamma Profile (Peak at ATM)")
    fig.suptitle("Delta & Gamma as Functions of Spot Price", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _watermark(fig)
    return _save(fig, proj, "03_delta_gamma_profile.png")


# ---------------------------------------------------------------------------
# Figure 04: Theta time decay
# ---------------------------------------------------------------------------
def plot_theta_decay(proj=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    T_arr = np.linspace(0.01, 1.0, 300)
    K, S, r, sigma = 100, 100, 0.05, 0.20

    for i, moneyness in enumerate([90, 100, 110]):
        prices = [_bs_price(S, moneyness, t, r, sigma) for t in T_arr]
        ax.plot(T_arr * 252, prices, color=COLORS[i], lw=2,
                label=f"K={moneyness} ({'ITM' if moneyness<S else 'ATM' if moneyness==S else 'OTM'})")

    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("Option Price ($)")
    ax.set_title("Time Decay: Option Value Erosion Near Expiry")
    ax.legend()
    ax.invert_xaxis()
    _watermark(fig)
    return _save(fig, proj, "04_theta_time_decay.png")


# ---------------------------------------------------------------------------
# Figure 05: P&L decomposition via Greeks
# ---------------------------------------------------------------------------
def plot_pnl_decomposition(proj=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    dS_range = np.linspace(-15, 15, 200)
    S0, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20
    g = _bs_greeks(S0, K, T, r, sigma)

    delta_pnl = g["delta"] * dS_range
    gamma_pnl = 0.5 * g["gamma"] * dS_range**2
    theta_pnl = np.full_like(dS_range, g["theta"] / 252)
    total_approx = delta_pnl + gamma_pnl + theta_pnl
    exact_pnl = np.array([_bs_price(S0 + ds, K, T - 1/252, r, sigma)
                          - _bs_price(S0, K, T, r, sigma) for ds in dS_range])

    ax.plot(dS_range, delta_pnl, color=TEAL, lw=2, label="Delta P&L")
    ax.plot(dS_range, gamma_pnl, color=CORAL, lw=2, label="Gamma P&L")
    ax.plot(dS_range, theta_pnl, color=GOLD, lw=2, label="Theta P&L (1d)")
    ax.plot(dS_range, total_approx, color=NAVY, lw=2.5, ls="--", label="Greek Approx")
    ax.plot(dS_range, exact_pnl, color="black", lw=1.5, ls=":", label="Exact Reprice")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Change in Spot ($)")
    ax.set_ylabel("P&L ($)")
    ax.set_title("P&L Decomposition: Delta + Gamma + Theta vs Exact")
    ax.legend(fontsize=9)
    _watermark(fig)
    return _save(fig, proj, "05_pnl_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 06: Implied vs Realized Volatility
# ---------------------------------------------------------------------------
def plot_iv_vs_rv(proj=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    np.random.seed(42)
    days = 252
    t = np.arange(days)
    rv = 0.18 + 0.05 * np.sin(2 * np.pi * t / 252) + 0.02 * np.random.randn(days)
    iv = rv + 0.03 + 0.01 * np.random.randn(days)  # vol risk premium

    ax.plot(t, rv * 100, color=TEAL, lw=1.5, label="Realized Vol (20d)")
    ax.plot(t, iv * 100, color=CORAL, lw=1.5, label="Implied Vol (ATM)")
    ax.fill_between(t, rv * 100, iv * 100, alpha=0.15, color=GOLD,
                    label="Volatility Risk Premium")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Implied vs Realized Volatility & Risk Premium")
    ax.legend()
    _watermark(fig)
    return _save(fig, proj, "06_implied_vs_realized.png")


# ---------------------------------------------------------------------------
# Figure 07: Put-Call Parity verification
# ---------------------------------------------------------------------------
def plot_put_call_parity(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    S_range = np.linspace(70, 130, 200)
    K, T, r, sigma = 100, 0.5, 0.05, 0.20

    calls = np.array([_bs_price(s, K, T, r, sigma, "call") for s in S_range])
    puts  = np.array([_bs_price(s, K, T, r, sigma, "put") for s in S_range])
    lhs = calls - puts
    rhs = S_range - K * np.exp(-r * T)

    ax1.plot(S_range, lhs, color=TEAL, lw=2, label="C - P")
    ax1.plot(S_range, rhs, color=CORAL, lw=2, ls="--", label="S - K*exp(-rT)")
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Value ($)")
    ax1.set_title("Put-Call Parity: C - P = S - PV(K)")
    ax1.legend()

    error = np.abs(lhs - rhs)
    ax2.semilogy(S_range, error, color=NAVY, lw=2)
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Absolute Error ($)")
    ax2.set_title("Parity Violation (Machine Precision)")
    fig.tight_layout()
    _watermark(fig)
    return _save(fig, proj, "07_put_call_parity.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 01 figures...")
    files = [
        plot_volatility_smile(project_dir),
        plot_greeks_surface(project_dir),
        plot_delta_gamma(project_dir),
        plot_theta_decay(project_dir),
        plot_pnl_decomposition(project_dir),
        plot_iv_vs_rv(project_dir),
        plot_put_call_parity(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files


if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))

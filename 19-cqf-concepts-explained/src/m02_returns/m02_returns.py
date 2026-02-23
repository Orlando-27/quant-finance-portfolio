#!/usr/bin/env python3
# =============================================================================
# MODULE 02: RETURN ANALYSIS — LOG VS SIMPLE RETURNS
# Project 19 — CQF Concepts Explained
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Output      : outputs/figures/m02_*.png
# Run         : python3 src/m02_returns/m02_returns.py
# =============================================================================
"""
RETURN ANALYSIS: LOG VS SIMPLE RETURNS
=======================================

THEORETICAL FOUNDATIONS
------------------------

1. SIMPLE (ARITHMETIC) RETURN
   The simplest measure of price change over one period:

       R_t = (P_t - P_{t-1}) / P_{t-1}  =  P_t / P_{t-1}  -  1

   Properties:
   - Interpretation: percentage gain/loss in the period
   - Range:  R_t ∈ (-1, +∞)   [bounded below at -100%]
   - Multi-period:  NOT additive. Must compound:
       R_{0→T} = (1 + R_1)(1 + R_2)...(1 + R_T) - 1
   - Portfolio: IS additive across assets (cross-sectional):
       R_p = Σ_i w_i * R_i

2. LOGARITHMIC (CONTINUOUSLY COMPOUNDED) RETURN
   Defined as the natural log of the gross return:

       r_t = ln(P_t / P_{t-1})  =  ln(1 + R_t)

   Properties:
   - Range: r_t ∈ (-∞, +∞)   [symmetric, can be negative without bound]
   - Multi-period: IS additive over time:
       r_{0→T} = r_1 + r_2 + ... + r_T = ln(P_T / P_0)
   - Portfolio: NOT exactly additive (approximation only)
   - Symmetric: a +10% move and -10% move do NOT cancel in simple returns,
     but roughly cancel in log returns

3. RELATIONSHIP BETWEEN THE TWO
   Via Taylor expansion for small r:

       r_t = ln(1 + R_t) ≈ R_t - R_t²/2 + R_t³/3 - ...
       R_t = e^{r_t} - 1

   For daily returns (|R_t| << 1):
       r_t ≈ R_t     (within 0.5% for |R_t| < 10%)

   The Jensen's inequality adjustment:
       E[e^r] = e^{E[r] + Var(r)/2}    (lognormal property)

   Therefore the expected simple return exceeds the expected log return:
       E[R_t] = e^{E[r_t] + σ²/2} - 1  >  E[r_t]

4. COMPOUNDING AND WEALTH GROWTH
   Starting wealth W_0, after T periods with constant log return μ:

       W_T = W_0 * e^{μ·T}           (log return: exact)
       W_T = W_0 * (1 + μ_s)^T       (simple return: exact)

   The continuously compounded wealth path is:
       W_t = W_0 * exp(Σ_{i=1}^{t} r_i)

   This is exactly the GBM solution: S_t = S_0 * exp(∫dW)

5. WHICH TO USE WHEN?
   Log returns:
   - Time-series modelling (GARCH, VaR, portfolio optimization)
   - When multi-period aggregation is needed
   - When normality assumption is invoked
   - In derivatives pricing (GBM assumption)

   Simple returns:
   - Cross-sectional factor models (Fama-French)
   - Portfolio construction (weights are linear in simple returns)
   - When reporting to stakeholders (more intuitive)
   - Over very long horizons where Jensen's gap is material

6. ANNUALISATION
   Daily log return: r_daily
   Annualised log return:  μ_ann = r_daily * 252      (additive)
   Annualised volatility:  σ_ann = σ_daily * √252     (by √time rule)

   The √252 rule assumes returns are i.i.d. — a useful first approximation
   that breaks down in the presence of serial correlation.

REFERENCES
----------
[1] Campbell, Lo & MacKinlay "The Econometrics of Financial Markets" Ch. 1
[2] Tsay, R.S. "Analysis of Financial Time Series" Ch. 1
[3] Hull, J.C. "Options, Futures, and Other Derivatives" Ch. 15
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.common.style import apply_style, PALETTE, save_fig, annotation_box
except ImportError:
    PALETTE = {
        "blue":"#4F9CF9", "green":"#4CAF82", "orange":"#F5A623",
        "red":"#E05C5C",  "purple":"#9B59B6", "cyan":"#00BCD4",
        "white":"#FFFFFF","grey":"#888888",   "yellow":"#F4D03F"
    }
    def apply_style():
        plt.rcParams.update({
            "figure.facecolor":"#0F1117","axes.facecolor":"#1A1D27",
            "axes.edgecolor":"#3A3D4D","axes.labelcolor":"#E0E0E0",
            "axes.titlecolor":"#FFFFFF","axes.grid":True,
            "axes.titlesize":13,"axes.labelsize":11,
            "grid.color":"#2A2D3A","grid.linewidth":0.6,
            "xtick.color":"#C0C0C0","ytick.color":"#C0C0C0",
            "legend.facecolor":"#1A1D27","legend.edgecolor":"#3A3D4D",
            "legend.labelcolor":"#E0E0E0","legend.fontsize":9,
            "text.color":"#E0E0E0","figure.dpi":130,"savefig.dpi":150,
            "savefig.bbox":"tight","savefig.facecolor":"#0F1117",
            "font.family":"DejaVu Sans","lines.linewidth":1.6,
        })
    def save_fig(fig, name, out_dir="outputs/figures"):
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.png"); plt.close(fig)
        print(f"  Saved: {p}/{name}.png")
    def annotation_box(ax, text, loc="lower right", fontsize=9):
        coords = {
            "lower right":(0.97,0.05,"right","bottom"),
            "lower left" :(0.03,0.05,"left","bottom"),
            "upper right":(0.97,0.95,"right","top"),
            "upper left" :(0.03,0.95,"left","top"),
        }
        x,y,ha,va = coords.get(loc,(0.97,0.05,"right","bottom"))
        ax.text(x,y,text,transform=ax.transAxes,fontsize=fontsize,
                color=PALETTE["cyan"],ha=ha,va=va,
                bbox=dict(boxstyle="round,pad=0.4",fc="#0F1117",
                          ec=PALETTE["cyan"],alpha=0.85))

warnings.filterwarnings("ignore")
apply_style()

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_price_series(mu: float = 0.08, sigma: float = 0.20,
                           S0: float = 100.0, n_days: int = 1260,
                           seed: int = 42) -> pd.Series:
    """
    Simulate a GBM price series.

    Exact discretisation:
        S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t)

    where Z_t ~ N(0,1) i.i.d. and dt = 1/252.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_days)
    prices = S0 * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, S0)
    dates  = pd.bdate_range("2019-01-01", periods=n_days + 1)
    return pd.Series(prices, index=dates, name="Price")


def compute_all_returns(prices: pd.Series) -> pd.DataFrame:
    """
    Compute simple, log and multi-period returns.

    Returns a DataFrame with columns:
        simple   : R_t = P_t/P_{t-1} - 1
        log      : r_t = ln(P_t/P_{t-1})
        diff     : r_t - R_t  (Jensen's correction term ≈ -R_t^2/2)
        log_5d   : 5-day log return  (sum of 5 daily log returns)
        simple_5d: 5-day simple return (compound of 5 daily simple returns)
    """
    df = pd.DataFrame(index=prices.index)
    df["simple"]    = prices.pct_change()
    df["log"]       = np.log(prices / prices.shift(1))
    df["diff"]      = df["log"] - df["simple"]

    # Multi-period returns
    df["log_5d"]    = df["log"].rolling(5).sum()          # additive
    df["simple_5d"] = (1 + df["simple"]).rolling(5).apply(
        np.prod, raw=True) - 1                            # compounded

    return df.dropna()


# =============================================================================
# FIGURE 1: Log vs Simple — Core Comparison
# =============================================================================

def plot_return_comparison(prices: pd.Series, rets: pd.DataFrame) -> None:
    """
    Four-panel figure:
      A. Price series with annotation of GBM parameters
      B. Scatter: log return vs simple return — shows identity line + Jensen gap
      C. Distribution overlay: log vs simple daily returns
      D. Divergence over time: cumulative log vs compounded simple return
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Module 02 — Return Analysis: Log Returns vs Simple Returns",
                 fontsize=15, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ------------------------------------------------------------------
    # Panel A: Price series
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(prices.index, prices.values,
              color=PALETTE["blue"], lw=1.5)
    ax_a.fill_between(prices.index, prices.values, prices.iloc[0],
                      alpha=0.12, color=PALETTE["blue"])
    ax_a.axhline(prices.iloc[0], color=PALETTE["grey"], lw=0.8, ls="--")

    # Annotate total return
    total_log    = np.log(prices.iloc[-1] / prices.iloc[0])
    total_simple = prices.iloc[-1] / prices.iloc[0] - 1
    ax_a.set_title("Simulated GBM Price Series", fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Price (USD)")
    annotation_box(ax_a,
        f"Total log return:    {total_log:.2%}\n"
        f"Total simple return: {total_simple:.2%}\n"
        f"Gap (Jensen):        {total_simple - total_log:.2%}",
        loc="upper left")

    # ------------------------------------------------------------------
    # Panel B: Scatter log vs simple — identity line and curvature
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    r_s  = rets["simple"].values
    r_l  = rets["log"].values

    ax_b.scatter(r_s * 100, r_l * 100,
                 s=4, alpha=0.3, color=PALETTE["blue"], zorder=3)

    # Identity line r_log = r_simple (first-order approximation)
    lim  = max(abs(r_s).max(), abs(r_l).max()) * 100 * 1.1
    xref = np.linspace(-lim, lim, 200)
    ax_b.plot(xref, xref,
              color=PALETTE["orange"], lw=2.0, ls="--",
              label=r"$r = R$ (1st order approx)")

    # Exact relationship: r_log = ln(1 + R_simple)
    ax_b.plot(xref, np.log(1 + xref/100) * 100,
              color=PALETTE["green"], lw=2.0,
              label=r"$r = \ln(1+R)$ (exact)")

    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_title(r"Log Return vs Simple Return: $r_t$ vs $R_t$",
                   fontweight="bold")
    ax_b.set_xlabel("Simple Return $R_t$ (%)")
    ax_b.set_ylabel("Log Return $r_t$ (%)")
    ax_b.legend(loc="upper left")
    annotation_box(ax_b,
        r"$r_t = \ln(1+R_t) \approx R_t - R_t^2/2$" + "\n"
        r"Gap $\approx -R_t^2/2$ (always negative)",
        loc="lower right")

    # ------------------------------------------------------------------
    # Panel C: Distribution comparison
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-0.06, 0.06, 80)

    ax_c.hist(r_s * 100, bins=bins*100, density=True, alpha=0.55,
              color=PALETTE["red"],  label="Simple $R_t$")
    ax_c.hist(r_l * 100, bins=bins*100, density=True, alpha=0.55,
              color=PALETTE["blue"], label="Log $r_t$")

    # Fit Normal to log returns (theoretical)
    mu_fit  = r_l.mean() * 100
    sig_fit = r_l.std()  * 100
    x_n     = np.linspace(-7, 7, 300)
    ax_c.plot(x_n, stats.norm.pdf(x_n, mu_fit, sig_fit),
              color=PALETTE["orange"], lw=2.2,
              label=r"Normal fit to $r_t$")

    # Key statistics
    kurt_s = stats.kurtosis(r_s)
    kurt_l = stats.kurtosis(r_l)
    ax_c.set_title("Distribution: Log vs Simple Returns", fontweight="bold")
    ax_c.set_xlabel("Daily Return (%)")
    ax_c.set_ylabel("Density")
    ax_c.legend()
    annotation_box(ax_c,
        f"Log  — skew={stats.skew(r_l):+.3f}, excess kurt={kurt_l:.3f}\n"
        f"Simple — skew={stats.skew(r_s):+.3f}, excess kurt={kurt_s:.3f}",
        loc="upper right")

    # ------------------------------------------------------------------
    # Panel D: Cumulative return path divergence
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    # Cumulative wealth via each method
    cum_log    = np.exp(rets["log"].cumsum())               # exact
    cum_simple = (1 + rets["simple"]).cumprod()             # exact
    # Naive: sum simple returns (WRONG — shows compounding error)
    cum_naive  = 1 + rets["simple"].cumsum()

    ax_d.plot(cum_log.index,    cum_log.values,
              color=PALETTE["green"],  lw=2.0,
              label="Log: $e^{\\sum r_t}$ (correct)")
    ax_d.plot(cum_simple.index, cum_simple.values,
              color=PALETTE["blue"],   lw=1.8, ls="--",
              label="Simple: $\\prod(1+R_t)$ (correct)")
    ax_d.plot(cum_naive.index,  cum_naive.values,
              color=PALETTE["red"],    lw=1.5, ls=":",
              label="Naive: $1+\\sum R_t$ (WRONG)")

    ax_d.set_title("Cumulative Wealth Paths — Compounding Methods",
                   fontweight="bold")
    ax_d.set_xlabel("Date")
    ax_d.set_ylabel("Wealth Index (base = 1.0)")
    ax_d.legend(fontsize=8)
    annotation_box(ax_d,
        "Log returns are time-additive:\n"
        r"$r_{0\to T} = \sum_{t=1}^T r_t = \ln(P_T/P_0)$",
        loc="upper left")

    save_fig(fig, "m02_return_comparison")


# =============================================================================
# FIGURE 2: Multi-Period Aggregation & Jensen's Inequality
# =============================================================================

def plot_multiperiod_and_jensens(rets: pd.DataFrame) -> None:
    """
    Four-panel figure:
      A. 5-day log vs simple returns: aggregation accuracy
      B. Jensen's inequality: E[R] vs E[r] for different sigma levels
      C. Annualisation: the sqrt(T) rule visualised
      D. Return horizon aggregation error as function of horizon
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Module 02 — Multi-Period Returns, Jensen's Inequality & Annualisation",
        fontsize=14, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ------------------------------------------------------------------
    # Panel A: 5-day log vs 5-day simple
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    r5s = rets["simple_5d"].values * 100
    r5l = rets["log_5d"].values    * 100

    # Remove NaN rows from rolling
    mask = ~(np.isnan(r5s) | np.isnan(r5l))
    r5s, r5l = r5s[mask], r5l[mask]

    ax_a.scatter(r5s, r5l, s=6, alpha=0.35, color=PALETTE["purple"])
    lim5 = max(abs(r5s).max(), abs(r5l).max()) * 1.1
    xref = np.linspace(-lim5, lim5, 200)
    ax_a.plot(xref, xref,              color=PALETTE["orange"],
              lw=1.5, ls="--", label=r"Identity $r=R$")
    ax_a.plot(xref, np.log(1+xref/100)*100, color=PALETTE["green"],
              lw=1.5, label=r"$r=\ln(1+R)$")
    ax_a.set_title("5-Day Aggregated Returns: Log vs Simple",
                   fontweight="bold")
    ax_a.set_xlabel("5-Day Simple Return $R_{t,5}$ (%)")
    ax_a.set_ylabel("5-Day Log Return $r_{t,5}$ (%)")
    ax_a.legend()
    annotation_box(ax_a,
        "Log returns: simply sum 5 daily values.\n"
        r"$r_{t,5} = r_t + r_{t-1} + ... + r_{t-4}$",
        loc="lower right")

    # ------------------------------------------------------------------
    # Panel B: Jensen's Inequality — E[R] vs E[r]
    # ------------------------------------------------------------------
    # For lognormal: E[R] = exp(mu + sigma^2/2) - 1
    #                E[r] = mu  (the log return drift)
    # Gap = Jensen's correction = sigma^2/2
    ax_b = fig.add_subplot(gs[0, 1])
    sigmas = np.linspace(0.01, 0.80, 200)   # annual volatility
    mu_log = 0.08                            # constant log return drift

    e_r_log    = mu_log * np.ones_like(sigmas)
    e_r_simple = np.exp(mu_log + 0.5 * sigmas**2) - 1
    jensens_gap = e_r_simple - e_r_log

    ax_b.plot(sigmas * 100, e_r_log    * 100,
              color=PALETTE["blue"],  lw=2.0, label=r"$E[r] = \mu$ (log return)")
    ax_b.plot(sigmas * 100, e_r_simple * 100,
              color=PALETTE["green"], lw=2.0,
              label=r"$E[R] = e^{\mu + \sigma^2/2} - 1$ (simple)")
    ax_b.fill_between(sigmas * 100, e_r_log * 100, e_r_simple * 100,
                      alpha=0.20, color=PALETTE["orange"],
                      label=r"Jensen gap $= \sigma^2/2$")

    ax_b.set_title(r"Jensen's Inequality: $E[R_t] > E[r_t]$",
                   fontweight="bold")
    ax_b.set_xlabel(r"Annual Volatility $\sigma$ (%)")
    ax_b.set_ylabel("Expected Annual Return (%)")
    ax_b.legend(fontsize=8)
    annotation_box(ax_b,
        r"For lognormal: $E[R] = e^{\mu + \sigma^2/2} - 1$" + "\n"
        r"Jensen gap $\approx \sigma^2/2$ per year",
        loc="upper left")

    # ------------------------------------------------------------------
    # Panel C: The sqrt(T) annualisation rule
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    daily_log = rets["log"].values
    horizons  = [1, 5, 10, 21, 63, 126, 252]
    empirical_vol, theoretical_vol = [], []

    for h in horizons:
        # Empirical: rolling sum of h daily log returns -> std
        roll_sum  = np.array([daily_log[i:i+h].sum()
                               for i in range(len(daily_log) - h)])
        empirical_vol.append(roll_sum.std())
        # Theoretical: sigma_daily * sqrt(h)  (i.i.d. assumption)
        theoretical_vol.append(daily_log.std() * np.sqrt(h))

    ax_c.plot(horizons, [v*100 for v in theoretical_vol],
              color=PALETTE["orange"], lw=2.5, ls="--",
              marker="o", ms=8, label=r"Theory: $\sigma_{daily}\sqrt{h}$")
    ax_c.plot(horizons, [v*100 for v in empirical_vol],
              color=PALETTE["blue"],   lw=2.0,
              marker="s", ms=8, label=r"Empirical: $\sigma$ of $h$-day returns")

    ax_c.set_title(r"Volatility Scaling: The $\sqrt{T}$ Rule",
                   fontweight="bold")
    ax_c.set_xlabel("Horizon $h$ (days)")
    ax_c.set_ylabel("Volatility (%)")
    ax_c.legend()
    annotation_box(ax_c,
        r"Assumes i.i.d.: $\sigma_{h\text{-day}} = \sigma_{1\text{-day}} \times \sqrt{h}$"
        "\nBreaks down with serial correlation.",
        loc="upper left")

    # ------------------------------------------------------------------
    # Panel D: Aggregation error as function of horizon
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    # For h-day horizon: error = h*sigma^2/2  (approximately)
    sigma_d   = daily_log.std()
    h_range   = np.arange(1, 253)
    error_theoretical = h_range * sigma_d**2 / 2 * 100   # in percent

    # Empirical error: |sum(r) - compound(R)| averaged over all windows
    empirical_errors = []
    for h in h_range:
        # Compute h-day log return (sum) and simple return (product)
        log_h    = np.array([daily_log[i:i+h].sum()
                             for i in range(len(daily_log) - h)])
        sim_h    = np.array([(np.prod(1 + rets["simple"].values[i:i+h]) - 1)
                             for i in range(len(daily_log) - h)])
        log_from_simple = np.log(1 + sim_h)
        empirical_errors.append(np.abs(log_h - log_from_simple).mean() * 100)

    ax_d.plot(h_range, error_theoretical,
              color=PALETTE["orange"], lw=2.0, ls="--",
              label=r"Theory: $h \cdot \sigma^2/2$")
    ax_d.plot(h_range, empirical_errors,
              color=PALETTE["blue"], lw=1.8,
              label="Empirical mean abs error")
    ax_d.set_title("Log vs Simple: Aggregation Error by Horizon",
                   fontweight="bold")
    ax_d.set_xlabel("Horizon $h$ (days)")
    ax_d.set_ylabel("Mean Absolute Divergence (%)")
    ax_d.legend()
    annotation_box(ax_d,
        r"Error grows linearly with horizon: $\approx h\sigma^2/2$"
        "\nAt 1-year horizon, gap is material.",
        loc="upper left")

    save_fig(fig, "m02_multiperiod_jensens")


# =============================================================================
# FIGURE 3: Descriptive Statistics Comparison
# =============================================================================

def plot_return_statistics(rets: pd.DataFrame, prices: pd.Series) -> None:
    """
    Three-panel figure:
      A. Rolling annualised statistics (mean and volatility)
      B. QQ-plot comparing log return tails to Normal
      C. Summary statistics table
    """
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Module 02 — Return Statistics & Rolling Risk Metrics",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(1, 3, hspace=0.40, wspace=0.38)

    # ------------------------------------------------------------------
    # Panel A: Rolling annualised mean and volatility (63-day window)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    window = 63
    roll_mu  = rets["log"].rolling(window).mean()  * 252 * 100
    roll_sig = rets["log"].rolling(window).std()   * np.sqrt(252) * 100
    roll_sr  = roll_mu / roll_sig

    ax_a.plot(rets.index, roll_mu,  color=PALETTE["green"],  lw=1.5,
              label=r"Ann. return $\hat{\mu}$ (%)")
    ax_a.plot(rets.index, roll_sig, color=PALETTE["red"],    lw=1.5,
              label=r"Ann. vol $\hat{\sigma}$ (%)")
    ax_a.axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
    ax_a.set_title(f"Rolling {window}-Day Annualised Statistics",
                   fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Annualised Value (%)")
    ax_a.legend()

    # ------------------------------------------------------------------
    # Panel B: QQ-plot of log returns vs Normal
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    r_log     = rets["log"].values
    (osm, osr), (slope, intercept, _) = stats.probplot(r_log, dist="norm")

    ax_b.scatter(osm, osr * 100, s=5, alpha=0.40, color=PALETTE["blue"],
                 label="Observed quantiles")
    ax_b.plot(osm, (slope * np.array(osm) + intercept) * 100,
              color=PALETTE["orange"], lw=2.0,
              label="Normal reference line")
    ax_b.set_title("Q-Q Plot: Log Returns vs Normal Distribution",
                   fontweight="bold")
    ax_b.set_xlabel("Theoretical Normal Quantiles")
    ax_b.set_ylabel("Sample Quantiles (%)")
    ax_b.legend()

    jb_stat, jb_p = stats.jarque_bera(r_log)
    annotation_box(ax_b,
        f"Jarque-Bera: stat={jb_stat:.1f}\n"
        f"p-value: {jb_p:.2e}\n"
        f"Tails deviate from Normal",
        loc="upper left")

    # ------------------------------------------------------------------
    # Panel C: Summary statistics table
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")

    r_s   = rets["simple"].values
    r_l   = rets["log"].values
    n_obs = len(r_l)
    ann   = 252
    sqrt_ann = np.sqrt(ann)

    rows = [
        ["Observations",           f"{n_obs}",               f"{n_obs}"],
        ["Daily Mean",             f"{r_s.mean()*100:.4f}%", f"{r_l.mean()*100:.4f}%"],
        ["Daily Std Dev",          f"{r_s.std()*100:.4f}%",  f"{r_l.std()*100:.4f}%"],
        ["Ann. Return",            f"{r_s.mean()*ann*100:.2f}%",
                                   f"{r_l.mean()*ann*100:.2f}%"],
        ["Ann. Volatility",        f"{r_s.std()*sqrt_ann*100:.2f}%",
                                   f"{r_l.std()*sqrt_ann*100:.2f}%"],
        ["Skewness",               f"{stats.skew(r_s):.4f}",  f"{stats.skew(r_l):.4f}"],
        ["Excess Kurtosis",        f"{stats.kurtosis(r_s):.4f}",
                                   f"{stats.kurtosis(r_l):.4f}"],
        ["Min",                    f"{r_s.min()*100:.2f}%",  f"{r_l.min()*100:.2f}%"],
        ["Max",                    f"{r_s.max()*100:.2f}%",  f"{r_l.max()*100:.2f}%"],
        ["Sharpe (ann)",           f"{r_s.mean()/r_s.std()*sqrt_ann:.3f}",
                                   f"{r_l.mean()/r_l.std()*sqrt_ann:.3f}"],
        ["JB p-value",             "—",                       f"{jb_p:.2e}"],
    ]

    col_labels = ["Statistic", "Simple $R_t$", "Log $r_t$"]
    table = ax_c.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#3A3D4D")
        if r == 0:
            cell.set_facecolor("#2A2D3A")
            cell.set_text_props(color=PALETTE["orange"], fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1E2130")
            cell.set_text_props(color=PALETTE["white"])
        else:
            cell.set_facecolor("#1A1D27")
            cell.set_text_props(color=PALETTE["white"])

    ax_c.set_title("Return Statistics Summary", fontweight="bold",
                   color=PALETTE["white"], pad=12)

    save_fig(fig, "m02_return_statistics")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  MODULE 02 — Return Analysis: Log vs Simple Returns")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate price series
    # ------------------------------------------------------------------
    print("\n[1] Generating GBM price series (mu=8%, sigma=20%)...")
    prices = generate_price_series(mu=0.08, sigma=0.20, S0=100.0,
                                   n_days=1260, seed=42)
    print(f"    Price range: {prices.min():.2f} — {prices.max():.2f}")
    print(f"    Total log return: {np.log(prices.iloc[-1]/prices.iloc[0]):.2%}")

    # ------------------------------------------------------------------
    # 2. Compute returns
    # ------------------------------------------------------------------
    print("\n[2] Computing simple, log and multi-period returns...")
    rets = compute_all_returns(prices)

    # ------------------------------------------------------------------
    # 3. Print key statistics
    # ------------------------------------------------------------------
    print("\n[3] Key statistics (daily):")
    r_l = rets["log"].values
    r_s = rets["simple"].values
    print(f"    Log  — mean: {r_l.mean()*100:.4f}%  std: {r_l.std()*100:.4f}%  "
          f"skew: {stats.skew(r_l):.3f}  kurt: {stats.kurtosis(r_l):.3f}")
    print(f"    Simple — mean: {r_s.mean()*100:.4f}%  std: {r_s.std()*100:.4f}%  "
          f"skew: {stats.skew(r_s):.3f}  kurt: {stats.kurtosis(r_s):.3f}")
    print(f"    Jensen gap (daily mean): {(r_s.mean()-r_l.mean())*100:.5f}%")
    print(f"    Jensen gap theoretical: {r_l.std()**2/2*100:.5f}%  (sigma^2/2)")

    # ------------------------------------------------------------------
    # 4. Verify time-additivity of log returns
    # ------------------------------------------------------------------
    print("\n[4] Verifying time-additivity of log returns:")
    total_log_sum  = rets["log"].sum()
    total_log_path = np.log(prices.iloc[-1] / prices.iloc[0])
    print(f"    Sum of daily log returns:  {total_log_sum:.6f}")
    print(f"    ln(P_T / P_0):             {total_log_path:.6f}")
    print(f"    Error (should be ~0):      {abs(total_log_sum - total_log_path):.2e}")

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    print("\n[5] Generating figures...")
    plot_return_comparison(prices, rets)
    plot_multiperiod_and_jensens(rets)
    plot_return_statistics(rets, prices)

    print("\n[6] Output figures:")
    for f in sorted(OUT.glob("m02_*.png")):
        print(f"    {f.name}")
    print("\n  MODULE 02 COMPLETE")


if __name__ == "__main__":
    main()
"""
Market Microstructure Visualization Engine
============================================
Publication-quality dark-theme charts with professional watermark.

All public methods return (fig, axes) for programmatic use or call
fig.savefig() directly via save_figure().

Design principles:
    - Consistent #0D1117 background (GitHub Dark default)
    - Neon accent palette for maximum contrast on dark background
    - Watermark: semi-transparent, rotated 30°, centred on each figure
    - DPI 150 for screen; 300 for print-ready exports
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style constants
# ---------------------------------------------------------------------------
BG_COLOR     = "#0D1117"
PANEL_COLOR  = "#161B22"
TEXT_COLOR   = "#C9D1D9"
GRID_COLOR   = "#21262D"
ACCENT_COLORS = [
    "#00D4FF", "#FF6B35", "#7FFF7F", "#FFD700", "#DA70D6",
    "#00CED1", "#FF4500", "#32CD32", "#FF8C00", "#9370DB",
]
WATERMARK_TEXT = "Jose Orlando Bobadilla Fuentes, CQF | MSc AI"
DPI            = 150


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    """Apply consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in np.array(ax_list).flatten():
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)


def _add_watermark(fig: plt.Figure) -> None:
    """Place a semi-transparent diagonal watermark on the figure."""
    fig.text(
        0.5, 0.5,
        WATERMARK_TEXT,
        fontsize=9,
        color="white",
        alpha=0.07,
        ha="center",
        va="center",
        rotation=30,
        transform=fig.transFigure,
        zorder=0,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, path: str, dpi: int = DPI) -> None:
    """Save figure with tight layout and dark background."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  [+] Saved: {path}")


# =============================================================================
# 1. Bid-Ask Spread Comparison
# =============================================================================
def plot_spread_comparison(
    spread_df: pd.DataFrame,
    ticker   : str = "Asset",
    save_path: str | None = None,
) -> tuple:
    """
    Multi-panel spread comparison: Roll, Corwin-Schultz, Quoted.

    Parameters
    ----------
    spread_df : pd.DataFrame
        Output of SpreadModels.spread_comparison(). Expected columns:
        roll_spread_pct, corwin_schultz_spread_pct, quoted_spread_pct.
    """
    cols   = spread_df.columns.tolist()
    labels = {
        "roll_spread_pct"           : "Roll (1984) Implicit Spread",
        "corwin_schultz_spread_pct" : "Corwin-Schultz (2012)",
        "quoted_spread_pct"         : "Quoted Spread (OHLC proxy)",
    }

    fig, axes = plt.subplots(len(cols), 1, figsize=(16, 10), sharex=True)
    _apply_dark_style(fig, axes)
    _add_watermark(fig)

    fig.suptitle(
        f"Bid-Ask Spread Models — {ticker}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, col, color in zip(axes, cols, ACCENT_COLORS):
        s = spread_df[col].dropna() * 100  # to bps × 100
        ax.plot(s.index, s.values, color=color, linewidth=0.8, alpha=0.9)
        ax.fill_between(s.index, 0, s.values, color=color, alpha=0.12)
        ax.set_ylabel("Spread (%)", color=TEXT_COLOR, fontsize=9)
        ax.set_title(labels.get(col, col), color=ACCENT_COLORS[0], fontsize=10)

        # Rolling mean overlay
        roll_mean = s.rolling(30).mean()
        ax.plot(roll_mean.index, roll_mean.values,
                color="white", linewidth=1.5, linestyle="--", alpha=0.6,
                label="30d MA")
        ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    axes[-1].set_xlabel("Date", color=TEXT_COLOR, fontsize=9)

    if save_path:
        save_figure(fig, save_path)
    return fig, axes


# =============================================================================
# 2. Order Flow Dashboard
# =============================================================================
def plot_order_flow_dashboard(
    prices  : pd.Series,
    ofi     : pd.Series,
    vpin    : pd.Series,
    ofi_acf : pd.Series,
    ticker  : str = "Asset",
    save_path: str | None = None,
) -> tuple:
    """
    4-panel order flow dashboard:
        (1) Price with OFI color overlay
        (2) Rolling OFI
        (3) VPIN time-series with threshold
        (4) OFI autocorrelation function
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _apply_dark_style(fig, [ax1, ax2, ax3, ax4])
    _add_watermark(fig)

    fig.suptitle(
        f"Order Flow Analysis Dashboard — {ticker}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
    )

    # --- Panel 1: Price + OFI coloring ---
    ofi_common = ofi.reindex(prices.index).dropna()
    p_common   = prices.reindex(ofi_common.index)

    buy_mask  = ofi_common > 0
    sell_mask = ofi_common <= 0

    ax1.plot(p_common.index, p_common.values, color=TEXT_COLOR,
             linewidth=0.6, alpha=0.5)
    ax1.fill_between(p_common.index, p_common.values,
                     where=buy_mask.values, color=ACCENT_COLORS[2],
                     alpha=0.25, label="Buying pressure")
    ax1.fill_between(p_common.index, p_common.values,
                     where=sell_mask.values, color=ACCENT_COLORS[1],
                     alpha=0.25, label="Selling pressure")
    ax1.set_title("Price with Order Flow Imbalance", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("Price ($)", color=TEXT_COLOR, fontsize=9)
    ax1.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # --- Panel 2: Rolling OFI ---
    ax2.plot(ofi.index, ofi.values, color=ACCENT_COLORS[0],
             linewidth=0.7, alpha=0.8)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.fill_between(ofi.index, 0, ofi.values,
                     where=ofi.values > 0, color=ACCENT_COLORS[2], alpha=0.3)
    ax2.fill_between(ofi.index, 0, ofi.values,
                     where=ofi.values <= 0, color=ACCENT_COLORS[1], alpha=0.3)
    ax2.set_title("Order Flow Imbalance (OFI)", color=TEXT_COLOR, fontsize=10)
    ax2.set_ylabel("OFI [-1, +1]", color=TEXT_COLOR, fontsize=9)

    # --- Panel 3: VPIN ---
    vpin_clean = vpin.dropna()
    ax3.plot(vpin_clean.index, vpin_clean.values, color=ACCENT_COLORS[3],
             linewidth=0.8, label="VPIN")
    threshold = 0.5
    ax3.axhline(threshold, color=ACCENT_COLORS[1], linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"Threshold={threshold}")
    ax3.fill_between(vpin_clean.index, threshold, vpin_clean.values,
                     where=vpin_clean.values > threshold,
                     color=ACCENT_COLORS[1], alpha=0.3,
                     label="Elevated toxicity")
    ax3.set_title("VPIN — Flow Toxicity (Easley et al. 2012)",
                  color=TEXT_COLOR, fontsize=10)
    ax3.set_ylabel("VPIN", color=TEXT_COLOR, fontsize=9)
    ax3.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # --- Panel 4: OFI ACF ---
    acf_vals = ofi_acf.values
    lags     = ofi_acf.index.values
    colors_acf = [ACCENT_COLORS[2] if v > 0 else ACCENT_COLORS[1] for v in acf_vals]
    ax4.bar(lags, acf_vals, color=colors_acf, alpha=0.8, width=0.7)
    ax4.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    # 95% confidence bands (±1.96/sqrt(N))
    n    = ofi.dropna().__len__()
    conf = 1.96 / np.sqrt(n)
    ax4.axhline(conf,  color="white", linewidth=0.8, linestyle=":",
                alpha=0.5, label="95% CI")
    ax4.axhline(-conf, color="white", linewidth=0.8, linestyle=":",
                alpha=0.5)
    ax4.set_title("OFI Autocorrelation Function", color=TEXT_COLOR, fontsize=10)
    ax4.set_xlabel("Lag (days)", color=TEXT_COLOR, fontsize=9)
    ax4.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    if save_path:
        save_figure(fig, save_path)
    return fig, [ax1, ax2, ax3, ax4]


# =============================================================================
# 3. Illiquidity Dashboard
# =============================================================================
def plot_illiquidity_dashboard(
    amihud  : pd.Series,
    kyle_lam: pd.Series,
    comp_liq: pd.Series,
    returns : pd.Series,
    ticker  : str = "Asset",
    save_path: str | None = None,
) -> tuple:
    """
    3-panel illiquidity dashboard:
        (1) Amihud ILLIQ ratio (rolling)
        (2) Kyle's lambda (permanent impact)
        (3) Composite liquidity score vs. cumulative return
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    _apply_dark_style(fig, axes)
    _add_watermark(fig)

    fig.suptitle(
        f"Illiquidity Analysis — {ticker}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.99,
    )

    # Panel 1: Amihud ILLIQ
    ax = axes[0]
    a  = amihud.dropna()
    ax.plot(a.index, a.values, color=ACCENT_COLORS[0], linewidth=0.8)
    ax.fill_between(a.index, 0, a.values, color=ACCENT_COLORS[0], alpha=0.15)
    ax.set_title("Amihud (2002) Illiquidity Ratio (×10⁶)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("ILLIQ", color=TEXT_COLOR, fontsize=9)

    # Add stress shading (ILLIQ > 90th percentile)
    threshold_a = a.quantile(0.90)
    ax.axhline(threshold_a, color=ACCENT_COLORS[1], linewidth=1.0,
               linestyle="--", alpha=0.7, label="90th pct")
    ax.fill_between(a.index, threshold_a, a.values,
                    where=a.values > threshold_a,
                    color=ACCENT_COLORS[1], alpha=0.25,
                    label="High illiquidity regime")
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 2: Kyle's lambda
    ax = axes[1]
    k  = kyle_lam.dropna()
    ax.plot(k.index, k.values, color=ACCENT_COLORS[3], linewidth=0.8)
    ax.fill_between(k.index, 0, k.values, color=ACCENT_COLORS[3], alpha=0.15)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.6, alpha=0.4)
    ax.set_title("Kyle's λ — Permanent Price Impact (OLS Estimate)",
                 color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("λ ($/share)", color=TEXT_COLOR, fontsize=9)

    # Panel 3: Composite liquidity vs cumulative return
    ax  = axes[2]
    ax2 = ax.twinx()

    cl  = comp_liq.dropna()
    cum = (1 + returns).cumprod().reindex(cl.index).dropna()

    ax.plot(cl.index, cl.values, color=ACCENT_COLORS[0],
            linewidth=0.8, label="Liquidity Score")
    ax.fill_between(cl.index, 0, cl.values,
                    where=cl.values > 0, color=ACCENT_COLORS[2], alpha=0.2)
    ax.fill_between(cl.index, 0, cl.values,
                    where=cl.values <= 0, color=ACCENT_COLORS[1], alpha=0.2)
    ax.set_ylabel("Liquidity Score (Z)", color=ACCENT_COLORS[0], fontsize=9)
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.4)

    ax2.plot(cum.index, cum.values, color=ACCENT_COLORS[4],
             linewidth=1.2, linestyle="--", alpha=0.8, label="Cum. Return")
    ax2.set_ylabel("Cumulative Return", color=ACCENT_COLORS[4], fontsize=9)
    ax2.tick_params(colors=ACCENT_COLORS[4])
    ax2.set_facecolor(PANEL_COLOR)

    ax.set_title("Composite Liquidity Score vs. Cumulative Return",
                 color=TEXT_COLOR, fontsize=10)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor=PANEL_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    axes[-1].set_xlabel("Date", color=TEXT_COLOR, fontsize=9)

    if save_path:
        save_figure(fig, save_path)
    return fig, axes


# =============================================================================
# 4. Almgren-Chriss Efficient Frontier
# =============================================================================
def plot_almgren_chriss_frontier(
    frontier_df: pd.DataFrame,
    traj_twap  : dict,
    traj_opt   : dict,
    save_path  : str | None = None,
) -> tuple:
    """
    3-panel Almgren-Chriss visualization:
        (1) Efficient frontier (E[Cost] vs Std[Cost])
        (2) Optimal vs TWAP inventory trajectory
        (3) Trade size schedule comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _apply_dark_style(fig, axes)
    _add_watermark(fig)

    fig.suptitle(
        "Almgren-Chriss (2001) Optimal Execution — Efficient Frontier",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )

    # Panel 1: Efficient Frontier
    ax = axes[0]
    sc = ax.scatter(
        frontier_df["std_cost"],
        frontier_df["E_cost"],
        c=np.log10(frontier_df["lambda"] + 1e-12),
        cmap="plasma",
        s=12, alpha=0.8,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₁₀(λ)", color=TEXT_COLOR, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)

    # Mark specific strategies
    twap_pt = frontier_df.iloc[-1]
    opt_pt  = frontier_df.iloc[len(frontier_df) // 2]
    ax.scatter(twap_pt["std_cost"], twap_pt["E_cost"],
               color=ACCENT_COLORS[2], s=80, zorder=5, label="TWAP (λ→0)")
    ax.scatter(opt_pt["std_cost"], opt_pt["E_cost"],
               color=ACCENT_COLORS[1], s=80, zorder=5, marker="*",
               label="Selected optimal")

    ax.set_xlabel("Execution Risk  Std[Cost] ($)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Expected Cost  E[Cost] ($)", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Efficient Frontier", color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 2: Inventory trajectories
    ax = axes[1]
    ax.plot(traj_twap["times"], traj_twap["inventory"] / traj_twap["inventory"][0],
            color=ACCENT_COLORS[2], linewidth=2.0, linestyle="--", label="TWAP")
    ax.plot(traj_opt["times"], traj_opt["inventory"] / traj_opt["inventory"][0],
            color=ACCENT_COLORS[0], linewidth=2.0, label="Optimal (λ selected)")
    ax.set_xlabel("Time (fraction of horizon)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Remaining Inventory (fraction)", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Liquidation Trajectory", color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 3: Trade size schedule
    ax = axes[2]
    t_mid_twap = 0.5 * (traj_twap["times"][:-1] + traj_twap["times"][1:])
    t_mid_opt  = 0.5 * (traj_opt["times"][:-1]  + traj_opt["times"][1:])

    width = 0.02
    ax.bar(t_mid_twap - width/2, traj_twap["trade_size"],
           width=width, color=ACCENT_COLORS[2], alpha=0.75, label="TWAP")
    ax.bar(t_mid_opt + width/2, traj_opt["trade_size"],
           width=width, color=ACCENT_COLORS[0], alpha=0.75, label="Optimal")
    ax.set_xlabel("Time (fraction of horizon)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Shares Traded per Period", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Trade Size Schedule", color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    if save_path:
        save_figure(fig, save_path)
    return fig, axes


# =============================================================================
# 5. Intraday Profile (VWAP / Volume Profile)
# =============================================================================
def plot_intraday_profile(
    bars    : pd.DataFrame,
    ticker  : str = "Synthetic Asset",
    save_path: str | None = None,
) -> tuple:
    """
    4-panel intraday microstructure:
        (1) OHLC price + VWAP bands
        (2) Volume profile (horizontal bar chart)
        (3) Bid-ask spread proxy over the day
        (4) Signed volume (order flow) by bar
    """
    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _apply_dark_style(fig, [ax1, ax2, ax3, ax4])
    _add_watermark(fig)

    fig.suptitle(
        f"Intraday Market Microstructure Profile — {ticker}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
    )

    idx   = bars.index
    opens  = bars["open"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    closes = bars["close"].values
    vols   = bars["volume"].values if "volume" in bars.columns else np.ones(len(bars))
    sv     = bars["signed_volume"].values if "signed_volume" in bars.columns else np.zeros(len(bars))

    vwap = (closes * vols).cumsum() / np.maximum(vols.cumsum(), 1)

    # VWAP bands (VWAP ± 1σ)
    cumulative_ret = pd.Series(closes).pct_change().fillna(0)
    roll_std = cumulative_ret.rolling(20, min_periods=1).std().values * closes
    upper_band = vwap + 1.5 * roll_std
    lower_band = vwap - 1.5 * roll_std

    # --- Panel 1: Price + VWAP ---
    bar_width = 0.6 / max(len(idx), 1)
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = ACCENT_COLORS[2] if c >= o else ACCENT_COLORS[1]
        ax1.plot([i, i], [l, h], color=color, linewidth=0.6, alpha=0.7)
        ax1.plot([i - bar_width/2, i + bar_width/2], [o, o],
                 color=color, linewidth=1.0)
        ax1.plot([i - bar_width/2, i + bar_width/2], [c, c],
                 color=color, linewidth=1.5)

    x_idx = np.arange(len(idx))
    ax1.plot(x_idx, vwap, color=ACCENT_COLORS[0], linewidth=1.5,
             label="VWAP", zorder=5)
    ax1.fill_between(x_idx, lower_band, upper_band,
                     color=ACCENT_COLORS[0], alpha=0.1, label="VWAP ±1.5σ")
    ax1.set_title("OHLC Price + VWAP Bands", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("Price ($)", color=TEXT_COLOR, fontsize=9)
    ax1.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # Sparse x-ticks
    tick_step = max(1, len(idx) // 6)
    xtick_idx = list(range(0, len(idx), tick_step))
    ax1.set_xticks(xtick_idx)
    ax1.set_xticklabels(
        [idx[i].strftime("%H:%M") for i in xtick_idx],
        rotation=30, fontsize=7, color=TEXT_COLOR,
    )

    # --- Panel 2: Volume Profile (horizontal) ---
    price_bins  = np.linspace(lows.min(), highs.max(), 30)
    vol_profile = np.zeros(len(price_bins) - 1)
    for i, (l, h, v) in enumerate(zip(lows, highs, vols)):
        mask = (price_bins[:-1] >= l) & (price_bins[1:] <= h)
        vol_profile[mask] += v / max(mask.sum(), 1)

    bar_centers = 0.5 * (price_bins[:-1] + price_bins[1:])
    ax2.barh(bar_centers, vol_profile,
             height=(price_bins[1] - price_bins[0]) * 0.85,
             color=ACCENT_COLORS[0], alpha=0.75)
    # POC – Point of Control (highest volume price)
    poc_idx = np.argmax(vol_profile)
    ax2.axhline(bar_centers[poc_idx], color=ACCENT_COLORS[3],
                linewidth=1.5, linestyle="--", label=f"POC=${bar_centers[poc_idx]:.2f}")
    ax2.set_title("Volume Profile (TPO)", color=TEXT_COLOR, fontsize=10)
    ax2.set_xlabel("Volume", color=TEXT_COLOR, fontsize=9)
    ax2.set_ylabel("Price ($)", color=TEXT_COLOR, fontsize=9)
    ax2.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # --- Panel 3: Spread proxy over day ---
    spread_proxy = (highs - lows) / ((highs + lows) / 2.0) * 100
    ax3.plot(x_idx, spread_proxy, color=ACCENT_COLORS[3], linewidth=0.8)
    ax3.fill_between(x_idx, 0, spread_proxy, color=ACCENT_COLORS[3], alpha=0.2)
    ax3.plot(x_idx, pd.Series(spread_proxy).rolling(20, min_periods=1).mean().values,
             color="white", linewidth=1.5, linestyle="--", alpha=0.7,
             label="20-bar MA")
    ax3.set_title("Intraday Spread Proxy (High-Low / Mid, %)",
                  color=TEXT_COLOR, fontsize=10)
    ax3.set_ylabel("Spread (%)", color=TEXT_COLOR, fontsize=9)
    ax3.set_xticks(xtick_idx)
    ax3.set_xticklabels(
        [idx[i].strftime("%H:%M") for i in xtick_idx],
        rotation=30, fontsize=7, color=TEXT_COLOR,
    )
    ax3.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # --- Panel 4: Signed Volume (Order Flow) ---
    colors_sv = [ACCENT_COLORS[2] if v >= 0 else ACCENT_COLORS[1] for v in sv]
    ax4.bar(x_idx, sv, color=colors_sv, alpha=0.8, width=0.7)
    ax4.axhline(0, color=TEXT_COLOR, linewidth=0.6, alpha=0.5)
    ax4.plot(x_idx, pd.Series(sv).rolling(20, min_periods=1).mean().values,
             color=ACCENT_COLORS[0], linewidth=1.5, label="20-bar MA")
    ax4.set_title("Signed Volume (Order Flow Imbalance per Bar)",
                  color=TEXT_COLOR, fontsize=10)
    ax4.set_ylabel("Signed Volume", color=TEXT_COLOR, fontsize=9)
    ax4.set_xticks(xtick_idx)
    ax4.set_xticklabels(
        [idx[i].strftime("%H:%M") for i in xtick_idx],
        rotation=30, fontsize=7, color=TEXT_COLOR,
    )
    ax4.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    if save_path:
        save_figure(fig, save_path)
    return fig, [ax1, ax2, ax3, ax4]

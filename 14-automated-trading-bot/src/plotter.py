"""
plotter.py
----------
Publication-quality charts for the trading bot dashboard.

All figures use matplotlib with the Agg backend (headless / Cloud Shell safe).

Charts produced
---------------
1. Multi-panel signal dashboard  (price, RSI, MACD, Bollinger Bands).
2. Portfolio equity curve with drawdown subplot.
3. Position P&L bar chart.
4. Signal heatmap across universe.
5. Full session dashboard (combined figure).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------
STYLE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "text":        "#e6edf3",
    "muted":       "#8b949e",
    "green":       "#3fb950",
    "red":         "#f85149",
    "blue":        "#58a6ff",
    "amber":       "#e3b341",
    "purple":      "#bc8cff",
    "grid":        "#21262d",
    "font_family": "monospace",
}
plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "axes.titlecolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.facecolor":  STYLE["panel"],
    "legend.edgecolor":  STYLE["grid"],
    "legend.labelcolor": STYLE["text"],
    "font.family":       STYLE["font_family"],
    "font.size":         9,
})

usd_fmt = FuncFormatter(lambda x, _: f"${x:,.2f}")
pct_fmt = FuncFormatter(lambda x, _: f"{x:.1f}%")


def _save(fig, path: str, title: str) -> str:
    """Save figure to PNG and close."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    return path


# =============================================================================
# Chart 1: Signal Dashboard (price + indicators)
# =============================================================================
def plot_signal_dashboard(
    symbol:   str,
    bars:     pd.DataFrame,
    rsi:      pd.Series,
    macd_l:   pd.Series,
    macd_h:   pd.Series,
    bb_upper: pd.Series,
    bb_mid:   pd.Series,
    bb_lower: pd.Series,
    ema_s:    pd.Series,
    ema_l:    pd.Series,
    signal:   int,
    output_path: str,
) -> str:
    """
    Four-panel chart: price + Bollinger + EMA, RSI, MACD histogram.

    Parameters
    ----------
    All series are assumed to share the same DatetimeIndex as bars.
    signal : Final consensus signal (-1, 0, +1) for annotation.

    Returns
    -------
    str : Path to saved PNG.
    """
    idx = bars.index
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{symbol} - Signal Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    gs = gridspec.GridSpec(
        4, 1, hspace=0.05,
        height_ratios=[3, 1, 1, 1],
    )

    # ---- Panel 1: Price + Bollinger + EMA --------------------------------
    ax1 = fig.add_subplot(gs[0])
    c   = bars["close"]
    ax1.plot(idx, c,        color=STYLE["blue"],   lw=1.2, label="Close")
    ax1.plot(idx, bb_upper, color=STYLE["muted"],  lw=0.8, ls="--", label="BB Upper")
    ax1.plot(idx, bb_mid,   color=STYLE["amber"],  lw=0.8, ls="--", label="BB Mid")
    ax1.plot(idx, bb_lower, color=STYLE["muted"],  lw=0.8, ls="--", label="BB Lower")
    ax1.plot(idx, ema_s,    color=STYLE["green"],  lw=1.0, label=f"EMA Short")
    ax1.plot(idx, ema_l,    color=STYLE["red"],    lw=1.0, label=f"EMA Long")
    ax1.fill_between(idx, bb_upper, bb_lower, alpha=0.06, color=STYLE["blue"])

    # Signal annotation
    sig_label = {1: "BUY", -1: "SELL", 0: "FLAT"}[signal]
    sig_color = {1: STYLE["green"], -1: STYLE["red"], 0: STYLE["muted"]}[signal]
    ax1.annotate(
        sig_label,
        xy=(idx[-1], float(c.iloc[-1])),
        xytext=(-60, 20), textcoords="offset points",
        fontsize=14, fontweight="bold", color=sig_color,
        arrowprops=dict(arrowstyle="->", color=sig_color, lw=1.5),
    )
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_major_formatter(usd_fmt)

    # ---- Panel 2: Volume -------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    vol_colors = [
        STYLE["green"] if bars["close"].iloc[i] >= bars["open"].iloc[i]
        else STYLE["red"]
        for i in range(len(bars))
    ]
    ax2.bar(idx, bars["volume"], color=vol_colors, alpha=0.7, width=0.0003)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}")
    )

    # ---- Panel 3: RSI ----------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(idx, rsi, color=STYLE["purple"], lw=1.2, label="RSI")
    ax3.axhline(70, color=STYLE["red"],   lw=0.8, ls="--", alpha=0.8)
    ax3.axhline(30, color=STYLE["green"], lw=0.8, ls="--", alpha=0.8)
    ax3.fill_between(idx, rsi, 70, where=(rsi >= 70), alpha=0.2, color=STYLE["red"])
    ax3.fill_between(idx, rsi, 30, where=(rsi <= 30), alpha=0.2, color=STYLE["green"])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.grid(True, alpha=0.4)
    ax3.tick_params(labelbottom=False)

    # ---- Panel 4: MACD histogram -----------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in macd_h]
    ax4.bar(idx, macd_h, color=colors, alpha=0.8, width=0.0003, label="Histogram")
    ax4.plot(idx, macd_l, color=STYLE["blue"], lw=1.0, label="MACD")
    ax4.axhline(0, color=STYLE["muted"], lw=0.8)
    ax4.set_ylabel("MACD")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.4)
    ax4.legend(loc="upper left", fontsize=7)

    # Rotate x-tick labels
    for ax in [ax4]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    return _save(fig, output_path, f"{symbol} Signal Dashboard")


# =============================================================================
# Chart 2: Equity Curve with Drawdown
# =============================================================================
def plot_equity_curve(
    equity_df:   pd.DataFrame,
    initial_eq:  float,
    output_path: str,
) -> str:
    """
    Two-panel chart: equity curve (top) and rolling drawdown (bottom).

    Parameters
    ----------
    equity_df  : DataFrame with 'net_liq' column and DatetimeIndex.
    initial_eq : Starting equity for reference line.

    Returns
    -------
    str : Path to saved PNG.
    """
    if len(equity_df) < 2:
        return ""

    curve = equity_df["net_liq"]
    hwm   = curve.cummax()
    dd    = (curve - hwm) / hwm * 100     # percentage drawdown

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        "Portfolio Equity Curve & Drawdown",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    # Equity
    ax1.plot(curve.index, curve.values, color=STYLE["blue"], lw=1.5, label="Net Liq")
    ax1.axhline(initial_eq, color=STYLE["amber"], lw=0.8, ls="--", label=f"Start {usd_fmt(initial_eq, None)}")
    ax1.fill_between(
        curve.index, curve.values, initial_eq,
        where=(curve.values >= initial_eq), alpha=0.15, color=STYLE["green"],
    )
    ax1.fill_between(
        curve.index, curve.values, initial_eq,
        where=(curve.values < initial_eq),  alpha=0.15, color=STYLE["red"],
    )
    ax1.set_ylabel("Net Liquidation (USD)")
    ax1.yaxis.set_major_formatter(usd_fmt)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.4)

    # Drawdown
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color=STYLE["red"])
    ax2.plot(dd.index, dd.values, color=STYLE["red"], lw=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.4)
    ax2.yaxis.set_major_formatter(pct_fmt)

    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()

    return _save(fig, output_path, "Equity Curve")


# =============================================================================
# Chart 3: Signal Heatmap
# =============================================================================
def plot_signal_heatmap(
    signals_df:  pd.DataFrame,
    output_path: str,
) -> str:
    """
    Heatmap showing individual indicator signals across all symbols.

    Parameters
    ----------
    signals_df : DataFrame with columns:
                 symbol, rsi_signal, macd_signal, bb_signal, ema_signal, score, signal
    Returns
    -------
    str : Path to saved PNG.
    """
    if signals_df.empty:
        return ""

    cols = ["rsi_signal", "macd_signal", "bb_signal", "ema_signal", "score"]
    df   = signals_df.set_index("symbol")[cols].astype(float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5 + 2)))
    fig.suptitle(
        f"Signal Heatmap  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    cmap  = plt.cm.RdYlGn
    im    = ax.imshow(df.values, cmap=cmap, aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["RSI", "MACD", "BB", "EMA", "Score"], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)

    # Annotate cells
    for i in range(len(df)):
        for j, col in enumerate(cols):
            val = df.iloc[i, j]
            ax.text(
                j, i, f"{val:+.0f}",
                ha="center", va="center",
                color="black", fontsize=8, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Signal (-1 = Sell, +1 = Buy)")
    fig.tight_layout()

    return _save(fig, output_path, "Signal Heatmap")


# =============================================================================
# Chart 4: Position P&L Bar Chart
# =============================================================================
def plot_position_pnl(
    positions_df: pd.DataFrame,
    output_path:  str,
) -> str:
    """
    Horizontal bar chart showing unrealised P&L per open position.

    Parameters
    ----------
    positions_df : Output of PositionMonitor.snapshot().
    Returns
    -------
    str : Path to saved PNG.
    """
    if positions_df.empty:
        return ""

    df     = positions_df.sort_values("unrealized_pnl")
    colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in df["unrealized_pnl"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6 + 2)))
    fig.suptitle(
        "Open Position Unrealised P&L",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    bars = ax.barh(df["symbol"], df["unrealized_pnl"], color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color=STYLE["muted"], lw=1.0)

    for bar, val in zip(bars, df["unrealized_pnl"]):
        ax.text(
            bar.get_width() + (0.003 * abs(df["unrealized_pnl"]).max()),
            bar.get_y() + bar.get_height() / 2,
            format(val, "+,.2f"),
            va="center", fontsize=8,
            color=STYLE["green"] if val >= 0 else STYLE["red"],
        )

    ax.set_xlabel("Unrealised P&L (USD)")
    ax.xaxis.set_major_formatter(usd_fmt)
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()

    return _save(fig, output_path, "Position P&L")


# =============================================================================
# Chart 5: Full Session Dashboard
# =============================================================================
def plot_session_dashboard(
    account:      dict,
    signals_df:   pd.DataFrame,
    positions_df: pd.DataFrame,
    orders_df:    pd.DataFrame,
    equity_df:    pd.DataFrame,
    initial_eq:   float,
    output_path:  str,
) -> str:
    """
    Single comprehensive figure: equity curve, signal summary, positions,
    and order log — suitable for daily email attachment.

    Returns
    -------
    str : Path to saved PNG.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Trading Bot Session Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight="bold", color=STYLE["text"], y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- 1. Equity Curve (top-left, wide) --------------------------------
    ax_eq = fig.add_subplot(gs[0, :])
    if len(equity_df) >= 2:
        curve = equity_df["net_liq"]
        ax_eq.plot(curve.index, curve.values, color=STYLE["blue"], lw=1.5, label="Net Liq")
        ax_eq.axhline(initial_eq, color=STYLE["amber"], lw=0.8, ls="--",
                      label=f"Start {initial_eq:,.0f}")
        ax_eq.fill_between(
            curve.index, curve.values, initial_eq,
            where=(curve.values >= initial_eq), alpha=0.15, color=STYLE["green"],
        )
        ax_eq.fill_between(
            curve.index, curve.values, initial_eq,
            where=(curve.values < initial_eq),  alpha=0.15, color=STYLE["red"],
        )
        ax_eq.yaxis.set_major_formatter(usd_fmt)
        ax_eq.legend(fontsize=8)
    ax_eq.set_title("Equity Curve", fontsize=10)
    ax_eq.grid(True, alpha=0.4)

    # ---- 2. Account Summary (middle-left) --------------------------------
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.axis("off")
    ax_acc.set_title("Account Summary", fontsize=10)
    rows = [(k.replace("_", " ").title(), f"${v:,.2f}") for k, v in account.items()]
    tbl  = ax_acc.table(
        cellText    = rows,
        colLabels   = ["Metric", "Value"],
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(STYLE["panel"] if r == 0 else STYLE["bg"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"])

    # ---- 3. Signal Summary Bar (middle-right) ----------------------------
    ax_sig = fig.add_subplot(gs[1, 1])
    if not signals_df.empty:
        sig_counts = signals_df["signal"].value_counts().reindex([-1, 0, 1], fill_value=0)
        labels     = ["SELL", "FLAT", "BUY"]
        colors_s   = [STYLE["red"], STYLE["muted"], STYLE["green"]]
        ax_sig.bar(labels, [sig_counts[-1], sig_counts[0], sig_counts[1]],
                   color=colors_s, alpha=0.85, width=0.5)
        ax_sig.set_ylabel("Count")
        ax_sig.grid(True, axis="y", alpha=0.4)
    ax_sig.set_title("Signal Distribution", fontsize=10)

    # ---- 4. Position P&L (bottom-left) ----------------------------------
    ax_pos = fig.add_subplot(gs[2, 0])
    if not positions_df.empty:
        pnl_vals = positions_df["unrealized_pnl"].values
        syms     = positions_df["symbol"].values
        colors_p = [STYLE["green"] if v >= 0 else STYLE["red"] for v in pnl_vals]
        ax_pos.barh(syms, pnl_vals, color=colors_p, alpha=0.85, height=0.5)
        ax_pos.axvline(0, color=STYLE["muted"], lw=1.0)
        ax_pos.xaxis.set_major_formatter(usd_fmt)
        ax_pos.grid(True, axis="x", alpha=0.4)
    ax_pos.set_title("Position Unrealised P&L", fontsize=10)

    # ---- 5. Order Distribution (bottom-right) ---------------------------
    ax_ord = fig.add_subplot(gs[2, 1])
    if not orders_df.empty and "action" in orders_df.columns:
        counts = orders_df["action"].value_counts()
        colors_o = [STYLE["green"] if a == "BUY" else STYLE["red"] for a in counts.index]
        ax_ord.bar(counts.index, counts.values, color=colors_o, alpha=0.85, width=0.4)
        ax_ord.set_ylabel("Orders")
        ax_ord.grid(True, axis="y", alpha=0.4)
    ax_ord.set_title("Orders by Action", fontsize=10)

    return _save(fig, output_path, "Session Dashboard")

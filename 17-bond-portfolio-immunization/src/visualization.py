"""
visualization.py — Publication-Quality Dark-Theme Visualizations
================================================================
All figures use:
  • Background   : #0a0a0a
  • Text/axes    : #e0e0e0
  • Accent/price : #00d4ff  (cyan)
  • Positive     : #00ff9f  (green)
  • Negative     : #ff4d6d  (red/pink)
  • Grid         : #1e1e2e
  • Watermark    : "Jose O. Bobadilla | CQF" (bottom-right, alpha=0.12)

Charts produced:
  1. yield_price_curve()        — Bond price vs YTM for multiple bonds
  2. duration_convexity_map()   — Scatter: Maturity vs Duration, sized by Convexity
  3. immunization_bar()         — Portfolio vs Liability: PV, Duration, Convexity
  4. parallel_shift_pnl()       — Surplus change across parallel shocks
  5. key_rate_duration_bar()    — KRD contribution by tenor bucket
  6. cash_flow_matching()       — Bond CF stacked vs liability timeline
  7. portfolio_structures()     — PV breakdown: Bullet vs Barbell vs Ladder
  8. scenario_heatmap()         — Surplus change heatmap by scenario/structure

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from typing import List, Optional, Dict
from pathlib import Path

# ── Global dark style ─────────────────────────────────────────────────────────
BG       = "#0a0a0a"
TEXT     = "#e0e0e0"
ACCENT   = "#00d4ff"
POS      = "#00ff9f"
NEG      = "#ff4d6d"
WARN     = "#f4c542"
GRID     = "#1e1e2e"
PANEL    = "#12121f"
WM_TEXT  = "Jose O. Bobadilla | CQF"
PALETTE  = ["#00d4ff", "#00ff9f", "#f4c542", "#ff4d6d", "#a78bfa", "#fb923c"]


def _apply_dark_style(fig, axes):
    """Apply consistent dark theme to figure and all axes."""
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)


def _watermark(fig, text: str = WM_TEXT):
    """Add translucent watermark in the bottom-right corner."""
    fig.text(0.98, 0.01, text, fontsize=7, color=TEXT,
             alpha=0.12, ha="right", va="bottom",
             fontfamily="monospace", style="italic")


def _save(fig, path: Path, fname: str, dpi: int = 150) -> str:
    """Save figure and close to free memory."""
    fpath = str(path / fname)
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    return fpath


# ── 1. Price vs YTM Curve ─────────────────────────────────────────────────────
def yield_price_curve(bonds, ytm_range: np.ndarray,
                       current_ytms: List[float],
                       outdir: Path) -> str:
    """
    Bond price as a function of YTM for each bond in the universe.
    Marks current market price with a scatter dot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    for i, (bond, ytm) in enumerate(zip(bonds, current_ytms)):
        color  = PALETTE[i % len(PALETTE)]
        prices = [bond.price(y) for y in ytm_range]
        label  = f"{bond.issuer or 'Bond'} ({bond.maturity:.0f}Y, C={bond.coupon_rate*100:.1f}%)"
        ax.plot(ytm_range * 100, prices, color=color, lw=1.8, label=label)
        ax.scatter([ytm * 100], [bond.price(ytm)],
                   color=color, s=70, zorder=5, edgecolors=BG, linewidths=1)

    ax.set_xlabel("Yield-to-Maturity (%)", fontsize=10)
    ax.set_ylabel("Clean Price (USD)", fontsize=10)
    ax.set_title("Bond Price vs YTM — Negative Convexity Relationship", fontsize=12, pad=12)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "01_price_yield_curve.png")


# ── 2. Duration–Convexity Scatter ─────────────────────────────────────────────
def duration_convexity_map(bonds, yields: List[float], outdir: Path) -> str:
    """
    Scatter plot: x=Maturity, y=Modified Duration, size & color = Convexity.
    Visualizes the duration–maturity relationship and convexity profile.
    """
    mats   = [b.maturity for b in bonds]
    d_mods = [b.modified_duration(y) for b, y in zip(bonds, yields)]
    convex = [b.convexity(y) for b, y in zip(bonds, yields)]
    labels = [b.issuer or f"T={b.maturity:.0f}Y" for b in bonds]

    fig, ax = plt.subplots(figsize=(9, 6))
    _apply_dark_style(fig, ax)

    sizes  = np.array(convex) * 8 + 40
    sc     = ax.scatter(mats, d_mods, s=sizes, c=convex,
                        cmap="cool", alpha=0.85, edgecolors=TEXT, linewidths=0.5)
    for x, y, label in zip(mats, d_mods, labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color=TEXT)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Convexity", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

    ax.set_xlabel("Maturity (Years)", fontsize=10)
    ax.set_ylabel("Modified Duration (Years)", fontsize=10)
    ax.set_title("Duration–Convexity Profile Across Bond Universe", fontsize=12, pad=12)
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "02_duration_convexity_map.png")


# ── 3. Immunization Bar Comparison ────────────────────────────────────────────
def immunization_bar(immu_result: dict, outdir: Path) -> str:
    """
    3-panel bar chart comparing Portfolio vs Liability:
    PV | Modified Duration | Convexity
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    _apply_dark_style(fig, axes)

    metrics = [
        ("PV (USD)", immu_result["portfolio_pv"], immu_result["liability_pv"]),
        ("Mod. Duration (Yrs)", immu_result["portfolio_duration"], immu_result["liability_duration"]),
        ("Convexity", immu_result["portfolio_convexity"], immu_result["liability_convexity"]),
    ]

    for ax, (title, port_val, liab_val) in zip(axes, metrics):
        bars = ax.bar(["Portfolio", "Liability"], [port_val, liab_val],
                      color=[ACCENT, WARN], alpha=0.85, width=0.4,
                      edgecolor=BG, linewidth=0.8)
        for bar, val in zip(bars, [port_val, liab_val]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:,.2f}", ha="center", va="bottom",
                    fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=9, color=TEXT, pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Portfolio", "Liability"], fontsize=8)

    # ── Immunization status banner ────────────────────────────────────────────
    immunized = immu_result.get("immunized", False)
    status    = "IMMUNIZED" if immunized else "NOT IMMUNIZED"
    color     = POS if immunized else NEG
    fig.suptitle(f"Redington Immunization Analysis — Status: {status}",
                 fontsize=13, color=color, y=1.02, fontweight="bold")

    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "03_immunization_comparison.png")


# ── 4. Parallel Shift P&L ─────────────────────────────────────────────────────
def parallel_shift_pnl(scenario_results: List[dict], outdir: Path) -> str:
    """
    Dual-panel chart:
      Left  — Portfolio PV vs Liability PV across yield shocks (line)
      Right — Surplus change in USD across shocks (bar, colored pos/neg)
    """
    shocks  = [r["shock_bps"] for r in scenario_results]
    port_pv = [r["portfolio_pv"] for r in scenario_results]
    liab_pv = [r["liability_pv"] for r in scenario_results]
    surplus = [r["surplus_change"] for r in scenario_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark_style(fig, [ax1, ax2])

    # Left: PV comparison
    ax1.plot(shocks, port_pv, color=ACCENT, lw=2, marker="o", ms=4, label="Portfolio PV")
    ax1.plot(shocks, liab_pv, color=WARN,  lw=2, marker="s", ms=4, label="Liability PV",
             linestyle="--")
    ax1.set_xlabel("Yield Shock (bps)", fontsize=10)
    ax1.set_ylabel("Present Value (USD)", fontsize=10)
    ax1.set_title("PV Sensitivity to Parallel Rate Shifts", fontsize=11, pad=10)
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Right: Surplus change bar
    colors = [POS if s >= 0 else NEG for s in surplus]
    bars   = ax2.bar(shocks, surplus, color=colors, alpha=0.8,
                     width=30, edgecolor=BG, linewidth=0.5)
    ax2.axhline(0, color=TEXT, lw=0.8, alpha=0.5)
    ax2.set_xlabel("Yield Shock (bps)", fontsize=10)
    ax2.set_ylabel("Surplus Change (USD)", fontsize=10)
    ax2.set_title("Immunization Surplus Under Rate Shocks", fontsize=11, pad=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Parallel Yield Curve Shock Analysis", fontsize=13,
                 color=TEXT, y=1.02, fontweight="bold")
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "04_parallel_shift_pnl.png")


# ── 5. Key Rate Duration Bar ──────────────────────────────────────────────────
def key_rate_duration_bar(krd_results: dict, key_labels: List[str],
                           outdir: Path) -> str:
    """
    Horizontal bar chart of KRD contributions by tenor.
    Also shows KR-DV01 (dollar sensitivity) on a secondary axis.
    """
    krd    = krd_results["krd_vector"]
    krdv01 = krd_results["krdv01"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark_style(fig, [ax1, ax2])

    y_pos = np.arange(len(key_labels))

    # KRD
    colors = [ACCENT if k >= 0 else NEG for k in krd]
    ax1.barh(y_pos, krd, color=colors, alpha=0.85, edgecolor=BG, linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(key_labels, fontsize=9)
    ax1.set_xlabel("Key Rate Duration (Years)", fontsize=10)
    ax1.set_title(f"Key Rate Duration Profile  (ΣD = {krd_results['sum_krd']:.3f} ≈ D_mod = {krd_results.get('d_mod', 0):.3f})",
                  fontsize=10, pad=10)
    ax1.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
    for y, v in zip(y_pos, krd):
        ax1.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=7, color=TEXT)

    # KR-DV01
    colors2 = [POS if k >= 0 else NEG for k in krdv01]
    ax2.barh(y_pos, krdv01, color=colors2, alpha=0.85, edgecolor=BG, linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(key_labels, fontsize=9)
    ax2.set_xlabel("KR-DV01 (USD per 1bps)", fontsize=10)
    ax2.set_title("Key Rate Dollar Duration (KR-DV01)", fontsize=10, pad=10)
    ax2.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

    fig.suptitle("Key Rate Duration Analysis — Portfolio Interest Rate Risk by Tenor",
                 fontsize=12, color=TEXT, y=1.02, fontweight="bold")
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "05_key_rate_duration.png")


# ── 6. Cash Flow Matching Timeline ────────────────────────────────────────────
def cash_flow_matching_chart(bonds, weights: np.ndarray,
                              liabilities, outdir: Path) -> str:
    """
    Stacked bar chart of bond cash flows by period vs liability amounts.
    Illustrates the dedication strategy coverage.
    """
    from src.bond import Bond

    all_times = sorted(set(
        t for b in bonds for t in b.cash_flows()[0]
    ))
    all_times = [round(t, 4) for t in all_times]

    # Build matrix: bond CF at each time
    cf_matrix = np.zeros((len(all_times), len(bonds)))
    for j, (bond, w) in enumerate(zip(bonds, weights)):
        times, cfs = bond.cash_flows()
        for i, t in enumerate(all_times):
            idx = np.where(np.abs(times - t) < 1e-6)[0]
            if len(idx):
                cf_matrix[i, j] = cfs[idx[0]] * w

    # Liability amounts at their times
    liab_dict = {round(L.time, 4): L.amount for L in liabilities}
    liab_bars = [liab_dict.get(t, 0) for t in all_times]

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_style(fig, ax)

    x    = np.arange(len(all_times))
    bott = np.zeros(len(all_times))
    for j, bond in enumerate(bonds):
        label = bond.issuer or f"Bond {j+1}"
        ax.bar(x, cf_matrix[:, j], bottom=bott, color=PALETTE[j % len(PALETTE)],
               alpha=0.75, label=label, edgecolor=BG, linewidth=0.3)
        bott += cf_matrix[:, j]

    # Liability as step overlay
    ax.step(x, liab_bars, where="mid", color=NEG, lw=2,
            linestyle="--", label="Liabilities")
    ax.scatter(x, liab_bars, color=NEG, s=40, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}Y" for t in all_times],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Time (Years)", fontsize=10)
    ax.set_ylabel("Cash Flow (USD)", fontsize=10)
    ax.set_title("Cash Flow Matching — Bond CF Coverage vs Liability Schedule",
                 fontsize=12, pad=12)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "06_cash_flow_matching.png")


# ── 7. Portfolio Structures Comparison ────────────────────────────────────────
def portfolio_structures_chart(structures: Dict[str, dict], outdir: Path) -> str:
    """
    Compare Bullet, Barbell, and Ladder portfolios across:
    PV | Modified Duration | Convexity | DV01
    """
    names   = list(structures.keys())
    pvs     = [structures[n]["pv"]       for n in names]
    durs    = [structures[n]["duration"] for n in names]
    convs   = [structures[n]["convexity"] for n in names]
    dv01s   = [structures[n]["dv01"]     for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _apply_dark_style(fig, axes.ravel())

    metrics = [
        (axes[0, 0], pvs,   "Portfolio PV (USD)",        "PV",       True),
        (axes[0, 1], durs,  "Modified Duration (Years)", "Duration", False),
        (axes[1, 0], convs, "Convexity",                 "Convexity",False),
        (axes[1, 1], dv01s, "Portfolio DV01 (USD/bps)",  "DV01",     True),
    ]

    for ax, vals, ylabel, label, is_usd in metrics:
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars   = ax.bar(names, vals, color=colors, alpha=0.85,
                        edgecolor=BG, linewidth=0.5, width=0.4)
        for bar, v in zip(bars, vals):
            fmt = f"${v:,.2f}" if is_usd else f"{v:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, fmt,
                    ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9, pad=8)
        if is_usd:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Portfolio Structure Comparison: Bullet vs Barbell vs Ladder",
                 fontsize=13, color=TEXT, fontweight="bold")
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "07_portfolio_structures.png")


# ── 8. Scenario Heatmap ───────────────────────────────────────────────────────
def scenario_heatmap(heatmap_data: np.ndarray,
                      row_labels: List[str],
                      col_labels: List[str],
                      title: str,
                      outdir: Path) -> str:
    """
    Heatmap of surplus changes (rows=portfolio structures, cols=scenarios).
    Green = positive surplus, Red = negative surplus.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    # Custom diverging colormap centered at 0
    from matplotlib.colors import TwoSlopeNorm
    vmax = np.max(np.abs(heatmap_data))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(heatmap_data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"${heatmap_data[i, j]:+,.0f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(heatmap_data[i, j]) < vmax * 0.5 else "white",
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Surplus Change (USD)", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

    ax.set_title(title, fontsize=12, color=TEXT, pad=12)
    _watermark(fig)
    fig.tight_layout()
    return _save(fig, outdir, "08_scenario_heatmap.png")

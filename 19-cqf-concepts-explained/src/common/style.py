"""
common/style.py
===============
Shared matplotlib dark-theme configuration for all 55 CQF modules.

Usage (every script):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.common.style import apply_style, PALETTE, save_fig
    apply_style()
"""

import matplotlib
matplotlib.use("Agg")           # headless / Cloud Shell safe
import matplotlib.pyplot as plt
from pathlib import Path

PALETTE = {
    "blue"   : "#4F9CF9",
    "green"  : "#4CAF82",
    "orange" : "#F5A623",
    "red"    : "#E05C5C",
    "purple" : "#9B59B6",
    "cyan"   : "#00BCD4",
    "yellow" : "#F4D03F",
    "white"  : "#FFFFFF",
    "grey"   : "#888888",
}

RC = {
    "figure.facecolor"  : "#0F1117",
    "axes.facecolor"    : "#1A1D27",
    "axes.edgecolor"    : "#3A3D4D",
    "axes.labelcolor"   : "#E0E0E0",
    "axes.titlecolor"   : "#FFFFFF",
    "axes.grid"         : True,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "grid.color"        : "#2A2D3A",
    "grid.linewidth"    : 0.6,
    "xtick.color"       : "#C0C0C0",
    "ytick.color"       : "#C0C0C0",
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.facecolor"  : "#1A1D27",
    "legend.edgecolor"  : "#3A3D4D",
    "legend.labelcolor" : "#E0E0E0",
    "legend.fontsize"   : 9,
    "text.color"        : "#E0E0E0",
    "figure.dpi"        : 130,
    "savefig.dpi"       : 150,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : "#0F1117",
    "font.family"       : "DejaVu Sans",
    "lines.linewidth"   : 1.6,
}

def apply_style() -> None:
    """Apply global dark academic style to all plots."""
    plt.rcParams.update(RC)

def save_fig(fig: plt.Figure, name: str,
             out_dir: str = "outputs/figures") -> None:
    """Save figure to output directory and close cleanly."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.png")
    plt.close(fig)
    print(f"  Saved: {p / name}.png")

def annotation_box(ax, text: str, loc: str = "lower right",
                   fontsize: int = 9) -> None:
    """Place a styled text box with formula/annotation on an axes."""
    coords = {
        "lower right" : (0.97, 0.05, "right", "bottom"),
        "lower left"  : (0.03, 0.05, "left",  "bottom"),
        "upper right" : (0.97, 0.95, "right", "top"),
        "upper left"  : (0.03, 0.95, "left",  "top"),
    }
    x, y, ha, va = coords.get(loc, (0.97, 0.05, "right", "bottom"))
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            color=PALETTE["cyan"], ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.4", fc="#0F1117",
                      ec=PALETTE["cyan"], alpha=0.85))

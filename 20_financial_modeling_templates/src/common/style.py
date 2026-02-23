#!/usr/bin/env python3
"""
=============================================================================
SHARED VISUALIZATION STYLE & UTILITIES
=============================================================================
Centralised dark-theme configuration and helper functions used across all
six financial modeling modules.  Every figure produced by this project
shares a consistent professional aesthetic.

Design specifications:
    Background   : #0a0a0a (near-black)
    Axes face    : #111111
    Grid         : #1a1a1a, alpha 0.3
    Primary text : #e0e0e0
    Accent colors: institutional palette (blues, greens, ambers)
    Watermark    : "Jose O. Bobadilla | CQF" -- bottom-right, subtle
    DPI          : 150 (publication quality)
    Backend      : Agg (headless / Cloud Shell compatible)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COLOR PALETTE
# =============================================================================
# Institutional palette designed for dark backgrounds
COLORS = {
    "primary"    : "#4fc3f7",   # light blue
    "secondary"  : "#81c784",   # soft green
    "accent"     : "#ffb74d",   # amber
    "danger"     : "#e57373",   # soft red
    "purple"     : "#ba68c8",   # lavender
    "teal"       : "#4db6ac",   # teal
    "pink"       : "#f06292",   # pink
    "white"      : "#e0e0e0",   # text white
    "grid"       : "#1a1a1a",   # grid lines
    "bg"         : "#0a0a0a",   # figure background
    "axes_bg"    : "#111111",   # axes background
}

# Sequential palette for bar charts, waterfall, etc.
PALETTE = [
    "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
    "#ba68c8", "#4db6ac", "#f06292", "#aed581",
    "#64b5f6", "#fff176",
]

# =============================================================================
# GLOBAL MATPLOTLIB CONFIGURATION
# =============================================================================
def apply_dark_theme():
    """Apply dark-theme rcParams globally."""
    plt.rcParams.update({
        "figure.facecolor"    : COLORS["bg"],
        "axes.facecolor"      : COLORS["axes_bg"],
        "axes.edgecolor"      : "#333333",
        "axes.labelcolor"     : COLORS["white"],
        "axes.titlesize"      : 14,
        "axes.titleweight"    : "bold",
        "axes.labelsize"      : 11,
        "axes.grid"           : True,
        "grid.color"          : COLORS["grid"],
        "grid.alpha"          : 0.3,
        "grid.linewidth"      : 0.5,
        "xtick.color"         : COLORS["white"],
        "ytick.color"         : COLORS["white"],
        "xtick.labelsize"     : 9,
        "ytick.labelsize"     : 9,
        "text.color"          : COLORS["white"],
        "legend.facecolor"    : "#1a1a1a",
        "legend.edgecolor"    : "#333333",
        "legend.fontsize"     : 9,
        "legend.framealpha"   : 0.8,
        "figure.dpi"          : 150,
        "savefig.dpi"         : 150,
        "savefig.facecolor"   : COLORS["bg"],
        "savefig.bbox"        : "tight",
        "savefig.pad_inches"  : 0.3,
        "font.family"         : "sans-serif",
        "font.size"           : 10,
    })

# Apply on import
apply_dark_theme()


# =============================================================================
# WATERMARK
# =============================================================================
def add_watermark(fig, text="Jose O. Bobadilla | CQF"):
    """Add subtle watermark to bottom-right of figure."""
    fig.text(
        0.98, 0.02, text,
        fontsize=7, color="#444444", alpha=0.6,
        ha="right", va="bottom",
        fontstyle="italic",
        transform=fig.transFigure,
    )


# =============================================================================
# SAVE FIGURE HELPER
# =============================================================================
def save_figure(fig, filename, subdir=None):
    """
    Save figure to outputs/figures/ with consistent formatting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        Filename without extension.
    subdir : str, optional
        Subdirectory within outputs/figures/.
    """
    target = FIGURES_DIR
    if subdir:
        target = target / subdir
        target.mkdir(parents=True, exist_ok=True)
    filepath = target / f"{filename}.png"
    add_watermark(fig)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    [SAVED] {filepath}")


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================
def fmt_millions(x, pos=None):
    """Format axis tick as $XM."""
    return f"${x/1e6:.0f}M"

def fmt_billions(x, pos=None):
    """Format axis tick as $XB."""
    return f"${x/1e9:.1f}B"

def fmt_pct(x, pos=None):
    """Format axis tick as X.X%."""
    return f"{x:.1f}%"

def fmt_multiple(x, pos=None):
    """Format axis tick as X.Xx."""
    return f"{x:.1f}x"

def fmt_currency(value, decimals=0):
    """Format value as $X,XXX."""
    if abs(value) >= 1e9:
        return f"${value/1e9:,.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:,.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:,.{decimals}f}K"
    else:
        return f"${value:,.{decimals}f}"


# =============================================================================
# TABLE RENDERING HELPER
# =============================================================================
def print_table(title, headers, rows, col_widths=None):
    """
    Print a formatted console table for Cloud Shell output.

    Parameters
    ----------
    title : str
    headers : list of str
    rows : list of list
    col_widths : list of int, optional
    """
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for r in rows:
                if i < len(r):
                    max_w = max(max_w, len(str(r[i])))
            col_widths.append(max_w + 2)

    total_w = sum(col_widths) + len(col_widths) + 1
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    print(f"\n{'=' * total_w}")
    print(f"  {title}")
    print(f"{'=' * total_w}")
    print(sep)
    header_str = "|"
    for i, h in enumerate(headers):
        header_str += str(h).center(col_widths[i]) + "|"
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "|"
        for i, val in enumerate(row):
            if i < len(col_widths):
                row_str += str(val).rjust(col_widths[i] - 1) + " |"
        print(row_str)
    print(sep)


# =============================================================================
# SECTION BANNER
# =============================================================================
def print_section(title, width=70):
    """Print a formatted section header for console output."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

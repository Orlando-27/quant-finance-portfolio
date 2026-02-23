#!/usr/bin/env python3
# =============================================================================
# MODULE 01: FINANCIAL DATA INGESTION
# Project 19 — CQF Concepts Explained
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Output      : outputs/figures/m01_*.png
# Run         : python src/m01_market_data/m01_market_data.py
# =============================================================================
"""
FINANCIAL DATA INGESTION
========================

THEORETICAL FOUNDATIONS
------------------------
Every quantitative model in finance depends critically on clean, well-structured
price data. Understanding how raw exchange prices are recorded and adjusted is
fundamental before any statistical or derivative analysis.

1. OHLCV DATA
   Financial exchanges record price activity in discrete intervals:
     - Open  : First traded price in the interval
     - High  : Maximum traded price in the interval
     - Low   : Minimum traded price in the interval
     - Close : Last traded price (the standard reference for returns)
     - Volume: Total number of shares/contracts traded

   The 'Close' is the most commonly used price in quantitative finance.
   However, raw close prices are NOT comparable across time when corporate
   actions occur.

2. CORPORATE ACTIONS: WHY RAW PRICES ARE MISLEADING
   
   a) STOCK SPLITS
      A company performing a 2-for-1 split doubles the number of shares
      and halves the price. Example:
        - Pre-split:  S_{t-1} = $200, shares = 1,000
        - Post-split: S_t     = $100, shares = 2,000
      
      If we use raw prices, the return r_t = ln(S_t / S_{t-1}) = ln(0.5)
      This would show a -50% return on split day, which is economically
      meaningless — the investor's wealth did not change.
      
      Adjustment: multiply all historical prices before the split date by
      the adjustment factor (0.5 in this case). The adjusted series is
      backward-compatible and stationary in returns.

   b) DIVIDENDS
      When a company pays a dividend D_t on ex-dividend date t,
      the stock price typically drops by approximately D_t:
        S_t^{raw} ≈ S_{t-1}^{raw} - D_t
      
      Raw return: r_t = ln(S_t^{raw} / S_{t-1}^{raw}) < 0 (artificial drop)
      
      Total return: r_t^{total} = ln((S_t^{raw} + D_t) / S_{t-1}^{raw})
      
      The adjusted close price 'rolls up' dividends so that:
        r_t^{adjusted} ≈ r_t^{total}
      
      This is critical for performance measurement: a high-dividend stock
      using raw prices will appear to underperform relative to a zero-
      dividend stock, even if total returns are identical.

3. TOTAL RETURN INDEX
   The adjusted close price series forms the basis of a Total Return Index:
     TRI_t = TRI_{t-1} * (1 + r_t^{adjusted})
   
   This is the correct input for:
     - Return computation
     - Volatility estimation
     - Sharpe ratio calculation
     - Portfolio backtests

4. DATA QUALITY CHECKS
   Before using any price series:
     - Check for missing values (holidays, halted trading, data vendor gaps)
     - Check for zero or negative prices (data errors)
     - Check for extreme single-day moves (check if corporate action or error)
     - Verify price continuity around split/dividend dates

REFERENCES
----------
[1] Hull, J.C. "Options, Futures, and Other Derivatives", 11th ed., Ch. 1
[2] Campbell, Lo & MacKinlay "The Econometrics of Financial Markets", Ch. 1
[3] yfinance documentation: https://github.com/ranaroussi/yfinance
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Path setup — allows running from project root or module directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.common.style import apply_style, PALETTE, save_fig, annotation_box
except ImportError:
    # Fallback if running standalone
    PALETTE = {"blue":"#4F9CF9","green":"#4CAF82","orange":"#F5A623",
                "red":"#E05C5C","purple":"#9B59B6","cyan":"#00BCD4",
                "white":"#FFFFFF","grey":"#888888","yellow":"#F4D03F"}
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
        coords = {"lower right":(0.97,0.05,"right","bottom"),
                  "lower left":(0.03,0.05,"left","bottom"),
                  "upper right":(0.97,0.95,"right","top"),
                  "upper left":(0.03,0.95,"left","top")}
        x,y,ha,va = coords.get(loc,(0.97,0.05,"right","bottom"))
        ax.text(x,y,text,transform=ax.transAxes,fontsize=fontsize,
                color=PALETTE["cyan"],ha=ha,va=va,
                bbox=dict(boxstyle="round,pad=0.4",fc="#0F1117",
                          ec=PALETTE["cyan"],alpha=0.85))

warnings.filterwarnings("ignore")
apply_style()

# ---------------------------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
# We use yfinance to download data. If network is unavailable in Cloud Shell,
# we generate synthetic data that mimics real OHLCV structure.
# =============================================================================

def load_ohlcv(ticker: str = "AAPL",
               start: str = "2018-01-01",
               end:   str = "2023-12-31") -> pd.DataFrame:
    """
    Download OHLCV data via yfinance.
    Returns both raw (unadjusted) and adjusted DataFrames.
    Falls back to synthetic GBM-based data if network unavailable.
    """
    try:
        import yfinance as yf
        raw  = yf.download(ticker, start=start, end=end,
                           auto_adjust=False, progress=False)
        adj  = yf.download(ticker, start=start, end=end,
                           auto_adjust=True,  progress=False)
        if len(raw) < 100:
            raise ValueError("Insufficient data received")
        print(f"  Downloaded {len(raw)} rows for {ticker} from yfinance")
        return raw, adj

    except Exception as e:
        print(f"  yfinance unavailable ({e}). Using synthetic data.")
        return _generate_synthetic_ohlcv(start, end)


def _generate_synthetic_ohlcv(start: str, end: str):
    """
    Generate synthetic OHLCV data using Geometric Brownian Motion.
    
    The GBM model for daily prices:
        S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t)
    
    OHLC bars are constructed by simulating intraday paths:
        - Open:  closing price of previous day (plus small gap)
        - High:  maximum of intraday path
        - Low:   minimum of intraday path
        - Close: end of intraday path
    
    We also simulate two stock split events (3-for-2 and 2-for-1)
    and quarterly dividends to demonstrate price adjustment.
    """
    rng     = np.random.default_rng(42)
    dates   = pd.bdate_range(start, end)
    n       = len(dates)
    dt      = 1 / 252

    # --- Daily close prices (GBM exact solution) ---
    mu, sigma, S0 = 0.10, 0.22, 150.0
    log_ret = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.normal(size=n)
    closes  = S0 * np.exp(np.cumsum(log_ret))

    # --- Intraday OHLC construction ---
    # Each day: 78 five-minute bars; simulate intraday GBM
    sigma_intra = sigma * np.sqrt(1/78)
    opens, highs, lows = [], [], []
    for i in range(n):
        intra = closes[i-1] if i > 0 else S0
        intra_path = intra * np.exp(
            np.cumsum((mu-0.5*sigma**2)*(dt/78)
                      + sigma_intra * rng.normal(size=78))
        )
        opens.append(intra_path[0])
        highs.append(intra_path.max())
        lows.append(intra_path.min())

    # --- Volume (mean-reverting log-normal) ---
    log_vol = rng.normal(np.log(5e6), 0.4, size=n)
    volumes = np.exp(log_vol).astype(int)

    raw_df = pd.DataFrame({
        "Open":closes*0.998, "High":highs, "Low":lows,
        "Close":closes, "Close":closes.copy(), "Volume":volumes
    }, index=dates)

    # --- Simulate a 2-for-1 split at midpoint ---
    split_date = dates[n // 2]
    split_idx  = raw_df.index.get_loc(split_date)
    
    # Raw prices: halve on split date onwards (visually shows the drop)
    raw_df.iloc[split_idx:, :5] *= 0.5   # Open/High/Low/Close/AdjClose

    # Adjusted prices: adjust ALL historical prices by split factor
    adj_df = raw_df.copy()
    adj_close = closes.copy()
    adj_close[:split_idx] *= 0.5    # pre-split history scaled down
    adj_df["Close"] = adj_close

    # --- Simulate quarterly dividends ($0.25/share) ---
    div_mask = pd.date_range(start, end, freq="QS")
    for ddate in div_mask:
        if ddate in adj_df.index:
            # Adjust prices before dividend date by dividend/price factor
            div   = 0.25
            price = adj_df.loc[ddate, "Close"]
            factor = 1 - div / price
            adj_df.loc[:ddate, "Close"] *= factor

    raw_df.columns.name  = None
    adj_df.columns.name  = None
    return raw_df, adj_df


def compute_returns(price_series: pd.Series) -> pd.DataFrame:
    """
    Compute simple and logarithmic daily returns.
    
    Simple return:       r_t = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1
    Logarithmic return:  r_t = ln(P_t / P_{t-1})
    
    Relationship:  r_simple ≈ r_log  for small returns
                   r_log    = ln(1 + r_simple)
    
    Log returns are preferred because:
      1. They are time-additive: sum of daily log returns = total log return
      2. They are approximately normally distributed
      3. They avoid negative prices in models
    """
    ret = pd.DataFrame(index=price_series.index)
    ret["simple"] = price_series.pct_change()
    ret["log"]    = np.log(price_series / price_series.shift(1))
    ret.dropna(inplace=True)
    return ret


def quality_report(df: pd.DataFrame) -> dict:
    """
    Run standard data quality checks on OHLCV data.
    Returns a dict with check results.
    """
    checks = {}
    
    # 1. Missing values
    checks["missing_rows"]   = df.isnull().any(axis=1).sum()
    checks["missing_pct"]    = 100 * checks["missing_rows"] / len(df)
    
    # 2. OHLC consistency: High >= Open, Close, Low and Low <= Open, Close
    _h = df["High"].squeeze();  _l = df["Low"].squeeze()
    _o = df["Open"].squeeze();  _c = df["Close"].squeeze()
    ohlc_ok = ((_h >= _o) & (_h >= _c) & (_l <= _o) & (_l <= _c)).all()
    checks["ohlc_consistent"] = bool(ohlc_ok)
    
    # 3. Zero / negative prices
    checks["zero_prices"] = (df["Close"] <= 0).sum()
    
    # 4. Extreme returns (|r| > 20% in a day — likely error or corporate action)
    _close    = df["Close"].squeeze()
    daily_ret = np.abs(np.log(_close / _close.shift(1)).dropna())
    checks["extreme_moves"]  = (daily_ret > 0.20).sum()
    checks["extreme_dates"]  = daily_ret.index[(daily_ret > 0.20)].tolist()
    
    # 5. Date range
    checks["start"]  = df.index[0].date()
    checks["end"]    = df.index[-1].date()
    checks["n_rows"] = len(df)
    checks["n_years"]= (df.index[-1] - df.index[0]).days / 365.25
    
    return checks


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_ohlcv_overview(raw: pd.DataFrame, adj: pd.DataFrame) -> None:
    # Normalise to 1-D Series (yfinance >= 0.2 returns MultiIndex DataFrames)
    raw = raw.copy(); adj = adj.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in raw.columns: raw[col] = raw[col].squeeze()
        if col in adj.columns: adj[col] = adj[col].squeeze()
    """
    Figure 1: Four-panel OHLCV overview
      A. Raw vs Adjusted Close — illustrates corporate action impact
      B. OHLC candlestick (subset) — shows price structure
      C. Volume bar chart — shows liquidity patterns
      D. Raw vs Adjusted return comparison — demonstrates adjustment effect
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Module 01 — Financial Data Ingestion: OHLCV & Corporate Actions",
                 fontsize=15, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    # ----------------------------------------------------------------
    # Panel A: Raw vs Adjusted Close Price
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(raw.index, raw["Close"],         color=PALETTE["red"],
              lw=1.2, alpha=0.85, label="Raw Close (unadjusted)")
    ax_a.plot(adj.index, adj["Close"],     color=PALETTE["green"],
              lw=1.5, label="Adjusted Close (total return)")
    ax_a.set_title("Raw vs Adjusted Close Price", fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Price (USD)")
    ax_a.legend()
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    annotation_box(ax_a,
        "Adjusted price accounts for\nsplits and dividends.\n"
        r"$P_t^{adj} = P_t^{raw} \times \prod$ (adj. factors)",
        loc="lower right")

    # ----------------------------------------------------------------
    # Panel B: OHLC Candlestick (last 60 trading days)
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    subset = raw.iloc[-60:].copy()
    x_idx  = np.arange(len(subset))
    
    # Candle colors: green if Close >= Open, red otherwise
    colors = [PALETTE["green"] if c >= o else PALETTE["red"]
              for o, c in zip(subset["Open"], subset["Close"])]
    
    # High-Low wick
    ax_b.vlines(x_idx,
                ymin=subset["Low"].values,
                ymax=subset["High"].values,
                colors=colors, lw=0.8, alpha=0.7)
    # Open-Close body
    body_bottom = np.minimum(subset["Open"].values, subset["Close"].values)
    body_height = np.abs(subset["Close"].values - subset["Open"].values)
    for xi, bb, bh, col in zip(x_idx, body_bottom, body_height, colors):
        ax_b.bar(xi, bh, bottom=bb, color=col, alpha=0.9, width=0.6)
    
    ax_b.set_title("OHLC Candlestick Chart (Last 60 Days)", fontweight="bold")
    ax_b.set_xlabel("Trading Day")
    ax_b.set_ylabel("Price (USD)")
    ax_b.set_xticks(x_idx[::10])
    ax_b.set_xticklabels(
        [d.strftime("%b-%d") for d in subset.index[::10]],
        rotation=30, fontsize=8)
    
    bull_patch = plt.matplotlib.patches.Patch(color=PALETTE["green"],
                                               label="Bullish (Close >= Open)")
    bear_patch = plt.matplotlib.patches.Patch(color=PALETTE["red"],
                                               label="Bearish (Close < Open)")
    ax_b.legend(handles=[bull_patch, bear_patch])

    # ----------------------------------------------------------------
    # Panel C: Volume Chart with 20-day MA
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    vol  = raw["Volume"].squeeze() / 1e6    # convert to millions
    vol_ma = vol.rolling(20).mean()
    _o = raw["Open"].squeeze(); _c = raw["Close"].squeeze()
    bar_colors = [PALETTE["blue"] if c >= o else PALETTE["red"]
                  for o, c in zip(_o, _c)]
    ax_c.bar(raw.index, vol.values, color=bar_colors, alpha=0.55, width=1.0)
    ax_c.plot(raw.index, vol_ma, color=PALETTE["orange"],
              lw=1.8, label="20-day MA Volume")
    ax_c.set_title("Daily Trading Volume", fontweight="bold")
    ax_c.set_xlabel("Date")
    ax_c.set_ylabel("Volume (Millions)")
    ax_c.legend()
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    annotation_box(ax_c,
        "Volume = liquidity proxy.\nSpikes often coincide\nwith corporate events.",
        loc="upper right")

    # ----------------------------------------------------------------
    # Panel D: Raw vs Adjusted Daily Return Distributions
    # ----------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    raw_ret = np.log(raw["Close"].squeeze() / raw["Close"].squeeze().shift(1)).dropna()
    adj_ret = np.log(adj["Close"].squeeze() / adj["Close"].squeeze().shift(1)).dropna()
    
    bins = np.linspace(-0.08, 0.08, 80)
    ax_d.hist(raw_ret, bins=bins, alpha=0.55, color=PALETTE["red"],
              density=True, label="Raw returns")
    ax_d.hist(adj_ret, bins=bins, alpha=0.55, color=PALETTE["green"],
              density=True, label="Adjusted returns")
    
    # Overlay Normal distribution for reference
    x_norm = np.linspace(-0.10, 0.10, 300)
    mu_a, sig_a = adj_ret.mean(), adj_ret.std()
    from scipy.stats import norm
    ax_d.plot(x_norm, norm.pdf(x_norm, mu_a, sig_a),
              color=PALETTE["orange"], lw=2.0,
              label=r"Normal $(\hat{\mu}, \hat{\sigma})$")
    
    ax_d.set_title("Daily Log-Return Distribution: Raw vs Adjusted",
                   fontweight="bold")
    ax_d.set_xlabel("Daily Log Return")
    ax_d.set_ylabel("Density")
    ax_d.legend()
    annotation_box(ax_d,
        f"Adj: mean={mu_a:.4f},  sigma={sig_a:.4f}\n"
        f"Ann. return ={mu_a*252:.2%},  Ann. vol ={sig_a*np.sqrt(252):.2%}",
        loc="upper right")

    save_fig(fig, "m01_ohlcv_overview")


def plot_adjustment_detail(raw: pd.DataFrame, adj: pd.DataFrame) -> None:
    raw = raw.copy(); adj = adj.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in raw.columns: raw[col] = raw[col].squeeze()
        if col in adj.columns: adj[col] = adj[col].squeeze()
    """
    Figure 2: Detailed split and dividend adjustment illustration.
      A. Price series zoomed around split event
      B. Ratio (Adj / Raw) over time — shows cumulative adjustment factor
      C. Total Return Index vs simple price index
      D. Data quality check summary table
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Module 01 — Corporate Action Adjustment & Total Return Index",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ----------------------------------------------------------------
    # Panel A: Zoomed view around split
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    n    = len(raw)
    mid  = n // 2
    zoom_start = max(0,   mid - 30)
    zoom_end   = min(n-1, mid + 30)
    zoom_idx   = raw.index[zoom_start:zoom_end]

    ax_a.plot(zoom_idx, raw["Close"].iloc[zoom_start:zoom_end],
              color=PALETTE["red"],   lw=2.0, label="Raw Close")
    ax_a.plot(zoom_idx, adj["Close"].iloc[zoom_start:zoom_end],
              color=PALETTE["green"], lw=2.0, label="Adjusted Close")
    
    split_line = raw.index[mid]
    ax_a.axvline(split_line, color=PALETTE["yellow"], lw=1.5, ls="--",
                 label="Split / Dividend date")
    ax_a.set_title("Price Around Corporate Action Event", fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Price (USD)")
    ax_a.legend()
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ----------------------------------------------------------------
    # Panel B: Adjustment factor over time
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    common = raw.index.intersection(adj.index)
    factor = (adj.loc[common, "Close"].squeeze() / raw.loc[common, "Close"].squeeze())
    
    ax_b.plot(common, factor, color=PALETTE["purple"], lw=1.8)
    ax_b.fill_between(common, 1.0, factor.values,
                      alpha=0.25, color=PALETTE["purple"])
    ax_b.axhline(1.0, color=PALETTE["grey"], lw=0.8, ls="--")
    ax_b.set_title("Cumulative Adjustment Factor  (Close Adjusted / Raw Close)",
                   fontweight="bold")
    ax_b.set_xlabel("Date")
    ax_b.set_ylabel("Adjustment Factor")
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    annotation_box(ax_b,
        "Factor < 1: stock has paid dividends\nor undergone splits over this period.",
        loc="lower left")

    # ----------------------------------------------------------------
    # Panel C: Total Return Index vs Price Index
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Total Return Index: reinvests dividends
    _ac = adj["Close"].squeeze()
    tri = (1 + np.log(_ac / _ac.shift(1)).fillna(0)).cumprod()
    pri = raw["Close"].squeeze(); pri = pri / pri.iloc[0]   # simple price index
    
    ax_c.plot(adj.index, tri,  color=PALETTE["green"], lw=2.0,
              label="Total Return Index (dividends reinvested)")
    ax_c.plot(raw.index, pri,  color=PALETTE["red"],   lw=1.5, ls="--",
              label="Price Index (raw, no adjustment)")
    ax_c.set_title("Total Return Index vs Price Index (base=1.0)",
                   fontweight="bold")
    ax_c.set_xlabel("Date")
    ax_c.set_ylabel("Index Value (base = 1.0)")
    ax_c.legend()
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    final_gap = tri.iloc[-1] - pri.iloc[-1]
    annotation_box(ax_c,
        f"Performance gap at end:\n"
        f"TRI = {tri.iloc[-1]:.2f}x  vs  PRI = {pri.iloc[-1]:.2f}x\n"
        f"Dividend contribution: {final_gap:.2f}x",
        loc="upper left")

    # ----------------------------------------------------------------
    # Panel D: Data quality report as visual table
    # ----------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    checks = quality_report(raw)
    rows = [
        ["Start date",          str(checks["start"])],
        ["End date",            str(checks["end"])],
        ["Total rows",          f"{checks['n_rows']:,}"],
        ["Years covered",       f"{checks['n_years']:.1f}"],
        ["Missing rows",        f"{checks['missing_rows']}  ({checks['missing_pct']:.2f}%)"],
        ["OHLC consistent",     "Yes" if checks["ohlc_consistent"] else "NO — CHECK DATA"],
        ["Zero/negative prices",f"{checks['zero_prices']}"],
        ["Extreme moves (>20%)",f"{checks['extreme_moves']}"],
    ]
    col_labels = ["Check", "Value"]
    table = ax_d.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style table cells
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

    ax_d.set_title("Data Quality Report", fontweight="bold",
                   color=PALETTE["white"], pad=12)

    save_fig(fig, "m01_adjustment_detail")


def plot_multi_asset_download(tickers_data: dict) -> None:
    """
    Figure 3: Multi-ticker comparison
      A. Normalised price series (base = 100)
      B. Rolling 30-day correlation heatmap
      C. Annualised return vs volatility scatter (risk-return space)
    """
    if len(tickers_data) < 2:
        print("  Skipping multi-asset plot (fewer than 2 tickers)")
        return

    names  = list(tickers_data.keys())
    prices = pd.DataFrame({n: tickers_data[n]["Close"]
                           for n in names}).dropna()

    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
              PALETTE["purple"], PALETTE["cyan"]]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Module 01 — Multi-Asset Analysis: Normalised Prices & Risk-Return",
                 fontsize=14, fontweight="bold", color=PALETTE["white"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ----------------------------------------------------------------
    # Panel A: Normalised price series
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:2])
    for i, name in enumerate(names):
        norm_p = prices[name] / prices[name].iloc[0] * 100
        ax_a.plot(prices.index, norm_p,
                  color=colors[i % len(colors)], lw=1.6, label=name)
    ax_a.axhline(100, color=PALETTE["grey"], lw=0.8, ls="--")
    ax_a.set_title("Normalised Price Series (Base = 100)",
                   fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Normalised Price")
    ax_a.legend(ncol=len(names))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ----------------------------------------------------------------
    # Panel B: Correlation matrix (full period)
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    log_rets = np.log(prices / prices.shift(1)).dropna()
    corr_mat = log_rets.corr()
    
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    im = ax_b.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    
    ax_b.set_xticks(range(len(names)))
    ax_b.set_yticks(range(len(names)))
    ax_b.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_b.set_yticklabels(names, fontsize=9)
    
    for i in range(len(names)):
        for j in range(len(names)):
            ax_b.text(j, i, f"{corr_mat.values[i,j]:.2f}",
                      ha="center", va="center", fontsize=9,
                      color="black" if abs(corr_mat.values[i,j]) < 0.7 else "white")
    ax_b.set_title("Return Correlation Matrix", fontweight="bold")

    # ----------------------------------------------------------------
    # Panel C: Risk-Return scatter
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    mu_ann    = log_rets.mean()    * 252
    sigma_ann = log_rets.std()     * np.sqrt(252)
    sharpe    = mu_ann / sigma_ann

    for i, name in enumerate(names):
        ax_c.scatter(sigma_ann[name]*100, mu_ann[name]*100,
                     s=200, color=colors[i % len(colors)],
                     zorder=4, label=f"{name} (SR={sharpe[name]:.2f})",
                     edgecolors="#0F1117", linewidths=1.5)
        ax_c.annotate(name,
                      (sigma_ann[name]*100, mu_ann[name]*100),
                      textcoords="offset points", xytext=(8, 4),
                      fontsize=9, color=colors[i % len(colors)])

    ax_c.axhline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    ax_c.axvline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    ax_c.set_title("Risk-Return Space (Annualised)", fontweight="bold")
    ax_c.set_xlabel("Annualised Volatility (%)")
    ax_c.set_ylabel("Annualised Return (%)")
    ax_c.legend(fontsize=8)
    annotation_box(ax_c,
        r"Sharpe = $\frac{\mu_{ann}}{\sigma_{ann}}$",
        loc="lower right")

    save_fig(fig, "m01_multi_asset")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  MODULE 01 — Financial Data Ingestion")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load single-ticker data (AAPL or synthetic)
    # ----------------------------------------------------------------
    print("\n[1] Loading OHLCV data...")
    raw, adj = load_ohlcv(ticker="AAPL", start="2018-01-01", end="2023-12-31")

    # ----------------------------------------------------------------
    # 2. Data quality checks
    # ----------------------------------------------------------------
    print("\n[2] Data quality report:")
    checks = quality_report(raw)
    for k, v in checks.items():
        if k != "extreme_dates":
            print(f"    {k:<25}: {v}")

    # ----------------------------------------------------------------
    # 3. Return analysis
    # ----------------------------------------------------------------
    print("\n[3] Return statistics (adjusted):")
    rets = compute_returns(adj["Close"])
    print(f"    Log return  mean (daily):  {rets['log'].mean():.6f}")
    print(f"    Log return  std  (daily):  {rets['log'].std():.6f}")
    print(f"    Ann. return:               {rets['log'].mean()*252:.2%}")
    print(f"    Ann. volatility:           {rets['log'].std()*np.sqrt(252):.2%}")
    print(f"    Skewness:                  {rets['log'].skew():.4f}")
    print(f"    Excess kurtosis:           {rets['log'].kurt():.4f}")

    # ----------------------------------------------------------------
    # 4. Generate figures
    # ----------------------------------------------------------------
    print("\n[4] Generating figures...")
    plot_ohlcv_overview(raw, adj)
    plot_adjustment_detail(raw, adj)

    # ----------------------------------------------------------------
    # 5. Multi-asset comparison (synthetic data for 4 tickers)
    # ----------------------------------------------------------------
    print("\n[5] Multi-asset comparison...")
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2018-01-01", "2023-12-31")
    n     = len(dates)

    params = {
        "AAPL": (0.18, 0.25, 150.0),
        "MSFT": (0.22, 0.23, 250.0),
        "SPY":  (0.12, 0.16, 280.0),
        "GLD":  (0.06, 0.14, 180.0),
    }
    # Build correlated paths via Cholesky
    # Correlation structure: [AAPL, MSFT, SPY, GLD]
    corr = np.array([
        [1.00, 0.75, 0.65, 0.05],
        [0.75, 1.00, 0.70, 0.03],
        [0.65, 0.70, 1.00, 0.08],
        [0.05, 0.03, 0.08, 1.00],
    ])
    L      = np.linalg.cholesky(corr)
    dt     = 1/252
    Z      = rng.normal(size=(n, 4))
    Z_corr = Z @ L.T

    tickers_data = {}
    for j, (name, (mu, sig, s0)) in enumerate(params.items()):
        log_r   = (mu - 0.5*sig**2)*dt + sig*np.sqrt(dt)*Z_corr[:, j]
        prices  = s0 * np.exp(np.cumsum(log_r))
        tickers_data[name] = pd.DataFrame(
            {"Close": prices}, index=dates)

    plot_multi_asset_download(tickers_data)

    # ----------------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------------
    print("\n[6] Output figures:")
    for f in sorted(OUT.glob("m01_*.png")):
        print(f"    {f.name}")
    print("\n  MODULE 01 COMPLETE")


if __name__ == "__main__":
    main()
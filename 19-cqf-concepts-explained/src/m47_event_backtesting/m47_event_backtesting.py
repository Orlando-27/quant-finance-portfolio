#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 47 -- EVENT-DRIVEN BACKTESTING: COMMISSIONS & SLIPPAGE
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Event-driven backtesting processes the market one bar at a time, simulating
the arrival of market events (bars, ticks, fills) in chronological order.
This architecture mirrors live execution systems and allows precise modelling
of market microstructure frictions that vectorised systems approximate away.

EVENT HIERARCHY
---------------
MarketEvent  -> new OHLCV bar available
SignalEvent  -> strategy emits directional signal (+1, -1, 0)
OrderEvent   -> portfolio converts signal to order (MKT, LMT, STP)
FillEvent    -> broker fills order at price + commission + slippage

TRANSACTION COST DECOMPOSITION
--------------------------------
The total cost of a round-trip trade has three components:

1. Commission:
   Fixed + proportional model:  C = max(C_min, c_rate * notional)
   Interactive Brokers tiered:  C = max($1, $0.005 * shares)

2. Bid-Ask Spread:
   The half-spread cost per trade: S/2 where S = ask - bid
   For liquid ETFs (SPY): S ~ 1 cent on a $450 stock ~ 0.2 bps

3. Market Impact (price slippage):
   Linear model:   slippage = sigma * sqrt(Q / ADV) * direction
   Square-root law: impact = alpha * sigma * sqrt(Q / V_daily)
   where Q = order size, ADV = average daily volume, sigma = volatility

Total friction per share:
   f = commission/shares + spread/2 + slippage

POSITION SIZING
---------------
Fixed fractional (Kelly-inspired):
   f* = mu / sigma^2  (full Kelly)
   f  = f* / 2        (half Kelly -- practical)

Volatility targeting:
   N_shares = (target_vol * capital) / (sigma_daily * price)
   This equalises risk contribution across time.

FILL MODELLING
--------------
Optimistic fill (vectorised): fill at close price, zero latency
Realistic fill:
  - Execution at next open (overnight latency)
  - Slippage proportional to bar range: slip = eta * (High - Low)
  - Partial fills: Q_filled = min(Q_ordered, ADV * max_participation)

REFERENCES
----------
[1] Chan, E. (2013). Algorithmic Trading. Wiley.
[2] Kissell, R. (2014). The Science of Algorithmic Trading. Academic Press.
[3] Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio
    Transactions." Journal of Risk 3(2):5-39.
[4] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m47_event_backtesting/m47_event_backtesting.py
=============================================================================
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m47")
os.makedirs(FIGS, exist_ok=True)

# =============================================================================
# DARK THEME
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
TEXT   = "#e6edf3"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
AMBER  = "#d29922"
VIOLET = "#bc8cff"
TEAL   = "#39d353"

plt.rcParams.update({
    "figure.facecolor" : DARK,  "axes.facecolor"  : PANEL,
    "axes.edgecolor"   : GRID,  "axes.labelcolor" : TEXT,
    "axes.titlecolor"  : TEXT,  "xtick.color"     : TEXT,
    "ytick.color"      : TEXT,  "text.color"      : TEXT,
    "grid.color"       : GRID,  "grid.linestyle"  : "--",
    "grid.alpha"       : 0.5,   "legend.facecolor": PANEL,
    "legend.edgecolor" : GRID,  "font.family"     : "monospace",
    "font.size"        : 9,     "axes.titlesize"  : 10,
})

def section(n, msg): print(f"  [{n:02d}] {msg}")

# =============================================================================
# 1.  EVENT CLASSES
# =============================================================================
@dataclass
class MarketEvent:
    """Signals arrival of new OHLCV bar."""
    bar_idx  : int
    open_    : float
    high     : float
    low      : float
    close    : float
    volume   : float

@dataclass
class SignalEvent:
    """Strategy-generated directional signal."""
    bar_idx  : int
    signal   : float          # +1 long, -1 short, 0 flat
    strength : float = 1.0   # optional signal strength [0,1]

@dataclass
class OrderEvent:
    """Portfolio-generated order."""
    bar_idx   : int
    order_type: str           # "MKT", "LMT", "STP"
    direction : float         # +1 buy, -1 sell
    quantity  : int           # shares
    limit_price: Optional[float] = None

@dataclass
class FillEvent:
    """Broker fill confirmation."""
    bar_idx    : int
    fill_price : float
    quantity   : int
    direction  : float
    commission : float
    slippage   : float

    @property
    def total_cost(self) -> float:
        return self.commission + abs(self.slippage * self.quantity)

# =============================================================================
# 2.  COST MODELS
# =============================================================================
class CommissionModel:
    """
    Interactive Brokers-style tiered commission model.
    C = max(C_min, rate * shares)
    """
    def __init__(self, rate: float = 0.005, min_comm: float = 1.0):
        self.rate     = rate
        self.min_comm = min_comm

    def calculate(self, shares: int, price: float) -> float:
        return max(self.min_comm, self.rate * shares)


class SlippageModel:
    """
    Fixed + volatility-scaled slippage model.

    slippage_per_share = fixed_bps * price / 10000
                       + vol_scale * bar_range * direction
    """
    def __init__(self, fixed_bps: float = 1.0, vol_scale: float = 0.1):
        self.fixed_bps = fixed_bps
        self.vol_scale = vol_scale

    def calculate(self, price: float, bar_range: float,
                  direction: float) -> float:
        fixed    = self.fixed_bps * price / 10000
        vol_slip = self.vol_scale * bar_range * direction
        return fixed + vol_slip


# =============================================================================
# 3.  PORTFOLIO & BROKER
# =============================================================================
class Portfolio:
    """
    Tracks cash, position, and equity curve in an event-driven loop.

    Position sizing: volatility targeting.
    N = (target_vol * equity) / (sigma_daily * price)
    """
    def __init__(self, initial_capital: float = 100_000.0,
                 target_vol: float = 0.10):
        self.capital      = initial_capital
        self.initial_cap  = initial_capital
        self.target_vol   = target_vol
        self.position     = 0          # current shares held
        self.equity_curve = []
        self.trade_log    = []

    def position_size(self, price: float, sigma: float) -> int:
        """Volatility-targeted share count."""
        if sigma < 1e-6 or price < 1e-6:
            return 0
        notional = self.target_vol * self.capital / (sigma * np.sqrt(252))
        return max(1, int(notional / price))

    def process_fill(self, fill: FillEvent):
        """Update cash and position after a fill."""
        cost = fill.fill_price * fill.quantity * fill.direction
        self.capital  -= cost + fill.total_cost
        self.position += int(fill.quantity * fill.direction)

    def update_equity(self, price: float):
        equity = self.capital + self.position * price
        self.equity_curve.append(equity)

    def record_trade(self, fill: FillEvent, signal: float):
        self.trade_log.append({
            "bar"       : fill.bar_idx,
            "direction" : fill.direction,
            "price"     : fill.fill_price,
            "qty"       : fill.quantity,
            "commission": fill.commission,
            "slippage"  : fill.slippage,
            "signal"    : signal,
        })


class Broker:
    """
    Simulated broker: executes orders at next-bar open + slippage.
    """
    def __init__(self, commission_model: CommissionModel,
                 slippage_model: SlippageModel):
        self.commission = commission_model
        self.slippage   = slippage_model

    def execute(self, order: OrderEvent, next_bar: MarketEvent) -> FillEvent:
        """Fill at next-bar open with friction."""
        exec_price = next_bar.open_
        bar_range  = next_bar.high - next_bar.low
        slip       = self.slippage.calculate(exec_price, bar_range, order.direction)
        fill_price = exec_price + slip
        comm       = self.commission.calculate(order.quantity, fill_price)
        return FillEvent(
            bar_idx    = next_bar.bar_idx,
            fill_price = fill_price,
            quantity   = order.quantity,
            direction  = order.direction,
            commission = comm,
            slippage   = slip,
        )


# =============================================================================
# 4.  STRATEGY: DUAL MA CROSSOVER (event-driven)
# =============================================================================
class DualMACrossover:
    """
    Event-driven implementation of the SMA crossover strategy.
    Maintains rolling price buffers for fast and slow MA computation.
    """
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast   = fast
        self.slow   = slow
        self.prices = deque(maxlen=slow)
        self.signal = 0.0

    def on_market(self, event: MarketEvent) -> Optional[SignalEvent]:
        self.prices.append(event.close)
        if len(self.prices) < self.slow:
            return None
        prices_arr = np.array(self.prices)
        sma_fast   = prices_arr[-self.fast:].mean()
        sma_slow   = prices_arr.mean()
        new_signal = 1.0 if sma_fast > sma_slow else -1.0
        if new_signal != self.signal:
            self.signal = new_signal
            return SignalEvent(event.bar_idx, new_signal)
        return None


# =============================================================================
# 5.  EVENT LOOP ENGINE
# =============================================================================
def run_backtest(ohlcv: np.ndarray, strategy, portfolio: Portfolio,
                 broker: Broker) -> dict:
    """
    Main event loop: processes bars sequentially, routing events through
    the strategy -> portfolio -> broker pipeline.

    Parameters
    ----------
    ohlcv     : (N, 5) array [open, high, low, close, volume]
    strategy  : strategy object with on_market() method
    portfolio : Portfolio instance
    broker    : Broker instance

    Returns
    -------
    dict with equity_curve, trade_log, total_commission, total_slippage
    """
    N = len(ohlcv)
    pending_order: Optional[OrderEvent] = None
    total_comm  = 0.0
    total_slip  = 0.0
    sigma_window = deque(maxlen=20)   # rolling vol for position sizing
    last_close   = ohlcv[0, 3]

    for i in range(N):
        bar = MarketEvent(
            bar_idx = i,
            open_   = ohlcv[i, 0],
            high    = ohlcv[i, 1],
            low     = ohlcv[i, 2],
            close   = ohlcv[i, 3],
            volume  = ohlcv[i, 4],
        )

        # Execute pending order at this bar's open
        if pending_order is not None:
            fill = broker.execute(pending_order, bar)
            portfolio.process_fill(fill)
            portfolio.record_trade(fill, fill.direction)
            total_comm += fill.commission
            total_slip += abs(fill.slippage * fill.quantity)
            pending_order = None

        # Update rolling volatility
        ret_i = np.log(bar.close / last_close + 1e-9)
        sigma_window.append(ret_i)
        sigma = np.std(sigma_window) if len(sigma_window) > 1 else 0.01
        last_close = bar.close

        # Strategy generates signal
        signal_event = strategy.on_market(bar)

        # Portfolio converts signal to order
        if signal_event is not None:
            target    = signal_event.signal
            current   = np.sign(portfolio.position) if portfolio.position != 0 else 0
            if target != current:
                qty       = portfolio.position_size(bar.close, sigma)
                direction = target
                # Flatten existing position first if needed
                if portfolio.position != 0:
                    flat_qty = abs(portfolio.position)
                    flat_dir = -np.sign(portfolio.position)
                    pending_order = OrderEvent(i, "MKT", flat_dir, flat_qty)
                else:
                    pending_order = OrderEvent(i, "MKT", direction, qty)

        portfolio.update_equity(bar.close)

    equity = np.array(portfolio.equity_curve)
    ret_eq = np.diff(np.log(equity + 1e-9))
    trades = portfolio.trade_log

    return {
        "equity"    : equity,
        "returns"   : ret_eq,
        "trades"    : trades,
        "total_comm": total_comm,
        "total_slip": total_slip,
        "n_trades"  : len(trades),
    }


def perf_metrics(equity: np.ndarray, ann: int = 252) -> dict:
    ret  = np.diff(np.log(equity + 1e-9))
    mu   = ret.mean();  sig = ret.std() + 1e-9
    neg  = ret[ret < 0]; sig_d = neg.std() + 1e-9 if len(neg) > 0 else sig
    peak = np.maximum.accumulate(equity)
    dd   = (peak - equity) / (peak + 1e-9)
    mdd  = dd.max()
    sr   = mu / sig * np.sqrt(ann)
    sor  = mu / sig_d * np.sqrt(ann)
    cagr = np.exp(mu * ann) - 1
    nz   = ret[ret != 0]
    hit  = (nz > 0).mean() if len(nz) > 0 else 0.5
    return {"sharpe": sr, "sortino": sor, "mdd": mdd,
            "cagr": cagr, "hit": hit}


# =============================================================================
# PRINT HEADER + LOAD DATA
# =============================================================================
print()
print("=" * 65)
print("  MODULE 47: EVENT-DRIVEN BACKTESTING")
print("  Events | Commission | Slippage | Vol-Target | Fill Model")
print("=" * 65)

raw  = yf.download("SPY", start="2015-01-01", end="2023-12-31",
                   auto_adjust=True, progress=False)
raw  = raw.dropna()
ohlcv = raw[["Open","High","Low","Close","Volume"]].values
dates = raw.index
N     = len(ohlcv)

section(1, f"SPY OHLCV: {N} bars  [{dates[0].date()} -- {dates[-1].date()}]")

# =============================================================================
# 6.  RUN BACKTESTS UNDER DIFFERENT COST ASSUMPTIONS
# =============================================================================
CAPITAL = 100_000.0

scenarios = {
    "Zero cost"      : (0.0,   0.0,  0.0),   # (fixed_bps, vol_scale, comm_rate)
    "Low cost"       : (0.5,   0.05, 0.003),
    "Realistic"      : (1.0,   0.10, 0.005),
    "High cost"      : (3.0,   0.20, 0.010),
}

results = {}
for name, (fbps, vscale, crate) in scenarios.items():
    strat = DualMACrossover(fast=20, slow=50)
    port  = Portfolio(CAPITAL, target_vol=0.10)
    comm  = CommissionModel(rate=crate, min_comm=1.0)
    slip  = SlippageModel(fixed_bps=fbps, vol_scale=vscale)
    brk   = Broker(comm, slip)
    res   = run_backtest(ohlcv, strat, port, brk)
    pm    = perf_metrics(res["equity"])
    res.update(pm)
    results[name] = res

# Buy-and-hold benchmark
bnh_ret   = np.log(ohlcv[:, 3] / np.roll(ohlcv[:, 3], 1))
bnh_ret[0]= 0.0
bnh_equity= CAPITAL * np.exp(np.cumsum(bnh_ret))
bnh_pm    = perf_metrics(bnh_equity)

section(2, "Scenarios computed: zero / low / realistic / high cost")

for name, res in results.items():
    section(0, f"{name:12s}  Sharpe={res['sharpe']:.3f}  MDD={res['mdd']:.3f}  "
               f"trades={res['n_trades']}  comm=${res['total_comm']:.0f}  "
               f"slip=${res['total_slip']:.0f}")

# =============================================================================
# 7.  SLIPPAGE DECOMPOSITION ANALYSIS
# =============================================================================
# Show how slippage scales with bar range for realistic scenario
res_real = results["Realistic"]
if res_real["trades"]:
    slips    = [t["slippage"] for t in res_real["trades"]]
    comms    = [t["commission"] for t in res_real["trades"]]
    prices   = [t["price"] for t in res_real["trades"]]
    slip_bps = [abs(s/p)*10000 for s, p in zip(slips, prices)]
    comm_bps = [c/p/q*10000 for c, p, q in
                zip(comms, prices, [t["qty"] for t in res_real["trades"]])]
    total_friction_bps = np.array(slip_bps) + np.array(comm_bps)
else:
    slip_bps = []; comm_bps = []; total_friction_bps = np.array([])

section(3, f"Realistic cost  mean_slip={np.mean(slip_bps):.2f}bps  "
           f"mean_comm={np.mean(comm_bps):.2f}bps  "
           f"mean_total={np.mean(total_friction_bps):.2f}bps")

# =============================================================================
# 8.  FILL PRICE DISTRIBUTION (optimistic close vs realistic open+slip)
# =============================================================================
# Compute difference: realistic fill vs close price as % of price
if res_real["trades"]:
    bar_indices = [t["bar"] for t in res_real["trades"] if t["bar"] < N]
    fill_vs_close = []
    for t in res_real["trades"]:
        if t["bar"] < N:
            close_px  = ohlcv[t["bar"], 3]
            fill_px   = t["price"]
            fill_vs_close.append((fill_px - close_px) / close_px * 10000)
else:
    fill_vs_close = [0.0]

section(4, f"Fill vs close: mean={np.mean(fill_vs_close):.2f}bps  "
           f"std={np.std(fill_vs_close):.2f}bps  "
           f"n_fills={len(fill_vs_close)}")

# =============================================================================
# FIGURE 1: EQUITY CURVES + DRAWDOWNS + COST IMPACT
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle("Module 47 -- Event-Driven Backtest: Commission & Slippage Impact",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.45)

colors_sc = [GREEN, ACCENT, AMBER, RED]
t_ = np.arange(N)

# 1A: Equity curves
ax = fig.add_subplot(gs[0, :])
ax.plot(t_, bnh_equity / CAPITAL * 100 - 100, color=TEXT, lw=1.0,
        alpha=0.6, label=f"Buy & Hold  SR={bnh_pm['sharpe']:.2f}")
for (name, res), col in zip(results.items(), colors_sc):
    eq_pct = res["equity"] / CAPITAL * 100 - 100
    n_plot = min(len(eq_pct), N)
    ax.plot(t_[:n_plot], eq_pct[:n_plot], color=col, lw=1.2,
            label=f"{name}  SR={res['sharpe']:.2f}  T={res['n_trades']}")
ax.set_title("Equity Curves by Cost Scenario (initial capital = $100,000)")
ax.set_xlabel("Bar"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=7, ncol=3); ax.grid(True)

# 1B: Drawdowns
ax = fig.add_subplot(gs[1, 0])
for (name, res), col in zip(results.items(), colors_sc):
    eq = res["equity"]
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-9) * 100
    ax.fill_between(t_[:len(dd)], -dd, 0, alpha=0.35, color=col,
                    label=f"{name} MDD={res['mdd']:.2f}")
ax.set_title("Drawdown by Cost Scenario (%)")
ax.set_xlabel("Bar"); ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1C: Performance bar comparison
ax = fig.add_subplot(gs[1, 1])
names = list(results.keys())
sharpes = [results[n]["sharpe"] for n in names]
mdds    = [results[n]["mdd"]    for n in names]
x_pos   = np.arange(len(names))
width   = 0.35
b1 = ax.bar(x_pos - width/2, sharpes, width, color=ACCENT, label="Sharpe", edgecolor=DARK)
b2 = ax.bar(x_pos + width/2, [-m for m in mdds], width, color=RED,
            label="-MDD", edgecolor=DARK)
ax.axhline(0, color=GRID, lw=0.8)
for bar, v in zip(b1, sharpes):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
ax.set_title("Sharpe & MDD by Cost Scenario")
ax.legend(fontsize=7); ax.grid(True, axis="y")

# 1D: Cost components (commission vs slippage)
ax = fig.add_subplot(gs[2, 0])
total_comms = [results[n]["total_comm"] for n in names]
total_slips = [results[n]["total_slip"] for n in names]
ax.bar(x_pos, total_comms, width=0.5, label="Commission", color=VIOLET, edgecolor=DARK)
ax.bar(x_pos, total_slips, width=0.5, bottom=total_comms,
       label="Slippage", color=AMBER, edgecolor=DARK)
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(" ","\\n") for n in names], fontsize=7)
ax.set_title("Total Transaction Costs ($)")
ax.set_ylabel("Cost ($)"); ax.legend(fontsize=7); ax.grid(True, axis="y")

# 1E: Friction distribution (realistic scenario)
ax = fig.add_subplot(gs[2, 1])
if len(total_friction_bps) > 0:
    ax.hist(total_friction_bps, bins=30, color=ACCENT, alpha=0.8,
            edgecolor=DARK, density=True)
    ax.axvline(np.mean(total_friction_bps), color=AMBER, lw=1.5, ls="--",
               label=f"mean={np.mean(total_friction_bps):.1f}bps")
ax.set_title("Total Friction Distribution (Realistic, bps)")
ax.set_xlabel("Total friction (bps)"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

fig.savefig(os.path.join(FIGS, "m47_fig1_equity_costs.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: FILL QUALITY + SLIPPAGE ANALYSIS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle("Module 47 -- Fill Quality & Slippage Decomposition",
             fontsize=12, color=TEXT, y=1.01)

# 2A: Fill vs close distribution
ax = axes[0, 0]
if fill_vs_close:
    ax.hist(fill_vs_close, bins=30, color=ACCENT, alpha=0.8,
            edgecolor=DARK, density=True)
    ax.axvline(0, color=GRID, lw=0.8, ls="--")
    ax.axvline(np.mean(fill_vs_close), color=AMBER, lw=1.5, ls="--",
               label=f"mean={np.mean(fill_vs_close):.1f}bps")
ax.set_title("Fill Price vs Close (bps)")
ax.set_xlabel("Deviation (bps)"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

# 2B: Slippage vs commission per trade
ax = axes[0, 1]
if slip_bps and comm_bps:
    ax.scatter(comm_bps, slip_bps, color=ACCENT, s=20, alpha=0.6)
    ax.set_xlabel("Commission (bps/share)"); ax.set_ylabel("Slippage (bps)")
    ax.set_title("Slippage vs Commission Per Trade")
ax.grid(True)

# 2C: Cumulative cost drag over time
ax = axes[1, 0]
if res_real["trades"]:
    trade_bars  = [t["bar"] for t in res_real["trades"]]
    cum_costs   = np.cumsum([t["commission"] + abs(t["slippage"]*t["qty"])
                             for t in res_real["trades"]])
    ax.plot(trade_bars, cum_costs, color=RED, lw=1.5, label="Cumulative costs")
    ax.set_title("Cumulative Transaction Cost Drag ($)")
    ax.set_xlabel("Bar index"); ax.set_ylabel("Cumulative cost ($)")
    ax.legend(fontsize=7); ax.grid(True)

# 2D: Architecture diagram
ax = axes[1, 1]
ax.set_facecolor(PANEL); ax.axis("off")
arch = (
    "EVENT-DRIVEN ARCHITECTURE\n"
    "==========================\n\n"
    "  [1] MarketEvent\n"
    "      new OHLCV bar arrives\n"
    "           |\n"
    "  [2] Strategy.on_market()\n"
    "      compute MAs, emit SignalEvent\n"
    "           |\n"
    "  [3] Portfolio.on_signal()\n"
    "      vol-target sizing, emit OrderEvent\n"
    "           |\n"
    "  [4] Broker.execute()\n"
    "      fill at next open + slippage\n"
    "      emit FillEvent\n"
    "           |\n"
    "  [5] Portfolio.on_fill()\n"
    "      update cash, position, equity\n\n"
    "  Cost model:\n"
    "    commission = max($1, $0.005 * shares)\n"
    "    slippage   = fixed_bps + vol_scale * range\n"
    "    fill_price = next_open + slippage\n"
)
ax.text(0.05, 0.95, arch, transform=ax.transAxes,
        fontsize=8.5, va="top", fontfamily="monospace",
        color=TEXT, linespacing=1.7)
ax.set_title("System Architecture")

for ax in axes.flat:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m47_fig2_fill_quality_architecture.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: TRADE ANALYSIS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("Module 47 -- Trade-Level Analysis (Realistic Scenario)",
             fontsize=12, color=TEXT, y=1.01)

real_trades = res_real["trades"]

# 3A: Trade P&L histogram (approximate from signal * next bar return)
if real_trades and len(real_trades) > 1:
    pnl_approx = []
    eq = res_real["equity"]
    for k in range(1, len(real_trades)):
        b0 = real_trades[k-1]["bar"]
        b1 = real_trades[k]["bar"]
        if b1 < len(eq) and b0 < len(eq):
            pnl_approx.append(eq[b1] - eq[b0])

    ax = axes[0]
    wins  = [p for p in pnl_approx if p > 0]
    loses = [p for p in pnl_approx if p <= 0]
    ax.hist(wins,  bins=20, color=GREEN, alpha=0.7, label=f"Wins n={len(wins)}")
    ax.hist(loses, bins=20, color=RED,   alpha=0.7, label=f"Losses n={len(loses)}")
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_title("Inter-Trade P&L Distribution")
    ax.set_xlabel("P&L ($)"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.grid(True)

# 3B: Holding period distribution
if len(real_trades) > 1:
    hold_periods = []
    for k in range(1, len(real_trades)):
        hold_periods.append(real_trades[k]["bar"] - real_trades[k-1]["bar"])
    ax = axes[1]
    ax.hist(hold_periods, bins=30, color=VIOLET, alpha=0.8, edgecolor=DARK)
    ax.axvline(np.mean(hold_periods), color=AMBER, lw=1.5, ls="--",
               label=f"mean={np.mean(hold_periods):.0f}d")
    ax.set_title("Holding Period Distribution (days)")
    ax.set_xlabel("Days held"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.grid(True)

# 3C: Cost per trade over time
if real_trades:
    cost_per_trade = [t["commission"] + abs(t["slippage"]*t["qty"])
                      for t in real_trades]
    ax = axes[2]
    ax.scatter(range(len(cost_per_trade)), cost_per_trade,
               color=AMBER, s=15, alpha=0.7)
    ax.axhline(np.mean(cost_per_trade), color=RED, lw=1.2, ls="--",
               label=f"mean=${np.mean(cost_per_trade):.1f}")
    ax.set_title("Transaction Cost Per Trade ($)")
    ax.set_xlabel("Trade number"); ax.set_ylabel("Cost ($)")
    ax.legend(fontsize=7); ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m47_fig3_trade_analysis.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
real = results["Realistic"]
zero = results["Zero cost"]
print()
print("  MODULE 47 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Event loop: MarketEvent->SignalEvent->OrderEvent->FillEvent")
print("  [2] Fill at next-bar open eliminates look-ahead bias")
print("  [3] commission = max($1, rate * shares)  IB-style model")
print("  [4] slippage = fixed_bps + vol_scale * bar_range")
print("  [5] Vol-target sizing: N = target_vol * equity / (sigma * price)")
print(f"  [6] Zero cost Sharpe={zero['sharpe']:.3f}  "
      f"Realistic Sharpe={real['sharpe']:.3f}  "
      f"degradation={zero['sharpe']-real['sharpe']:.3f}")
print(f"  [7] Realistic: {real['n_trades']} trades  "
      f"comm=${real['total_comm']:.0f}  slip=${real['total_slip']:.0f}")
print(f"  [8] Mean friction={np.mean(total_friction_bps):.1f}bps  "
      f"fill_dev={np.mean(fill_vs_close):.1f}bps vs close")
print(f"  NEXT: M48 -- Factor IC, Signal Decay & Alphalens Tearsheets")
print()

"""
M56 -- Algorithmic Trading: Momentum & Mean Reversion Signals
==============================================================
CQF Concepts Explained | Project 19 | Quantitative Finance Portfolio

Theory
------
Two empirically robust anomalies drive systematic trading strategies:

1. Time-Series Momentum (TSMOM)
--------------------------------
Asset i is long (short) if its own past return is positive (negative):
    signal_i,t = sign( r_{i,t-h:t} )
Volatility-scaled position:
    w_{i,t} = signal_i,t * (sigma_target / sigma_{i,t})
where sigma_target = 0.40/sqrt(252) (40% annualised target vol).

Moskowitz, Ooi & Pedersen (2012) document TSMOM profits across 58
liquid futures contracts over 25 years.

2. Cross-Sectional Momentum (CSMOM)
-------------------------------------
Rank all assets by past J-month return, long top decile, short bottom:
    w_{i,t} = rank_i,t / (n/2) - 1   (normalised to [-1,+1])
Jegadeesh & Titman (1993) document 12-1 momentum in US equities.

3. Mean Reversion: Ornstein-Uhlenbeck Signal
---------------------------------------------
OU process:  dX_t = kappa*(mu - X_t)*dt + sigma*dW_t
Discrete:    X_{t+1} = X_t + kappa*(mu - X_t) + sigma*eps_t

Half-life:   t_{1/2} = ln(2) / kappa
Z-score:     z_t = (X_t - EMA(X_t, tau)) / RollingStd(X_t, tau)
Signal:      short if z_t > +z_entry, long if z_t < -z_entry
             exit if |z_t| < z_exit

4. Volatility Scaling (risk parity position sizing)
----------------------------------------------------
w_i = signal_i * (sigma_target / sigma_{realised,i,t})
Ensures each position contributes equal ex-ante volatility.

5. Signal Combination
---------------------
Combined signal: w_comb = alpha*w_mom + (1-alpha)*w_mr
Weights alpha optimised by walk-forward Sharpe maximisation.

References
----------
Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum", JFE 104(2)
Jegadeesh & Titman (1993) "Returns to Buying Winners...", JF 48(1)
Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
ACCENT = "#58a6ff"
GOLD   = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  TEXT,
})

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "m56_algo_trading")
os.makedirs(FIGS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print()
print("=" * 65)
print("  MODULE 56: ALGORITHMIC TRADING")
print("  TSMOM | CSMOM | OU Mean Reversion | Vol Scaling | Backtest")
print("=" * 65)

# =============================================================================
# 1. SYNTHETIC MULTI-ASSET UNIVERSE
# =============================================================================
# 8 assets with regime-dependent momentum and mean-reversion properties.
N_ASSETS = 8
N_DAYS   = 2000
DT       = 1 / 252

# Regime: alternating momentum / mean-reversion windows
regime_len = 200
regimes = np.zeros(N_DAYS, dtype=int)
for i in range(N_DAYS // regime_len):
    regimes[i*regime_len:(i+1)*regime_len] = i % 2  # 0=momentum, 1=mean-rev

# Correlated asset returns
rho_base  = 0.30
cov_base  = (np.full((N_ASSETS, N_ASSETS), rho_base)
             + np.eye(N_ASSETS) * (1 - rho_base))
vol_assets = 0.01 + 0.01 * np.random.rand(N_ASSETS)   # 1-2% daily vol
D  = np.diag(vol_assets)
Sigma = D @ cov_base @ D
L    = np.linalg.cholesky(Sigma)

# Drift: momentum regime has trending drift, MR regime has near-zero drift
drift_trend = (np.random.randn(N_ASSETS) * 0.0003
               + np.array([0.0003, -0.0002, 0.0004, -0.0003,
                            0.0002, -0.0004, 0.0003, -0.0002]))
drift_mr    = np.zeros(N_ASSETS)

eps  = (L @ np.random.randn(N_ASSETS, N_DAYS)).T
ret  = np.zeros((N_DAYS, N_ASSETS))
for t in range(N_DAYS):
    d = drift_trend if regimes[t] == 0 else drift_mr
    ret[t] = d + eps[t]

price = np.cumprod(1 + ret, axis=0) * 100.0

print(f"  [01] Universe: {N_ASSETS} assets  N={N_DAYS} days")
print(f"       Vol range: {vol_assets.min()*100:.2f}%--{vol_assets.max()*100:.2f}%  "
      f"Base corr={rho_base}")

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    cs  = np.cumsum(x)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-(w)]]) ) / w
    return out

def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(w-1, len(x)):
        out[t] = x[t-w+1:t+1].std()
    return out

def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    cs  = np.cumsum(x)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-(w)]]))
    return out

def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out   = np.full_like(x, np.nan)
    # Find first non-nan
    start = 0
    while start < len(x) and np.isnan(x[start]):
        start += 1
    if start >= len(x):
        return out
    out[start] = x[start]
    for t in range(start + 1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t-1]
    return out

def realised_vol(ret_col: np.ndarray, w: int = 21) -> np.ndarray:
    rv = rolling_std(ret_col, w) * np.sqrt(252)
    rv = np.where(rv < 0.01, 0.01, rv)   # floor at 1% ann
    return rv

# =============================================================================
# 3. TIME-SERIES MOMENTUM SIGNAL
# =============================================================================
LOOKBACK_MOM = 63    # ~3-month momentum
SKIP         = 5     # skip last 5 days (avoid reversal)
VOL_W        = 21    # vol estimation window
SIGMA_TGT    = 0.40 / np.sqrt(252)   # daily target vol

def tsmom_signal(ret: np.ndarray) -> np.ndarray:
    """
    TSMOM volatility-scaled position for each asset.
    Returns position matrix (N_DAYS, N_ASSETS).
    """
    T, A = ret.shape
    pos  = np.zeros((T, A))
    for a in range(A):
        r    = ret[:, a]
        # Past return over lookback (skip last `SKIP` days)
        cum  = rolling_sum(r, LOOKBACK_MOM)
        skip = rolling_sum(r, SKIP)
        signal = np.sign(cum - skip)
        # Volatility scaling
        rv   = realised_vol(r, VOL_W)
        pos[:, a] = signal * (SIGMA_TGT / rv)
    return pos

pos_mom = tsmom_signal(ret)
print(f"  [02] TSMOM: lookback={LOOKBACK_MOM}d  skip={SKIP}d  "
      f"sigma_tgt={SIGMA_TGT*100:.3f}%/day")

# =============================================================================
# 4. CROSS-SECTIONAL MOMENTUM SIGNAL
# =============================================================================
LOOKBACK_CS = 63

def csmom_signal(ret: np.ndarray) -> np.ndarray:
    """
    CSMOM: rank assets by lookback return, long top half, short bottom half.
    Volatility-scaled positions.
    """
    T, A  = ret.shape
    pos   = np.zeros((T, A))
    for t in range(LOOKBACK_CS, T):
        cum_ret = ret[t-LOOKBACK_CS:t].sum(axis=0)
        ranks   = cum_ret.argsort().argsort().astype(float)
        # Normalise ranks to [-1, +1]
        norm    = ranks / (A - 1) * 2 - 1
        # Volatility scale
        rv      = np.array([realised_vol(ret[:t, a], VOL_W)[t-1]
                             for a in range(A)])
        pos[t]  = norm * (SIGMA_TGT / rv)
    return pos

pos_cs = csmom_signal(ret)
print(f"  [03] CSMOM: lookback={LOOKBACK_CS}d  cross-sectional rank normalisation")

# =============================================================================
# 5. ORNSTEIN-UHLENBECK MEAN REVERSION SIGNAL
# =============================================================================
# Estimate OU parameters for each asset via OLS regression:
#   X_{t+1} - X_t = kappa*(mu - X_t)*dt + sigma*eps
#   => delta_X = a + b*X_t + eps  =>  kappa = -b/dt, mu = -a/b

OU_WIN    = 63     # rolling window for OU parameter estimation
Z_ENTRY   = 1.5   # enter trade when |z| > z_entry
Z_EXIT    = 0.5   # exit trade when |z| < z_exit

def ou_params(price_col: np.ndarray, w: int) -> tuple:
    """Estimate OU kappa and half-life from rolling window OLS."""
    y = np.diff(price_col[-w:])
    x = price_col[-w-1:-1]
    n   = min(len(y), len(x))
    y, x = y[-n:], x[-n:]
    beta = np.cov(y, x)[0, 1] / (np.var(x) + 1e-12)
    kappa = -beta / DT
    kappa = max(kappa, 1e-4)
    half_life = np.log(2) / kappa
    return float(kappa), float(half_life)

def mr_signal(price: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """
    Mean reversion z-score signal with OU-inspired z-bands.
    Entry/exit based on price z-score vs rolling mean.
    """
    T, A = price.shape
    pos  = np.zeros((T, A))
    for a in range(A):
        p  = price[:, a]
        mu_roll = rolling_mean(p, OU_WIN)
        sd_roll = rolling_std(p, OU_WIN)
        z   = (p - mu_roll) / (sd_roll + 1e-9)
        cur_pos = 0.0
        for t in range(OU_WIN, T):
            if np.isnan(z[t]):
                continue
            if cur_pos == 0:
                if z[t] > Z_ENTRY:
                    cur_pos = -1.0    # short: price above mean
                elif z[t] < -Z_ENTRY:
                    cur_pos =  1.0    # long:  price below mean
            else:
                if abs(z[t]) < Z_EXIT:
                    cur_pos = 0.0     # exit
            # Volatility scaling
            rv = realised_vol(ret[:t, a], VOL_W)[t-1]
            pos[t, a] = cur_pos * (SIGMA_TGT / rv)
    return pos

pos_mr = mr_signal(price, ret)
print(f"  [04] MR (OU): z_entry={Z_ENTRY}  z_exit={Z_EXIT}  win={OU_WIN}d")

# OU half-life diagnostics (last window, each asset)
hl_list = []
for a in range(N_ASSETS):
    _, hl = ou_params(price[:, a], OU_WIN)
    hl_list.append(hl)
print(f"       OU half-lives: min={min(hl_list):.1f}d  "
      f"max={max(hl_list):.1f}d  "
      f"mean={np.mean(hl_list):.1f}d")

# =============================================================================
# 6. COMBINED SIGNAL + WALK-FORWARD BACKTEST
# =============================================================================
TC       = 0.0005    # one-way transaction cost (5 bps per asset)
WARMUP   = max(LOOKBACK_MOM, LOOKBACK_CS, OU_WIN) + 1
ALPHA    = 0.50      # combination weight (equal blend)

pos_comb = ALPHA * pos_mom + (1 - ALPHA) * pos_mr

def backtest(pos: np.ndarray, ret: np.ndarray,
             tc: float = TC, warmup: int = WARMUP) -> np.ndarray:
    """
    Portfolio backtest: gross PnL - transaction costs.
    Returns cumulative portfolio log-return series.
    """
    T, A = ret.shape
    pnl  = np.zeros(T)
    for t in range(warmup, T):
        gross  = np.sum(pos[t] * ret[t])
        # TC on position changes
        if t > warmup:
            turnover = np.sum(np.abs(pos[t] - pos[t-1]))
        else:
            turnover = np.sum(np.abs(pos[t]))
        pnl[t] = gross - tc * turnover
    return np.cumsum(pnl)

pnl_mom  = backtest(pos_mom,  ret)
pnl_cs   = backtest(pos_cs,   ret)
pnl_mr   = backtest(pos_mr,   ret)
pnl_comb = backtest(pos_comb, ret)

# Equal-weight buy-and-hold benchmark
ew_pos = np.ones((N_DAYS, N_ASSETS)) / N_ASSETS
pnl_bnh = backtest(ew_pos, ret, tc=0)

def sharpe(pnl, warmup=WARMUP, freq=252):
    d = np.diff(pnl[warmup:])
    return float(np.mean(d) / (np.std(d) + 1e-9) * np.sqrt(freq))

def max_dd(pnl):
    peak = np.maximum.accumulate(pnl)
    return float(np.max(peak - pnl))

def ann_ret(pnl, warmup=WARMUP, freq=252):
    n = len(pnl) - warmup
    return float(pnl[-1] - pnl[warmup]) / n * freq

print(f"  [05] Backtest Results (N={N_DAYS}d  warmup={WARMUP}d):")
print(f"       {'Strategy':<14} {'Ann Ret':>9} {'Sharpe':>8} {'Max DD':>8}")
for name, pnl in [("TSMOM",    pnl_mom),
                  ("CSMOM",    pnl_cs),
                  ("MR (OU)",  pnl_mr),
                  ("Combined", pnl_comb),
                  ("EW BnH",   pnl_bnh)]:
    print(f"       {name:<14} {ann_ret(pnl)*100:>8.2f}%"
          f" {sharpe(pnl):>8.3f} {max_dd(pnl):>8.4f}")

# =============================================================================
# 7. WALK-FORWARD ANALYSIS -- rolling 252d Sharpe
# =============================================================================
WF_WIN = 252

def rolling_sharpe(pnl, w=WF_WIN):
    d   = np.diff(pnl)
    out = np.full(len(pnl), np.nan)
    for t in range(w, len(d)):
        seg = d[t-w:t]
        out[t+1] = float(np.mean(seg) / (np.std(seg) + 1e-9) * np.sqrt(252))
    return out

rs_mom  = rolling_sharpe(pnl_mom)
rs_mr   = rolling_sharpe(pnl_mr)
rs_comb = rolling_sharpe(pnl_comb)

# =============================================================================
# 8. FIGURE 1 -- Signals & Positions
# =============================================================================
t_ax = np.arange(N_DAYS)
ASSET_SHOW = 0   # asset 0 for signal visualisation

fig = plt.figure(figsize=(15, 10), facecolor=DARK)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M56 -- Algorithmic Trading: Signal Construction",
             color=TEXT, fontsize=11)

# 8A: Price & MR z-score for asset 0
mu_roll = rolling_mean(price[:, ASSET_SHOW], OU_WIN)
sd_roll = rolling_std(price[:, ASSET_SHOW], OU_WIN)
z_score = (price[:, ASSET_SHOW] - mu_roll) / (sd_roll + 1e-9)

ax = fig.add_subplot(gs[0, :])
ax2 = ax.twinx()
ax.plot(t_ax, price[:, ASSET_SHOW], color=TEXT, lw=0.6, alpha=0.7,
        label=f"Asset {ASSET_SHOW} Price")
ax.plot(t_ax, mu_roll, color=GOLD, lw=1.2, ls="--", label="Rolling Mean")
ax2.plot(t_ax, z_score, color=ACCENT, lw=0.8, alpha=0.6, label="Z-score")
ax2.axhline( Z_ENTRY, color=RED,   lw=0.8, ls="--", alpha=0.7)
ax2.axhline(-Z_ENTRY, color=GREEN, lw=0.8, ls="--", alpha=0.7)
ax2.axhline(0, color=TEXT, lw=0.4, ls=":")
ax2.set_ylabel("Z-Score", color=ACCENT)
ax.set_title(f"Asset {ASSET_SHOW}: Price, Rolling Mean & MR Z-score")
ax.set_ylabel("Price")
ax.set_xlabel("Day")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
ax.set_facecolor(PANEL)
ax.grid(True)

# 8B: TSMOM position heatmap
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(pos_mom[WARMUP:].T, aspect="auto", cmap="RdYlGn",
               vmin=-3, vmax=3, origin="upper")
ax.set_title(f"TSMOM Positions (all assets)\n(rows=assets, cols=time)")
ax.set_xlabel("Day (post-warmup)")
ax.set_ylabel("Asset")
ax.set_yticks(range(N_ASSETS))
ax.set_yticklabels([f"A{i}" for i in range(N_ASSETS)], fontsize=6)
fig.colorbar(im, ax=ax, label="Position", fraction=0.046, pad=0.04)
ax.set_facecolor(PANEL)

# 8C: MR position heatmap
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(pos_mr[WARMUP:].T, aspect="auto", cmap="RdYlGn",
               vmin=-3, vmax=3, origin="upper")
ax.set_title(f"MR (OU) Positions (all assets)")
ax.set_xlabel("Day (post-warmup)")
ax.set_ylabel("Asset")
ax.set_yticks(range(N_ASSETS))
ax.set_yticklabels([f"A{i}" for i in range(N_ASSETS)], fontsize=6)
fig.colorbar(im, ax=ax, label="Position", fraction=0.046, pad=0.04)
ax.set_facecolor(PANEL)

# 8D: Turnover comparison
ax = fig.add_subplot(gs[2, 0])
for (name, pos, col) in [("TSMOM",    pos_mom,  ACCENT),
                           ("CSMOM",    pos_cs,   GREEN),
                           ("MR (OU)",  pos_mr,   GOLD),
                           ("Combined", pos_comb, RED)]:
    turn = np.array([np.sum(np.abs(pos[t] - pos[t-1]))
                     for t in range(WARMUP+1, N_DAYS)])
    roll_turn = np.convolve(turn, np.ones(21)/21, mode="valid")
    ax.plot(roll_turn, color=col, lw=1.0, label=name, alpha=0.8)
ax.set_title("Rolling 21d Average Turnover\n(sum |delta_pos| per day)")
ax.set_xlabel("Day")
ax.set_ylabel("Turnover")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 8E: Regime overlay on regime indicator
ax = fig.add_subplot(gs[2, 1])
ax.fill_between(t_ax, regimes, 0, color=ACCENT, alpha=0.4,
                label="Regime 0: Trend")
ax.fill_between(t_ax, 1 - regimes, 0, color=GOLD, alpha=0.4,
                label="Regime 1: Mean-Rev")
ax.plot(t_ax, ema(np.abs(z_score), 21), color=RED, lw=1.0,
        label="|z| EMA(21)")
ax.axhline(Z_ENTRY, color=TEXT, lw=0.6, ls="--")
ax.set_title("Regime Structure & |Z-score| EMA")
ax.set_xlabel("Day")
ax.set_ylabel("Regime / |Z-score|")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.savefig(os.path.join(FIGS, "m56_fig1_signals_positions.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [06] Fig 1 saved: signals & position heatmaps")

# =============================================================================
# 9. FIGURE 2 -- Backtest PnL
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=DARK)
fig.suptitle("M56 -- Backtest: Cumulative PnL & Performance Attribution",
             color=TEXT, fontsize=11)

# 9A: Cumulative PnL
ax = axes[0, 0]
for name, pnl, col in [("TSMOM",    pnl_mom,  ACCENT),
                        ("CSMOM",    pnl_cs,   GREEN),
                        ("MR (OU)",  pnl_mr,   GOLD),
                        ("Combined", pnl_comb, RED),
                        ("EW BnH",   pnl_bnh,  TEXT)]:
    ax.plot(t_ax, pnl, color=col, lw=1.4 if name=="Combined" else 1.0,
            label=name, alpha=0.9)
ax.axhline(0, color=TEXT, lw=0.4, ls=":")
ax.set_title("Cumulative PnL: All Strategies")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative Log-Return")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9B: Drawdown for each strategy
ax = axes[0, 1]
for name, pnl, col in [("TSMOM",    pnl_mom,  ACCENT),
                        ("Combined", pnl_comb, RED),
                        ("EW BnH",   pnl_bnh,  TEXT)]:
    peak = np.maximum.accumulate(pnl)
    dd   = peak - pnl
    ax.plot(t_ax, -dd, color=col, lw=1.0, label=name)
ax.set_title("Drawdown: TSMOM vs Combined vs BnH")
ax.set_xlabel("Day")
ax.set_ylabel("Drawdown")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9C: Rolling 252d Sharpe
ax = axes[1, 0]
ax.plot(t_ax, rs_mom,  color=ACCENT, lw=1.0, label="TSMOM")
ax.plot(t_ax, rs_mr,   color=GOLD,   lw=1.0, label="MR (OU)")
ax.plot(t_ax, rs_comb, color=RED,    lw=1.5, label="Combined")
ax.axhline(0, color=TEXT, lw=0.6, ls="--")
ax.axhline(1, color=GREEN, lw=0.6, ls=":", alpha=0.6)
ax.set_title(f"Rolling {WF_WIN}d Sharpe Ratio")
ax.set_xlabel("Day")
ax.set_ylabel("Sharpe")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9D: Sharpe & Ann Return bar chart
ax = axes[1, 1]
strats  = ["TSMOM", "CSMOM", "MR (OU)", "Combined", "EW BnH"]
pnls_l  = [pnl_mom, pnl_cs, pnl_mr, pnl_comb, pnl_bnh]
sharpes = [sharpe(p) for p in pnls_l]
ann_rets = [ann_ret(p) * 100 for p in pnls_l]
cols_b  = [ACCENT, GREEN, GOLD, RED, TEXT]

x = np.arange(len(strats))
w = 0.35
bars1 = ax.bar(x - w/2, sharpes,   w, color=cols_b, alpha=0.8,
               edgecolor=DARK, linewidth=0.5, label="Sharpe")
ax2 = ax.twinx()
bars2 = ax2.bar(x + w/2, ann_rets, w, color=cols_b, alpha=0.4,
                edgecolor=DARK, linewidth=0.5, hatch="//", label="Ann Ret %")
ax.set_xticks(x)
ax.set_xticklabels(strats, rotation=15, ha="right", fontsize=7)
ax.set_ylabel("Sharpe Ratio")
ax2.set_ylabel("Ann Return (%)")
ax.axhline(0, color=TEXT, lw=0.4)
ax.set_title("Sharpe Ratio & Annualised Return")
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")
handles = [bars1, bars2]
labels  = ["Sharpe", "Ann Ret %"]
ax.legend(handles, labels, fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m56_fig2_backtest_pnl.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [07] Fig 2 saved: backtest PnL, drawdown, rolling Sharpe")

# =============================================================================
# 10. FIGURE 3 -- Signal Diagnostics & Alpha Decay
# =============================================================================
fig = plt.figure(figsize=(15, 8), facecolor=DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M56 -- Signal Diagnostics: IC, Alpha Decay & Combination",
             color=TEXT, fontsize=11)

# 10A: Information Coefficient (IC) -- signal vs next-day return
def ic_series(pos: np.ndarray, ret: np.ndarray,
              warmup: int = WARMUP) -> np.ndarray:
    """Pearson IC between today's position and tomorrow's return."""
    T = ret.shape[0]
    ic = np.full(T, np.nan)
    for t in range(warmup, T-1):
        p_t = pos[t]
        r_t = ret[t+1]
        if p_t.std() > 1e-9:
            ic[t] = float(np.corrcoef(p_t, r_t)[0, 1])
    return ic

ic_mom  = ic_series(pos_mom, ret)
ic_mr   = ic_series(pos_mr,  ret)

ax = fig.add_subplot(gs[0, 0])
ic_mom_clean = ic_mom[~np.isnan(ic_mom)]
ic_mr_clean  = ic_mr[~np.isnan(ic_mr)]
ax.hist(ic_mom_clean, bins=40, density=True, color=ACCENT, alpha=0.6,
        label=f"TSMOM IC\nMean={ic_mom_clean.mean():.4f}")
ax.hist(ic_mr_clean,  bins=40, density=True, color=GOLD,   alpha=0.6,
        label=f"MR IC\nMean={ic_mr_clean.mean():.4f}")
ax.axvline(0, color=TEXT, lw=0.8, ls="--")
ax.set_title("Information Coefficient Distribution")
ax.set_xlabel("IC (signal vs next-day return)")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 10B: Alpha decay -- TSMOM at multiple horizons
ax = fig.add_subplot(gs[0, 1])
horizons = np.arange(1, 31)
ic_by_h  = []
for h in horizons:
    ic_h = []
    for t in range(WARMUP, N_DAYS - h):
        p_t = pos_mom[t]
        r_h = ret[t:t+h].sum(axis=0)
        if p_t.std() > 1e-9:
            ic_h.append(float(np.corrcoef(p_t, r_h)[0, 1]))
    ic_by_h.append(np.mean(ic_h) if ic_h else 0.0)

ax.bar(horizons, ic_by_h, color=ACCENT, alpha=0.7,
       edgecolor=DARK, linewidth=0.3)
ax.axhline(0, color=TEXT, lw=0.6, ls="--")
ax.set_title("TSMOM Alpha Decay\n(IC vs forward horizon)")
ax.set_xlabel("Holding Period (days)")
ax.set_ylabel("Mean IC")
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

# 10C: Signal combination frontier (alpha sweep)
ax = fig.add_subplot(gs[0, 2])
alphas = np.linspace(0, 1, 41)
combo_sharpes = []
for a in alphas:
    p_c = a * pos_mom + (1 - a) * pos_mr
    pnl_c = backtest(p_c, ret)
    combo_sharpes.append(sharpe(pnl_c))

ax.plot(alphas, combo_sharpes, color=RED, lw=1.5, marker="o", markersize=3)
best_alpha = alphas[np.argmax(combo_sharpes)]
ax.axvline(best_alpha, color=GOLD, lw=1.2, ls="--",
           label=f"Optimal alpha={best_alpha:.2f}")
ax.axvline(ALPHA, color=TEXT, lw=0.8, ls=":",
           label=f"Used alpha={ALPHA:.2f}")
ax.set_title("Signal Combination Frontier\n(TSMOM weight alpha)")
ax.set_xlabel("alpha (weight on TSMOM)")
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 10D: OU half-life per asset
ax = fig.add_subplot(gs[1, 0])
ax.bar(range(N_ASSETS), hl_list, color=PURPLE, alpha=0.8,
       edgecolor=DARK, linewidth=0.5)
ax.axhline(np.mean(hl_list), color=GOLD, lw=1.2, ls="--",
           label=f"Mean={np.mean(hl_list):.1f}d")
ax.set_title("OU Half-Life per Asset\n(t_{1/2} = ln(2)/kappa)")
ax.set_xlabel("Asset")
ax.set_ylabel("Half-Life (days)")
ax.set_xticks(range(N_ASSETS))
ax.set_xticklabels([f"A{i}" for i in range(N_ASSETS)], fontsize=7)
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

# 10E: Rolling IC (21d) for TSMOM and MR
ax = fig.add_subplot(gs[1, 1:])
ic_roll_mom = np.array([
    np.nanmean(ic_mom[max(0,t-21):t]) for t in range(N_DAYS)
])
ic_roll_mr = np.array([
    np.nanmean(ic_mr[max(0,t-21):t]) for t in range(N_DAYS)
])
ax.plot(t_ax, ic_roll_mom, color=ACCENT, lw=1.0, label="TSMOM IC(21)")
ax.plot(t_ax, ic_roll_mr,  color=GOLD,   lw=1.0, label="MR IC(21)")
ax.axhline(0, color=TEXT, lw=0.6, ls="--")
# Shade regimes
for i in range(N_DAYS // regime_len):
    st = i * regime_len
    en = (i + 1) * regime_len
    col_r = ACCENT if i % 2 == 0 else GOLD
    ax.axvspan(st, en, alpha=0.06, color=col_r)
ax.set_title("Rolling 21d IC: TSMOM vs MR\n(shading: trend/MR regimes)")
ax.set_xlabel("Day")
ax.set_ylabel("Rolling IC")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.savefig(os.path.join(FIGS, "m56_fig3_signal_diagnostics.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [08] Fig 3 saved: IC distribution, alpha decay, combination frontier")

# =============================================================================
# SUMMARY
# =============================================================================
opt_pnl = backtest(best_alpha * pos_mom + (1-best_alpha) * pos_mr, ret)

print()
print("  MODULE 56 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] TSMOM: sign(past_return) * (sigma_tgt/sigma_realised)")
print("  [2] CSMOM: rank assets by lookback return, long top / short bottom")
print("  [3] OU MR: dX = kappa*(mu-X)dt + sigma*dW  =>  z-score entry/exit")
print(f"  [4] OU half-lives: {min(hl_list):.1f}d -- {max(hl_list):.1f}d  "
      f"mean={np.mean(hl_list):.1f}d")
print(f"  [5] TSMOM Sharpe={sharpe(pnl_mom):.3f}  "
      f"CSMOM Sharpe={sharpe(pnl_cs):.3f}  "
      f"MR Sharpe={sharpe(pnl_mr):.3f}")
print(f"  [6] Combined (alpha={ALPHA}) Sharpe={sharpe(pnl_comb):.3f}  "
      f"MDD={max_dd(pnl_comb):.4f}")
print(f"  [7] Optimal alpha={best_alpha:.2f}  "
      f"Opt Sharpe={sharpe(opt_pnl):.3f}")
print(f"  [8] Vol scaling reduces MDD vs EW BnH by "
      f"{(max_dd(pnl_bnh)-max_dd(pnl_comb))*100:.2f}pp")
print("  NEXT: M57 -- Risk Parity & Factor Investing")
print()

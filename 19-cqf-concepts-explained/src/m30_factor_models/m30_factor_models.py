#!/usr/bin/env python3
"""
M30 — Factor Models: CAPM, Fama-French and APT
================================================
Module 30 | CQF Concepts Explained
Group 6   | Portfolio Construction

Theory
------
Capital Asset Pricing Model (Sharpe, 1964; Lintner, 1965)
----------------------------------------------------------
In equilibrium, every asset's expected excess return is proportional
to its covariance with the market portfolio:

    E[R_i] - rf = beta_i * (E[R_m] - rf)

    beta_i = Cov(R_i, R_m) / Var(R_m)

Assumptions: mean-variance investors, homogeneous expectations,
no taxes/transaction costs, single period, unlimited borrowing at rf.

Time-series regression (Jensen, 1968):
    R_i,t - rf = alpha_i + beta_i*(R_m,t - rf) + eps_i,t

alpha_i (Jensen's alpha) = abnormal return unexplained by CAPM.
Under CAPM, E[alpha_i] = 0 for all assets.

Security Market Line (SML): E[R_i] = rf + beta_i*(E[Rm]-rf)
Assets above SML are undervalued, below SML are overvalued.

Arbitrage Pricing Theory (Ross, 1976)
--------------------------------------
APT makes no assumptions about investor preferences. It only requires
that no-arbitrage holds in a factor structure:

    R_i = E[R_i] + sum_k beta_{ik} * F_k + eps_i

where F_k are zero-mean systematic factors, eps_i is idiosyncratic.
No-arbitrage implies:
    E[R_i] - rf = sum_k beta_{ik} * lambda_k

lambda_k = factor risk premium (price of risk for factor k).

Fama-French Three-Factor Model (1993)
--------------------------------------
    R_i - rf = alpha + b*(Rm-rf) + s*SMB + h*HML + eps

SMB (Small Minus Big): return spread small-cap vs large-cap stocks.
    Captures size premium: small stocks earn higher average returns.

HML (High Minus Low): return spread high book-to-market vs low.
    Captures value premium: value stocks earn more than growth stocks.

Empirical: R^2 improves from ~70% (CAPM) to ~90% (FF3).

Carhart Four-Factor Model (1997)
---------------------------------
    R_i - rf = alpha + b*(Rm-rf) + s*SMB + h*HML + m*MOM + eps

MOM: momentum factor (winners minus losers, past 12-2 months).
Captures momentum anomaly: past winners continue to outperform.

Factor Attribution
-------------------
Contribution of factor k to portfolio return:
    Contrib_k = beta_k * F_k * 100   (in return space)

Active return = alpha + sum_k (beta_k - beta_k^bench) * F_k

References
----------
- Sharpe, W.F. (1964). Capital asset prices. Journal of Finance, 19(3).
- Ross, S.A. (1976). The arbitrage theory of capital asset pricing.
  Journal of Economic Theory, 13(3), 341-360.
- Fama, E.F., French, K.R. (1993). Common risk factors in returns on
  stocks and bonds. Journal of Financial Economics, 33(1), 3-56.
- Carhart, M.M. (1997). On persistence in mutual fund performance.
  Journal of Finance, 52(1), 57-82.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a";  PANEL  = "#111111"; GRID   = "#1e1e1e"
WHITE  = "#e8e8e8";  BLUE   = "#4a9eff"; GREEN  = "#00d4aa"
ORANGE = "#ff8c42";  RED    = "#ff4757"; PURPLE = "#a855f7"
YELLOW = "#ffd700";  CYAN   = "#00bcd4"

WATERMARK = "Jose O. Bobadilla | CQF"
OUT_DIR   = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": DARK,   "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,   "axes.labelcolor":  WHITE,
    "axes.titlecolor":  WHITE,  "xtick.color":      WHITE,
    "ytick.color":      WHITE,  "text.color":       WHITE,
    "grid.color":       GRID,   "grid.linewidth":   0.6,
    "legend.facecolor": PANEL,  "legend.edgecolor": GRID,
    "font.family":      "monospace",
    "axes.spines.top":  False,  "axes.spines.right": False,
})

def watermark(ax):
    ax.text(0.99, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=WHITE, alpha=0.35, ha="right", va="bottom",
            fontstyle="italic")

# ============================================================
# SECTION 1 — SYNTHETIC FACTOR DATA
# ============================================================

def simulate_factor_returns(n_obs, seed=42):
    """
    Simulate monthly factor returns (annualized premia in parentheses):
      MKT-RF: ~6% premium, sigma~15%
      SMB:    ~2% premium, sigma~10%
      HML:    ~3% premium, sigma~10%
      MOM:    ~8% premium, sigma~15%
    All monthly: divide by 12.
    """
    rng = np.random.default_rng(seed)
    rf_monthly = 0.04 / 12

    # Factor correlation structure
    corr_f = np.array([
        [1.00, -0.20, -0.30,  0.00],
        [-0.20, 1.00,  0.10, -0.10],
        [-0.30, 0.10,  1.00, -0.15],
        [ 0.00,-0.10, -0.15,  1.00],
    ])
    vols_f = np.array([0.15, 0.10, 0.10, 0.15]) / np.sqrt(12)
    mus_f  = np.array([0.06, 0.02, 0.03, 0.08]) / 12

    L  = np.linalg.cholesky(corr_f)
    Z  = rng.standard_normal((n_obs, 4))
    F  = Z @ L.T * vols_f + mus_f
    rf = np.full(n_obs, rf_monthly)
    return F, rf   # columns: [MKT-RF, SMB, HML, MOM] excess returns

def simulate_asset_returns(F, rf, betas, alpha_ann, idio_vol_ann, seed=0):
    """
    Asset return: R_i - rf = alpha + B*F + eps
    alpha_ann: annualized alpha
    betas: (4,) array [b_mkt, s_smb, h_hml, m_mom]
    """
    rng = np.random.default_rng(seed)
    n   = len(F)
    alpha_m = alpha_ann / 12
    eps     = rng.normal(0, idio_vol_ann/np.sqrt(12), size=n)
    excess  = alpha_m + F @ betas + eps
    return excess + rf   # total return

N_OBS = 120   # 10 years monthly

F, rf = simulate_factor_returns(N_OBS)
MKT_RF = F[:, 0]

# Four synthetic assets (stocks A-D)
ASSETS = {
    "Stock A\n(market-like)":    {"betas": [1.00, 0.10, 0.05,  0.00], "alpha": 0.01,  "idio": 0.10},
    "Stock B\n(small-value)":    {"betas": [0.80, 0.60, 0.50,  0.10], "alpha": 0.02,  "idio": 0.15},
    "Stock C\n(growth/momentum)":{"betas": [1.20,-0.30,-0.40,  0.50], "alpha":-0.01,  "idio": 0.18},
    "Stock D\n(low-beta value)": {"betas": [0.50, 0.20, 0.70, -0.10], "alpha": 0.03,  "idio": 0.08},
}

returns = {}
for name, params in ASSETS.items():
    returns[name] = simulate_asset_returns(
        F, rf, np.array(params["betas"]),
        params["alpha"], params["idio"], seed=list(ASSETS.keys()).index(name)
    )

# ============================================================
# SECTION 2 — OLS FACTOR REGRESSIONS
# ============================================================

def ols_regression(y, X, add_const=True):
    """
    OLS: y = X*beta + eps.  Returns dict with betas, tstat, R2, alpha.
    """
    n = len(y)
    if add_const:
        X = np.column_stack([np.ones(n), X])
    k   = X.shape[1]
    b   = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b
    res  = y - yhat
    ss_res = res @ res
    ss_tot = ((y - y.mean())**2).sum()
    r2    = 1 - ss_res/ss_tot
    r2_adj = 1 - (1-r2)*(n-1)/(n-k)
    sigma2 = ss_res / (n - k)
    var_b  = sigma2 * np.linalg.inv(X.T @ X)
    se_b   = np.sqrt(np.diag(var_b))
    tstat  = b / se_b
    pval   = 2 * (1 - stats.t.cdf(np.abs(tstat), df=n-k))
    return {"betas": b, "se": se_b, "tstat": tstat, "pval": pval,
            "R2": r2, "R2_adj": r2_adj, "residuals": res, "fitted": yhat}

factor_names = ["alpha", "MKT-RF", "SMB", "HML", "MOM"]
results_capm = {}
results_ff3  = {}
results_ff4  = {}

print("[M30] Factor Regression Results")
print("=" * 70)
for name, ret in returns.items():
    excess = ret - rf
    rc = ols_regression(excess, MKT_RF.reshape(-1,1), add_const=True)
    r3 = ols_regression(excess, F[:, :3], add_const=True)
    r4 = ols_regression(excess, F,        add_const=True)
    results_capm[name] = rc
    results_ff3[name]  = r3
    results_ff4[name]  = r4
    lbl = name.replace("\n", " ")
    print(f"\n  {lbl}")
    print(f"    CAPM: alpha={rc['betas'][0]*12*100:.2f}%/yr  "
          f"beta={rc['betas'][1]:.3f}  R2={rc['R2']*100:.1f}%")
    print(f"    FF3:  alpha={r3['betas'][0]*12*100:.2f}%/yr  "
          f"b={r3['betas'][1]:.3f}  s={r3['betas'][2]:.3f}  "
          f"h={r3['betas'][3]:.3f}  R2={r3['R2']*100:.1f}%")
    print(f"    FF4:  alpha={r4['betas'][0]*12*100:.2f}%/yr  "
          f"b={r4['betas'][1]:.3f}  s={r4['betas'][2]:.3f}  "
          f"h={r4['betas'][3]:.3f}  m={r4['betas'][4]:.3f}  "
          f"R2={r4['R2']*100:.1f}%")

# ============================================================
# FIGURE 1 — CAPM: SML and beta estimation
# ============================================================
t0 = time.perf_counter()
print("\n[M30] Figure 1: CAPM — Security Market Line ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M30 — CAPM: Security Market Line\n"
             "E[Ri]-rf = beta_i*(E[Rm]-rf)  |  Jensen's alpha",
             color=WHITE, fontsize=12, fontweight="bold")

colors4 = [BLUE, GREEN, ORANGE, PURPLE]
names4  = list(ASSETS.keys())
rf_ann  = 0.04

# (a) Security Market Line
ax = axes[0]
beta_arr = np.linspace(-0.2, 1.6, 200)
mkt_prem = MKT_RF.mean() * 12
sml      = rf_ann + beta_arr * mkt_prem
ax.plot(beta_arr, sml*100, color=WHITE, lw=2, label="SML", zorder=2)
for (name, res), col in zip(results_capm.items(), colors4):
    beta_i  = res["betas"][1]
    alpha_i = res["betas"][0] * 12   # annualized
    ret_i   = returns[name].mean() * 12
    excess_i = ret_i - rf_ann
    sml_pred = rf_ann + beta_i * mkt_prem
    ax.scatter(beta_i, excess_i*100, color=col, s=120, zorder=5,
               label=name.replace("\n"," "))
    ax.annotate(f"alpha={alpha_i*100:+.1f}%",
                (beta_i, excess_i*100),
                textcoords="offset points", xytext=(6, -3),
                fontsize=7, color=col)
    # Arrow from SML to actual return
    ax.annotate("", xy=(beta_i, excess_i*100),
                xytext=(beta_i, sml_pred*100),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
ax.set_xlabel("Beta"); ax.set_ylabel("Expected Excess Return (%/yr)")
ax.set_title("Security Market Line\nArrows = Jensen's alpha (distance from SML)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) CAPM scatter regression for Stock B
name_b  = names4[1]
excess_b = returns[name_b] - rf
ax = axes[1]
ax.scatter(MKT_RF*100, excess_b*100, s=8, alpha=0.5, color=GREEN,
           label="Monthly observations")
x_line = np.linspace(MKT_RF.min(), MKT_RF.max(), 100)
b_capm  = results_capm[name_b]["betas"]
ax.plot(x_line*100,
        (b_capm[0] + b_capm[1]*x_line)*100,
        color=ORANGE, lw=2.5,
        label=f"OLS: alpha={b_capm[0]*12*100:.2f}%/yr  "
              f"beta={b_capm[1]:.3f}")
ax.set_xlabel("MKT-RF (%)"); ax.set_ylabel("R_i - rf (%)")
ax.set_title(f"CAPM Regression — {name_b.replace(chr(10),' ')}\n"
             f"R2={results_capm[name_b]['R2']*100:.1f}%",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) R2 comparison: CAPM vs FF3 vs FF4
ax = axes[2]
r2_capm = [results_capm[n]["R2"]*100 for n in names4]
r2_ff3  = [results_ff3[n]["R2"]*100  for n in names4]
r2_ff4  = [results_ff4[n]["R2"]*100  for n in names4]
x  = np.arange(len(names4))
bw = 0.25
ax.bar(x - bw, r2_capm, width=bw, color=BLUE,   alpha=0.85, label="CAPM")
ax.bar(x,      r2_ff3,  width=bw, color=GREEN,  alpha=0.85, label="FF3")
ax.bar(x + bw, r2_ff4,  width=bw, color=ORANGE, alpha=0.85, label="FF4 (Carhart)")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n"," ")[:12] for n in names4],
                   fontsize=7, rotation=10)
ax.set_ylabel("R-squared (%)")
ax.set_title("R² by Model\nMore factors => higher explanatory power",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m30_01_capm_sml.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Fama-French factor betas and premia
# ============================================================
t0 = time.perf_counter()
print("[M30] Figure 2: Fama-French factor loadings and premia ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M30 — Fama-French Four-Factor Model\n"
             "R_i - rf = alpha + b*MKT + s*SMB + h*HML + m*MOM + eps",
             color=WHITE, fontsize=12, fontweight="bold")

fn = ["MKT-RF", "SMB", "HML", "MOM"]

# (a) Factor betas heatmap
ax = axes[0]
beta_matrix = np.array([results_ff4[n]["betas"][1:] for n in names4])
im = ax.imshow(beta_matrix, cmap="RdBu_r", aspect="auto",
               vmin=-1.0, vmax=1.0)
plt.colorbar(im, ax=ax, label="Factor loading (beta)")
ax.set_xticks(range(4)); ax.set_xticklabels(fn, fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels([n.replace("\n"," ")[:15] for n in names4], fontsize=7)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{beta_matrix[i,j]:.2f}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
ax.set_title("Factor Beta Heatmap (FF4)\nBlue=negative, Red=positive",
             color=WHITE, fontsize=9)
watermark(ax)

# (b) Factor return cumulative performance
ax = axes[1]
factor_labels = ["MKT-RF", "SMB", "HML", "MOM"]
factor_cols   = [BLUE, GREEN, ORANGE, PURPLE]
for k, (lbl, col) in enumerate(zip(factor_labels, factor_cols)):
    cum = np.cumprod(1 + F[:, k]) - 1
    ax.plot(cum*100, color=col, lw=2, label=f"{lbl}: {F[:,k].mean()*12*100:.1f}%/yr")
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Month"); ax.set_ylabel("Cumulative return (%)")
ax.set_title("Cumulative Factor Returns\n10-year simulation",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Return attribution (FF4) for each stock
ax = axes[2]
x  = np.arange(len(names4))
bw = 0.15
factor_contribs = []
for n in names4:
    b  = results_ff4[n]["betas"]
    fmeans = F.mean(axis=0) * 12 * 100   # annualized %
    contribs = b[1:] * fmeans            # beta_k * E[F_k]
    factor_contribs.append([b[0]*12*100] + list(contribs))

factor_contribs = np.array(factor_contribs)  # (4 assets, 5: alpha+4 factors)
fc_labels = ["Alpha"] + fn
fc_cols   = [YELLOW, BLUE, GREEN, ORANGE, PURPLE]
bottoms   = np.zeros(len(names4))
for k, (lbl, col) in enumerate(zip(fc_labels, fc_cols)):
    vals = factor_contribs[:, k]
    ax.bar(x, vals, bottom=bottoms, width=0.5, color=col, alpha=0.85, label=lbl)
    bottoms += vals
ax.axhline(0, color=WHITE, lw=0.8, linestyle=":")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n"," ")[:12] for n in names4],
                   fontsize=7, rotation=10)
ax.set_ylabel("Return attribution (%/yr)")
ax.set_title("Return Attribution by Factor (FF4)\n"
             "Stacked: alpha + 4 factor contributions",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m30_02_fama_french.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — APT: factor premia and cross-sectional pricing
# ============================================================
t0 = time.perf_counter()
print("[M30] Figure 3: APT cross-sectional pricing and residual analysis ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M30 — Arbitrage Pricing Theory (APT)\n"
             "E[Ri]-rf = sum_k beta_ik * lambda_k  (factor risk premia)",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Cross-sectional regression: actual vs model-predicted excess returns
ax = axes[0]
lambda_k = F.mean(axis=0) * 12   # annualized factor premia
for (name, res), col in zip(results_ff4.items(), colors4):
    b_k      = res["betas"][1:]
    pred_exc = b_k @ lambda_k
    act_exc  = (returns[name].mean() - rf.mean()) * 12
    ax.scatter(pred_exc*100, act_exc*100, color=col, s=150, zorder=5,
               label=name.replace("\n"," "))
    ax.annotate(name.split("\n")[0], (pred_exc*100, act_exc*100),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color=col)
x_45 = np.linspace(-2, 14, 100)
ax.plot(x_45, x_45, color=WHITE, lw=1.5, linestyle="--",
        label="45° (perfect pricing)")
ax.set_xlabel("APT predicted excess return (%/yr)")
ax.set_ylabel("Actual excess return (%/yr)")
ax.set_title("APT Cross-Sectional Pricing\n"
             "Points on 45° => no alpha (APT holds)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) Residual diagnostics: Stock B FF4 residuals
res_b = results_ff4[name_b]["residuals"]
ax = axes[1]
ax.hist(res_b*100, bins=25, density=True, color=GREEN, alpha=0.6,
        label="FF4 residuals")
x_norm = np.linspace(res_b.min()*100, res_b.max()*100, 200)
ax.plot(x_norm,
        stats.norm.pdf(x_norm, res_b.mean()*100, res_b.std()*100),
        color=ORANGE, lw=2.5, label="Normal fit")
sk  = stats.skew(res_b)
ku  = stats.kurtosis(res_b)
_, jb_p = stats.jarque_bera(res_b)
ax.set_xlabel("Residual (%)"); ax.set_ylabel("Density")
ax.set_title(f"FF4 Residual Distribution — {name_b.split(chr(10))[0]}\n"
             f"Skew={sk:.3f}  Excess Kurt={ku:.3f}  JB p={jb_p:.3f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Rolling beta (MKT) for Stock C (momentum/growth)
name_c   = names4[2]
excess_c = returns[name_c] - rf
window   = 24   # 24-month rolling window
betas_roll = []
times_roll = []
for i in range(window, N_OBS+1):
    y_w = excess_c[i-window:i]
    x_w = MKT_RF[i-window:i]
    b_, _, _, _, _ = stats.linregress(x_w, y_w)
    betas_roll.append(b_)
    times_roll.append(i)

ax = axes[2]
ax.plot(times_roll, betas_roll, color=ORANGE, lw=2.5,
        label="Rolling 24M beta (MKT)")
full_beta = results_capm[name_c]["betas"][1]
ax.axhline(full_beta, color=WHITE, lw=1.5, linestyle="--",
           label=f"Full-sample beta = {full_beta:.3f}")
ax.fill_between(times_roll, betas_roll, full_beta,
                color=ORANGE, alpha=0.15)
ax.set_xlabel("Month"); ax.set_ylabel("Rolling beta")
ax.set_title(f"Rolling Market Beta — {name_c.replace(chr(10),' ')}\n"
             "24-month window: beta instability over time",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m30_03_apt_cross_section.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 30 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] CAPM: E[Ri]-rf = beta*(E[Rm]-rf)  one-factor model")
print("  [2] Jensen alpha: abnormal return unexplained by beta")
print("  [3] APT: E[Ri]-rf = sum_k beta_ik*lambda_k  (no-arb)")
print("  [4] FF3: market+SMB+value  R2 ~90% vs CAPM ~70%")
print("  [5] MOM (Carhart): past winners continue outperforming")
print("  [6] Rolling beta: market beta unstable over time")
ff4_r2 = [results_ff4[n]["R2"]*100 for n in names4]
capm_r2 = [results_capm[n]["R2"]*100 for n in names4]
print(f"  Average R2 CAPM: {np.mean(capm_r2):.1f}%  |  "
      f"FF4: {np.mean(ff4_r2):.1f}%")
for n in names4:
    a4 = results_ff4[n]["betas"][0]*12*100
    print(f"  {n.replace(chr(10),' ')[:25]:25s}: "
          f"FF4 alpha={a4:+.2f}%/yr  R2={results_ff4[n]['R2']*100:.1f}%")
print("=" * 65)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 43 -- AUTOENCODERS: ANOMALY DETECTION & VOL SURFACE DENOISING
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
An autoencoder is a neural network trained to reproduce its own input through
a compressed intermediate representation called the *bottleneck* or *latent
code*.  Formally, given input x in R^d:

    Encoder :  h  = f_enc(x; theta_e)    h in R^k,  k << d
    Decoder :  x~ = f_dec(h; theta_d)    x~ in R^d

The model is trained by minimising reconstruction loss:

    L(theta) = (1/N) * sum_{i=1}^N || x_i - x~_i ||^2

Key insight: if x is *normal*, the autoencoder learns to reconstruct it well.
Anomalies -- structurally different from the training distribution -- produce
high reconstruction error L(x), which serves as an unsupervised anomaly score.

FINANCIAL APPLICATIONS
----------------------
1. Return anomaly detection: crash days, flash crashes, regime breaks
2. Volatility surface denoising: compress and reconstruct noisy implied vols
3. Latent factor extraction: compress cross-section of returns to common drivers
4. Credit portfolio compression: encode correlated default intensities

PCA AS LINEAR AUTOENCODER
--------------------------
PCA is the optimal *linear* autoencoder.  When the encoder and decoder are
restricted to linear maps (W, W^T), the minimum MSE solution recovers the
top-k principal components.  Nonlinear autoencoders can capture richer
structure at the cost of interpretability.

IMPLEMENTATION STRATEGY
-----------------------
This module implements autoencoders from scratch using NumPy, following the
same philosophy as M41 and M42.  No deep learning framework is required.
Architecture:  d -> H1 -> k -> H1 -> d  (symmetric, tied or untied weights)
Activation:    ReLU in hidden layers, linear output (regression-type loss)
Optimiser:     mini-batch SGD with momentum
Regularisation: optional L2 weight decay to prevent overfitting

MATHEMATICAL PREREQUISITES
---------------------------
Module 36 -- Factor Engineering (cross-sectional features)
Module 41 -- MLP / Feedforward Neural Network (backpropagation)
Module 42 -- RNN/LSTM (sequence features, regime detection)

REFERENCES
----------
[1] Baldi, P. & Hornik, K. (1989). "Neural networks and principal component
    analysis." Neural Networks 2(1):53-58.
[2] Vincent, P. et al. (2010). "Stacked Denoising Autoencoders." JMLR 11.
[3] Goodfellow, I., Bengio, Y. & Courville, A. (2016). Deep Learning. MIT Press.
[4] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m43_autoencoders/m43_autoencoders.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m43")
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

plt.rcParams.update({
    "figure.facecolor" : DARK,  "axes.facecolor"   : PANEL,
    "axes.edgecolor"   : GRID,  "axes.labelcolor"  : TEXT,
    "axes.titlecolor"  : TEXT,  "xtick.color"      : TEXT,
    "ytick.color"      : TEXT,  "text.color"       : TEXT,
    "grid.color"       : GRID,  "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,   "legend.facecolor" : PANEL,
    "legend.edgecolor" : GRID,  "font.family"      : "monospace",
    "font.size"        : 9,     "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
})

def hdr(msg):  print(f"  {msg}")
def section(n, msg): print(f"  [{n:02d}] {msg}")

# =============================================================================
# 1.  DATA
# =============================================================================
print()
print("=" * 65)
print("  MODULE 43: AUTOENCODERS")
print("  Anomaly Detection | Vol Surface | Latent Space | Denoising")
print("=" * 65)

raw   = yf.download("SPY", start="2018-01-01", end="2023-12-31",
                    auto_adjust=True, progress=False)
close = raw["Close"].squeeze().dropna()
ret   = np.log(close / close.shift(1)).dropna()
N     = len(ret)
label_bear = (ret < -2 * ret.std()).astype(int).values

section(1, f"Return series: {N} days  extreme-neg days: {label_bear.sum()}")

# =============================================================================
# 2.  ROLLING WINDOW FEATURE MATRIX
# =============================================================================
L     = 30
X_raw = np.array([ret.values[i:i+L] for i in range(N - L)])
y_lbl = label_bear[L:]

scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)
n, d   = X.shape

section(2, f"Feature matrix: {n} x {d}  (L={L} window)")

# =============================================================================
# 3.  PCA BASELINE (linear autoencoder)
# =============================================================================
K_pca = 5
pca   = PCA(n_components=K_pca)
Z_pca = pca.fit_transform(X)
X_pca_rec   = pca.inverse_transform(Z_pca)
mse_pca     = np.mean((X - X_pca_rec) ** 2, axis=1)
var_explained = pca.explained_variance_ratio_.cumsum()

section(3, f"PCA k={K_pca}: cumulative variance = {var_explained[-1]:.3f}")

# =============================================================================
# 4.  NUMPY AUTOENCODER
# =============================================================================
class DenseAutoencoder:
    """
    Symmetric fully-connected autoencoder implemented in NumPy.

    Architecture (encoder):
        Input (d) --[W1,b1]--> H1 (h_dim) --ReLU--> [W2,b2]--> Latent (k)

    Architecture (decoder):
        Latent (k) --[W3,b3]--> H1 (h_dim) --ReLU--> [W4,b4]--> Output (d)

    Loss:   MSE = (1/N) ||X - X_rec||^2_F
    Optimiser: mini-batch SGD with momentum
    """

    def __init__(self, d, h_dim, k, lr=1e-3, l2=1e-4):
        self.d = d; self.h_dim = h_dim; self.k = k
        self.lr = lr; self.l2 = l2
        s1 = np.sqrt(2.0 / d);      s2 = np.sqrt(2.0 / h_dim)
        self.W1 = np.random.randn(d,     h_dim) * s1
        self.b1 = np.zeros(h_dim)
        self.W2 = np.random.randn(h_dim, k)     * s2
        self.b2 = np.zeros(k)
        self.W3 = np.random.randn(k,     h_dim) * s2
        self.b3 = np.zeros(h_dim)
        self.W4 = np.random.randn(h_dim, d)     * s1
        self.b4 = np.zeros(d)
        for nm in ["W1","b1","W2","b2","W3","b3","W4","b4"]:
            setattr(self, "v"+nm, np.zeros_like(getattr(self, nm)))

    @staticmethod
    def relu(z):  return np.maximum(0.0, z)
    @staticmethod
    def drelu(z): return (z > 0.0).astype(float)

    def forward(self, X):
        self.z1 = X  @ self.W1 + self.b1;  self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.z3 = self.z2 @ self.W3 + self.b3; self.a3 = self.relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        return self.z4

    def encode(self, X):
        a1 = self.relu(X @ self.W1 + self.b1)
        return a1 @ self.W2 + self.b2

    def backward(self, X, X_rec, momentum=0.9):
        m    = X.shape[0]
        dz4  = (2.0 / m) * (X_rec - X)
        dW4  = self.a3.T @ dz4 + self.l2 * self.W4;  db4 = dz4.sum(0)
        da3  = dz4 @ self.W4.T;  dz3 = da3 * self.drelu(self.z3)
        dW3  = self.z2.T @ dz3 + self.l2 * self.W3;  db3 = dz3.sum(0)
        dz2  = dz3 @ self.W3.T
        dW2  = self.a1.T @ dz2 + self.l2 * self.W2;  db2 = dz2.sum(0)
        da1  = dz2 @ self.W2.T;  dz1 = da1 * self.drelu(self.z1)
        dW1  = X.T @ dz1 + self.l2 * self.W1;        db1 = dz1.sum(0)
        mu   = momentum
        for nm, g in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2),
                      ("W3",dW3),("b3",db3),("W4",dW4),("b4",db4)]:
            v = getattr(self, "v"+nm)
            v[:] = mu * v - self.lr * g
            getattr(self, nm)[:] += v

    def fit(self, X, epochs=200, batch=64, verbose_every=50):
        losses = []
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(X.shape[0]);  ep_loss = 0.0
            for i in range(0, X.shape[0], batch):
                Xb = X[idx[i:i+batch]];  Xr = self.forward(Xb)
                loss = np.mean((Xb - Xr) ** 2)
                ep_loss += loss * len(Xb);  self.backward(Xb, Xr)
            ep_loss /= X.shape[0];  losses.append(ep_loss)
            if ep % verbose_every == 0:
                hdr(f"       epoch {ep:4d}/{epochs}  MSE={ep_loss:.6f}")
        return losses

    def reconstruction_error(self, X):
        return np.mean((X - self.forward(X)) ** 2, axis=1)


hdr("  Training autoencoder (d=30, h=64, k=5) ...")
ae = DenseAutoencoder(d=d, h_dim=64, k=K_pca, lr=5e-4, l2=1e-4)
loss_curve = ae.fit(X, epochs=300, batch=64, verbose_every=100)
mse_ae = ae.reconstruction_error(X)
Z_ae   = ae.encode(X)

section(4, f"AE trained: final MSE={loss_curve[-1]:.6f}")

# =============================================================================
# 5.  ANOMALY DETECTION
# =============================================================================
auc_ae  = roc_auc_score(y_lbl, mse_ae)  if y_lbl.sum() > 0 else 0.5
auc_pca = roc_auc_score(y_lbl, mse_pca) if y_lbl.sum() > 0 else 0.5
p       = 5.0
thr_ae  = np.percentile(mse_ae, 100 - p)

section(5, f"Anomaly AUC  AE={auc_ae:.3f}  PCA={auc_pca:.3f}  flagged(top {p:.0f}%)={(mse_ae>thr_ae).sum()}")

# =============================================================================
# 6.  VOL SURFACE DENOISING
# =============================================================================
def make_vol_surface(n_K=20, n_T=10, seed=0):
    rng     = np.random.default_rng(seed)
    K_grid  = np.linspace(-0.4, 0.4, n_K)
    T_grid  = np.linspace(0.08, 2.0, n_T)
    vol_clean = np.zeros((n_K, n_T))
    for j, T in enumerate(T_grid):
        atm  = 0.18 + 0.04 * np.exp(-T)
        skew = -0.25 * np.exp(-0.5 * T)
        curv =  0.12 * np.exp(-0.3 * T)
        vol_clean[:, j] = atm + skew * K_grid + curv * K_grid ** 2
    vol_noisy = vol_clean + rng.normal(0, 0.008, vol_clean.shape)
    return K_grid, T_grid, vol_clean, vol_noisy

n_K, n_T = 20, 10
K_grid, T_grid, vol_clean, _ = make_vol_surface(n_K, n_T)

rng_s  = np.random.default_rng(99)
X_surf = np.array([
    (vol_clean + rng_s.normal(0, 0.008, vol_clean.shape)).ravel()
    for _ in range(500)
])
sc_surf  = StandardScaler()
X_surf_s = sc_surf.fit_transform(X_surf)

ae_surf = DenseAutoencoder(d=n_K*n_T, h_dim=40, k=4, lr=1e-3, l2=1e-4)
ae_surf.fit(X_surf_s, epochs=400, batch=32, verbose_every=200)

_, _, _, vol_noisy_ex = make_vol_surface(n_K, n_T, seed=7)
x_noisy_s = sc_surf.transform(vol_noisy_ex.ravel().reshape(1, -1))
vol_rec   = sc_surf.inverse_transform(ae_surf.forward(x_noisy_s)).reshape(n_K, n_T)

mse_noisy = np.mean((vol_noisy_ex - vol_clean) ** 2)
mse_rec   = np.mean((vol_rec      - vol_clean) ** 2)

section(6, f"Vol denoising  MSE(noisy)={mse_noisy:.6f}  MSE(AE)={mse_rec:.6f}  "
           f"improvement={100*(1-mse_rec/mse_noisy):.1f}%")

# =============================================================================
# 7.  LATENT SPACE
# =============================================================================
pca2 = PCA(n_components=2)
Z2   = pca2.fit_transform(Z_ae)
rv   = np.array([ret.values[L+i-20:L+i].std() for i in range(n)])
rv_norm = (rv - rv.min()) / (rv.max() - rv.min() + 1e-9)

section(7, f"Latent 2D PCA var_expl={pca2.explained_variance_ratio_.sum():.3f}")

# =============================================================================
# 8.  DENOISING AUTOENCODER (DAE)
# =============================================================================
def add_noise(X, sigma=0.15):
    return X + np.random.randn(*X.shape) * sigma

hdr("  Training denoising autoencoder (sigma=0.15) ...")
ae_dae = DenseAutoencoder(d=d, h_dim=64, k=K_pca, lr=5e-4, l2=1e-4)
dae_losses = []
for ep in range(1, 301):
    idx = np.random.permutation(n);  ep_loss = 0.0
    for i in range(0, n, 64):
        Xb_clean = X[idx[i:i+64]];  Xb_noisy = add_noise(Xb_clean)
        Xr = ae_dae.forward(Xb_noisy)
        ep_loss += np.mean((Xb_clean - Xr) ** 2) * len(Xb_clean)
        ae_dae.backward(Xb_noisy, Xr)
    dae_losses.append(ep_loss / n)

mse_dae = ae_dae.reconstruction_error(X)
auc_dae = roc_auc_score(y_lbl, mse_dae) if y_lbl.sum() > 0 else 0.5

section(8, f"DAE MSE={dae_losses[-1]:.6f}  AUC={auc_dae:.3f}")

# =============================================================================
# FIGURE 1: TRAINING + ANOMALY SCORES + LATENT SPACE
# =============================================================================
fig = plt.figure(figsize=(16, 10), facecolor=DARK)
fig.suptitle("Module 43 -- Autoencoder: Training, Anomaly Scores & Latent Space",
             fontsize=12, color=TEXT, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.42)

ax = fig.add_subplot(gs[0, 0])
ep_x = np.arange(1, len(loss_curve) + 1)
ax.plot(ep_x, loss_curve,  color=ACCENT, lw=1.5, label="AE")
ax.plot(ep_x, dae_losses,  color=AMBER,  lw=1.5, label="DAE")
ax.set_title("Training Loss (MSE)"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(fontsize=8); ax.grid(True)

ax = fig.add_subplot(gs[0, 1])
t_axis = np.arange(n)
ax.plot(t_axis, mse_ae,  color=ACCENT, lw=0.7, alpha=0.8, label="AE score")
ax.plot(t_axis, mse_pca, color=GREEN,  lw=0.7, alpha=0.8, label="PCA score")
flag_idx = np.where(mse_ae > thr_ae)[0]
ax.scatter(flag_idx, mse_ae[flag_idx], color=RED, s=10, zorder=5,
           label=f"Flagged top {p:.0f}%")
ax.axhline(thr_ae, color=RED, lw=0.8, ls="--", alpha=0.7)
ax.set_title("Reconstruction Error Over Time")
ax.set_xlabel("Day index"); ax.set_ylabel("MSE")
ax.legend(fontsize=7); ax.grid(True)

ax = fig.add_subplot(gs[0, 2])
bins = 60
ax.hist(mse_pca, bins=bins, alpha=0.5, color=GREEN,  density=True, label="PCA")
ax.hist(mse_ae,  bins=bins, alpha=0.6, color=ACCENT, density=True, label="AE")
ax.hist(mse_dae, bins=bins, alpha=0.5, color=AMBER,  density=True, label="DAE")
ax.axvline(thr_ae, color=RED, lw=1.2, ls="--", label=f"AE thr p{100-p:.0f}")
ax.set_title("Reconstruction Error Distribution")
ax.set_xlabel("MSE"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

ax = fig.add_subplot(gs[1, 0])
sc_ = ax.scatter(Z2[:, 0], Z2[:, 1], c=rv_norm, cmap="plasma", s=4, alpha=0.6)
cb  = fig.colorbar(sc_, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Norm. realised vol", color=TEXT, fontsize=7)
cb.ax.yaxis.set_tick_params(color=TEXT)
ax.set_title("Latent Space (AE k=5, PCA to 2D)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True)

ax = fig.add_subplot(gs[1, 1])
methods = ["PCA", "AE", "DAE"]
aucs    = [auc_pca, auc_ae, auc_dae]
colors  = [GREEN, ACCENT, AMBER]
bars    = ax.bar(methods, aucs, color=colors, edgecolor=DARK, width=0.5)
ax.axhline(0.5, color=RED, lw=1.0, ls="--", label="Random")
for bar, v in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8, color=TEXT)
ax.set_ylim(0.4, max(aucs) * 1.12)
ax.set_title(f"Anomaly Detection AUC (top {p:.0f}% flagged)")
ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=7); ax.grid(True, axis="y")

ax = fig.add_subplot(gs[1, 2])
cumvar = np.cumsum(pca.explained_variance_ratio_)
ks     = np.arange(1, len(cumvar) + 1)
ax.bar(ks, pca.explained_variance_ratio_ * 100, color=VIOLET, alpha=0.8, edgecolor=DARK)
ax.step(ks, cumvar * 100, color=ACCENT, lw=1.5, where="mid", label="Cumulative")
ax.axhline(80, color=AMBER, lw=0.9, ls="--", alpha=0.8, label="80% threshold")
for k, v in zip(ks, cumvar * 100):
    ax.text(k, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_xlabel("PC"); ax.set_ylabel("Variance (%)"); ax.set_title("PCA Explained Variance")
ax.legend(fontsize=7); ax.grid(True, axis="y")

fig.savefig(os.path.join(FIGS, "m43_fig1_training_anomaly_latent.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: VOL SURFACE DENOISING
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("Module 43 -- Volatility Surface Denoising via Autoencoder",
             fontsize=12, color=TEXT, y=1.01)

TT, KK = np.meshgrid(T_grid, K_grid)
vmin = vol_clean.min() * 0.95
vmax = vol_clean.max() * 1.05

for ax, (surf, title) in zip(axes, [
    (vol_clean,    "Clean Surface"),
    (vol_noisy_ex, f"Noisy Observation  MSE={mse_noisy:.5f}"),
    (vol_rec,      f"AE Reconstruction  MSE={mse_rec:.5f}"),
]):
    ax.set_facecolor(PANEL)
    cf = ax.contourf(TT, KK, surf, levels=20, cmap="plasma", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Implied Vol", color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Log-moneyness")
    ax.set_title(title); ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m43_fig2_vol_surface_denoising.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: ARCHITECTURE DIAGRAM + BOTTLENECK HEATMAP
# =============================================================================
n_params = d*64+64 + 64*K_pca+K_pca + K_pca*64+64 + 64*d+d

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle("Module 43 -- AE Architecture & Bottleneck Activations",
             fontsize=12, color=TEXT, y=1.01)

ax = axes[0]
ax.set_facecolor(PANEL); ax.axis("off")
arch_text = (
    "AUTOENCODER ARCHITECTURE\n"
    "========================\n\n"
    "  Input layer  :  x  in  R^{d}        d = {d}\n"
    "       |\n"
    "  [W1, b1]  {d} x 64   He init\n"
    "       |  ReLU activation\n"
    "  Hidden H1  :  a1  in  R^64\n"
    "       |\n"
    "  [W2, b2]  64 x {k}   linear\n"
    "       |\n"
    "  BOTTLENECK  :  z  in  R^{k}   << latent code\n"
    "       |\n"
    "  [W3, b3]  {k} x 64\n"
    "       |  ReLU activation\n"
    "  Hidden H1' :  a3  in  R^64\n"
    "       |\n"
    "  [W4, b4]  64 x {d}   linear\n"
    "       |\n"
    "  Output  :  x~  in  R^{d}\n\n"
    "  Loss     :  L = (1/N)||X - X~||^2_F\n"
    "  Compress :  {d}/{k} = {ratio}x\n"
    "  Params   :  {p:,}"
).format(d=d, k=K_pca, ratio=d//K_pca, p=n_params)

ax.text(0.05, 0.95, arch_text, transform=ax.transAxes,
        fontsize=8.5, va="top", fontfamily="monospace",
        color=TEXT, linespacing=1.7)
ax.set_title("Architecture Summary")

ax = axes[1]
Z_sample = Z_ae[:200]
im = ax.imshow(Z_sample.T, aspect="auto", cmap="RdBu_r",
               vmin=-3, vmax=3, interpolation="nearest")
ax.set_xlabel("Sample index (day)"); ax.set_ylabel("Latent dimension")
ax.set_yticks(range(K_pca))
ax.set_yticklabels([f"z{i+1}" for i in range(K_pca)], fontsize=8)
ax.set_title("Bottleneck Activations: Z[0:200]")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Activation", color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m43_fig3_architecture_bottleneck.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("  MODULE 43 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] AE: encoder f_enc(x)->z, decoder f_dec(z)->x~, loss MSE")
print("  [2] Bottleneck k<<d forces compact latent representation")
print("  [3] PCA = optimal linear AE (Baldi & Hornik, 1989)")
print(f"  [4] PCA k={K_pca} explains {var_explained[-1]*100:.1f}% variance")
print(f"  [5] Reconstruction error as anomaly score  AE AUC={auc_ae:.3f}")
print(f"  [6] DAE: noisy input, clean target, robust repr.  AUC={auc_dae:.3f}")
print(f"  [7] Vol surface denoising improvement={100*(1-mse_rec/mse_noisy):.1f}%")
print(f"  [8] Latent space encodes volatility regimes")
print(f"  AE  reconstruction MSE : {loss_curve[-1]:.6f}")
print(f"  DAE reconstruction MSE : {dae_losses[-1]:.6f}")
print(f"  NEXT: M44 -- NLP Sentiment Analysis of Financial Headlines")
print()

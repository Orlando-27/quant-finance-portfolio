"""
Publication-quality visualizations for Sentiment Trading Signals.

Figures generated:
    01_sentiment_timeseries.png    - Aggregate sentiment score over time
    02_sentiment_vs_returns.png    - Scatter: sentiment vs forward returns
    03_word_importance.png         - Top positive/negative financial terms
    04_signal_backtest.png         - Sentiment strategy equity curve
    05_confusion_matrix.png        - Signal classification performance
    06_regime_analysis.png         - Sentiment regimes & market states

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

NAVY = "#1a1a2e"; TEAL = "#16697a"; CORAL = "#db6400"
GOLD = "#c5a880"; SLATE = "#4a4e69"
COLORS = [NAVY, TEAL, CORAL, GOLD, SLATE, "#2d6a4f", "#e07a5f"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "savefig.facecolor": "white",
})

def _wm(fig):
    fig.text(0.99, 0.01, "J. Bobadilla | CQF", fontsize=7,
             color="gray", alpha=0.5, ha="right", va="bottom")

def _sv(fig, proj, name):
    path = os.path.join(proj, "outputs", "figures", name)
    fig.savefig(path); plt.close(fig); return path


def plot_sentiment_timeseries(proj=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    np.random.seed(42)
    days = 500
    t = np.arange(days)
    # Simulated sentiment score (-1 to 1)
    sentiment = np.cumsum(0.01 * np.random.randn(days))
    sentiment = np.tanh(sentiment)  # bound to (-1, 1)
    # Simulated price
    price = 100 * np.cumprod(1 + 0.0003 + 0.005 * sentiment + 0.01 * np.random.randn(days))

    ax1.plot(t, price, color=NAVY, lw=1.5)
    ax1.set_ylabel("Price ($)"); ax1.set_title("Asset Price")

    ax2.fill_between(t, sentiment, where=sentiment > 0, alpha=0.4, color=TEAL, label="Bullish")
    ax2.fill_between(t, sentiment, where=sentiment < 0, alpha=0.4, color=CORAL, label="Bearish")
    ax2.plot(t, sentiment, color=NAVY, lw=1)
    ax2.axhline(0, color="gray", ls="--")
    ax2.set_xlabel("Trading Days"); ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Aggregate Sentiment Index")
    ax2.legend()
    fig.suptitle("Sentiment-Price Co-Movement", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "01_sentiment_timeseries.png")


def plot_sentiment_vs_returns(proj=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    np.random.seed(42)
    n = 300
    sentiment = np.random.uniform(-1, 1, n)
    fwd_ret = 0.002 * sentiment + 0.008 * np.random.randn(n)
    colors = [TEAL if s > 0 else CORAL for s in sentiment]
    ax.scatter(sentiment, fwd_ret * 100, c=colors, alpha=0.5, s=30, edgecolor="white", lw=0.5)
    # Regression line
    z = np.polyfit(sentiment, fwd_ret * 100, 1)
    x_line = np.linspace(-1, 1, 100)
    ax.plot(x_line, np.polyval(z, x_line), color=NAVY, lw=2.5,
            label=f"Slope = {z[0]:.3f}% per unit")
    ax.axhline(0, color="gray", ls=":")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("Sentiment Score"); ax.set_ylabel("5-Day Forward Return (%)")
    ax.set_title("Sentiment Score vs Forward Returns")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "02_sentiment_vs_returns.png")


def plot_word_importance(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    pos_words = ["growth", "profit", "upgrade", "dividend", "bullish",
                 "revenue", "outperform", "expansion", "strong", "opportunity"]
    pos_scores = [0.89, 0.85, 0.82, 0.78, 0.76, 0.72, 0.69, 0.65, 0.62, 0.58]
    neg_words = ["loss", "downgrade", "bankruptcy", "default", "recession",
                 "decline", "volatile", "risk", "underperform", "liability"]
    neg_scores = [-0.91, -0.87, -0.84, -0.80, -0.77, -0.73, -0.70, -0.66, -0.63, -0.59]

    ax1.barh(pos_words[::-1], pos_scores[::-1], color=TEAL, edgecolor="white")
    ax1.set_xlabel("Sentiment Weight"); ax1.set_title("Top Positive Financial Terms")

    ax2.barh(neg_words[::-1], [abs(s) for s in neg_scores[::-1]], color=CORAL, edgecolor="white")
    ax2.set_xlabel("|Sentiment Weight|"); ax2.set_title("Top Negative Financial Terms")
    fig.suptitle("Loughran-McDonald Financial Lexicon: Key Terms",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "03_word_importance.png")


def plot_signal_backtest(proj=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    np.random.seed(42)
    days = 500
    rets = 0.0003 + 0.012 * np.random.randn(days)
    sentiment = np.tanh(np.cumsum(0.01 * np.random.randn(days)))
    signal = np.where(sentiment > 0.1, 1, np.where(sentiment < -0.1, -1, 0))
    strat_rets = signal[:-1] * rets[1:]

    eq_bh = np.cumprod(1 + rets) * 100
    eq_sent = np.cumprod(1 + np.append(0, strat_rets)) * 100

    ax1.plot(eq_bh, color="gray", lw=1.5, label="Buy & Hold")
    ax1.plot(eq_sent, color=TEAL, lw=2.5, label="Sentiment Strategy")
    ax1.set_ylabel("Portfolio Value ($)"); ax1.set_title("Sentiment Strategy vs Buy & Hold")
    ax1.legend()

    dd = eq_sent / np.maximum.accumulate(eq_sent) - 1
    ax2.fill_between(range(len(dd)), dd * 100, color=CORAL, alpha=0.5)
    ax2.set_xlabel("Trading Days"); ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Strategy Drawdown")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "04_signal_backtest.png")


def plot_confusion_matrix(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Simulated confusion matrix
    cm = np.array([[120, 35], [28, 117]])
    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=16,
                     fontweight="bold", color="white" if cm[i,j] > 80 else "black")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Bearish", "Bullish"]); ax1.set_yticklabels(["Bearish", "Bullish"])
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")

    # Metrics
    prec = 117 / (117 + 35)
    rec = 117 / (117 + 28)
    f1 = 2 * prec * rec / (prec + rec)
    acc = (120 + 117) / cm.sum()
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    bars = ax2.bar(metrics.keys(), metrics.values(), color=[NAVY, TEAL, CORAL, GOLD],
                   edgecolor="white", lw=1.5)
    for b, v in zip(bars, metrics.values()):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Score")
    ax2.set_title("Classification Metrics")
    fig.suptitle("Sentiment Signal Classification Performance",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "05_confusion_matrix.png")


def plot_regime_analysis(proj=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    np.random.seed(42)
    days = 500
    t = np.arange(days)
    sentiment = np.tanh(np.cumsum(0.015 * np.random.randn(days)))
    price = 100 * np.cumprod(1 + 0.0003 + 0.01 * np.random.randn(days))

    regime = np.where(sentiment > 0.3, 2, np.where(sentiment > -0.3, 1, 0))
    regime_names = {0: "Fear", 1: "Neutral", 2: "Greed"}
    regime_colors = {0: CORAL, 1: GOLD, 2: TEAL}

    ax1.plot(t, price, color=NAVY, lw=1.5)
    for r, col in regime_colors.items():
        mask = regime == r
        ax1.fill_between(t, price.min(), price.max(), where=mask,
                         alpha=0.1, color=col, label=regime_names[r])
    ax1.set_ylabel("Price ($)"); ax1.set_title("Market Regimes from Sentiment")
    ax1.legend(loc="upper left")

    ax2.plot(t, sentiment, color=NAVY, lw=1)
    ax2.axhline(0.3, color=TEAL, ls="--", alpha=0.5)
    ax2.axhline(-0.3, color=CORAL, ls="--", alpha=0.5)
    ax2.fill_between(t, sentiment, where=sentiment > 0.3, alpha=0.3, color=TEAL)
    ax2.fill_between(t, sentiment, where=sentiment < -0.3, alpha=0.3, color=CORAL)
    ax2.set_xlabel("Trading Days"); ax2.set_ylabel("Sentiment")
    ax2.set_title("Sentiment Regime Detection")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "06_regime_analysis.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 08 figures...")
    files = [
        plot_sentiment_timeseries(project_dir),
        plot_sentiment_vs_returns(project_dir),
        plot_word_importance(project_dir),
        plot_signal_backtest(project_dir),
        plot_confusion_matrix(project_dir),
        plot_regime_analysis(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files

if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))

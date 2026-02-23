"""
Forecast Visualization
Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                      "axes.titleweight": "bold", "savefig.dpi": 300,
                      "axes.grid": True, "grid.alpha": 0.3})

COLORS = {"lstm": "#1a365d", "gru": "#e53e3e", "transformer": "#38a169",
          "actual": "black"}


def plot_predictions_comparison(actual, predictions_dict,
                                output_dir="outputs/figures"):
    """Plot actual vs predicted returns for all models."""
    fig, axes = plt.subplots(len(predictions_dict), 1,
                             figsize=(14, 4*len(predictions_dict)), sharex=True)
    if len(predictions_dict) == 1:
        axes = [axes]

    colors = list(COLORS.values())
    for ax, (name, preds), c in zip(axes, predictions_dict.items(), colors):
        n = min(len(actual), len(preds))
        ax.plot(actual[:n], color="black", lw=1, alpha=0.7, label="Actual")
        ax.plot(preds[:n], color=c, lw=1.5, label=f"{name} Prediction")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Return")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Model Predictions vs Actual Returns",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "predictions_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {output_dir}/predictions_comparison.png")


def plot_training_history(histories, output_dir="outputs/figures"):
    """Plot training/validation loss curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = list(COLORS.values())
    for (name, hist), c in zip(histories.items(), colors):
        ax.plot(hist["loss"], color=c, lw=2, label=f"{name} Train")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], color=c, lw=2, ls="--",
                    label=f"{name} Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History", fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "training_history.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {output_dir}/training_history.png")

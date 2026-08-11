"""
Generate Graphical Abstract for Science Bulletin submission.
Creates a single compelling figure summarizing the paper's key message:
  1. Random split on sliding windows → temporal data leakage
  2. Leakage can reverse method rankings (ρ = -0.7 on SWaT)
  3. Fix: Temporal-Split Protocol (TSP) — split BEFORE windowing

Output: paper/figures/graphical_abstract.pdf (vector) + .png (raster)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
import os

# ── Style ──
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
DARK = "#333333"
LIGHT = "#888888"

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_panel_1(ax):
    """Panel 1: Random split creates leakage."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(A) Random Split on Sliding Windows", fontweight="bold", color=CB_RED, fontsize=12)

    # Draw time series
    t = np.linspace(0, 10, 200)
    ts = np.sin(t * 1.5) + 0.3 * np.sin(t * 5) + 0.1 * np.random.randn(200)
    ax.plot(t * 0.9 + 0.3, ts * 0.35 + 4.2, color=DARK, linewidth=1.2)
    ax.text(0.2, 5.2, "Raw Time Series", fontsize=9, color=DARK)

    # Draw 3 overlapping windows
    colors_w = [CB_BLUE, CB_ORANGE, CB_GREEN]
    for i, (x_start, color) in enumerate(zip([1.2, 2.8, 4.4], colors_w)):
        rect = FancyBboxPatch((x_start, 1.2), 3.3, 1.8, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor="gray", alpha=0.25, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x_start + 1.65, 3.1, f"Window {i+1}", ha="center", fontsize=8, color=color, fontweight="bold")

    # Overlap annotation
    ax.annotate("", xy=(4.5, 1.2), xytext=(4.5, 3.0),
                arrowprops=dict(arrowstyle="<->", color=CB_RED, lw=1.5))
    ax.text(5.2, 1.8, "75% overlap\n= LEAKAGE", fontsize=7.5, color=CB_RED, fontweight="bold")

    # Random split
    ax.text(1.0, 0.5, "Random Split → Train/Test: ", fontsize=8, color=DARK)
    ax.text(3.0, 0.5, "Window 2 in TRAIN, Window 3 in TEST", fontsize=8, color=CB_RED, fontweight="bold")
    ax.text(3.0, 0.1, "→ Model has already seen 75% of test data!", fontsize=8, color=CB_RED, fontstyle="italic")


def draw_panel_2(ax):
    """Panel 2: Leakage reverses method rankings."""
    ax.set_title("(B) Leakage Reverses Rankings (SWaT)", fontweight="bold", fontsize=12)

    methods = ["Z-score", "IsoForest", "OCSVM", "LSTM-AE", "Transf-AE"]

    # Data from experiments
    random_auc = [0.846, 0.683, 0.816, 0.619, 0.743]
    temporal_auc = [0.712, 0.822, 0.773, 0.870, 0.908]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, random_auc, width, label="Random Split (current standard)",
                   color=CB_RED, alpha=0.7, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, temporal_auc, width, label="Temporal Split (TSP, honest)",
                   color=CB_GREEN, alpha=0.7, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("AUC-ROC", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Rank reversal annotation
    ax.annotate("ρ = −0.70\nRankings INVERTED!",
                xy=(3, 0.908), xytext=(2.3, 0.95),
                fontsize=10, fontweight="bold", color=CB_RED,
                arrowprops=dict(arrowstyle="->", color=CB_RED, lw=1.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Add rank numbers
    for i in range(5):
        ax.text(x[i] - width/2, random_auc[i] + 0.01, f"#{i+1}", ha="center", fontsize=7, color=CB_RED, fontweight="bold")
        # Temporal rankings (re-sort)
    temp_ranks = np.argsort(np.argsort(temporal_auc)[::-1]) + 1
    for i in range(5):
        ax.text(x[i] + width/2, temporal_auc[i] + 0.01, f"#{temp_ranks[i]}", ha="center", fontsize=7, color=CB_GREEN, fontweight="bold")


def draw_panel_3(ax):
    """Panel 3: TSP — the fix."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("(C) Temporal-Split Protocol (TSP): Split BEFORE Windowing", fontweight="bold", color=CB_GREEN, fontsize=12)

    # Draw time series
    t = np.linspace(0, 10, 200)
    ts = np.sin(t * 1.5) + 0.3 * np.sin(t * 5) + 0.1 * np.random.randn(200)
    ax.plot(t * 0.9 + 0.3, ts * 0.3 + 3.5, color=DARK, linewidth=1.2)

    # Split line
    ax.axvline(x=5.5, color=CB_GREEN, linewidth=2.5, linestyle="--")
    ax.text(5.5, 4.7, "SPLIT", fontsize=10, color=CB_GREEN, fontweight="bold", ha="center")

    # Training period
    rect_train = FancyBboxPatch((0.5, 1.5), 5.0, 1.2, boxstyle="round,pad=0.08",
                                facecolor=CB_GREEN, alpha=0.15, edgecolor=CB_GREEN, linewidth=1.5)
    ax.add_patch(rect_train)
    ax.text(3.0, 2.85, "TRAINING\n(70%)", ha="center", fontsize=10, color=CB_GREEN, fontweight="bold")

    # Test period
    rect_test = FancyBboxPatch((5.5, 1.5), 4.0, 1.2, boxstyle="round,pad=0.08",
                               facecolor=CB_BLUE, alpha=0.15, edgecolor=CB_BLUE, linewidth=1.5)
    ax.add_patch(rect_test)
    ax.text(7.5, 2.85, "TEST\n(30%)", ha="center", fontsize=10, color=CB_BLUE, fontweight="bold")

    # Code snippet
    code_box = FancyBboxPatch((0.5, 0.1), 9.0, 1.2, boxstyle="round,pad=0.08",
                              facecolor="black", alpha=0.85, edgecolor="gray")
    ax.add_patch(code_box)
    ax.text(1.0, 1.05, "# TSP: 1 line, zero cost\nsplit_idx = len(ts) * 0.7\nX_train = windows(ts[:split_idx])\nX_test  = windows(ts[split_idx:])",
            fontsize=7.2, color="white", fontfamily="monospace", va="center")


def main():
    fig = plt.figure(figsize=(14, 10))

    # ── Top panel: visual comparison ──
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    draw_panel_1(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    draw_panel_2(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    draw_panel_3(ax3)

    # ── Overall title ──
    fig.suptitle("When Benchmarks Lie:\nTemporal Data Leakage Reverses Time Series Anomaly Detection Rankings",
                 fontsize=15, fontweight="bold", y=0.99, color=DARK)

    # Bottom tagline
    fig.text(0.5, 0.005, "Proposed fix: Temporal-Split Protocol (TSP) — one line of code, zero new infrastructure, eliminates the largest source of TSAD evaluation inflation",
             ha="center", fontsize=9, fontstyle="italic", color=LIGHT)

    # Save
    for fmt in ["pdf", "png"]:
        out_path = f"{OUTPUT_DIR}/graphical_abstract.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved: {out_path}")

    plt.close(fig)
    print("\nGraphical abstract generated successfully!")


if __name__ == "__main__":
    main()

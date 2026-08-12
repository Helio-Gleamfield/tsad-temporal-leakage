"""
Science Bulletin Graphical Abstract — Professional Redesign
===========================================================
Clean, minimal, publication-quality figure using Wong 2011 colorblind-safe palette.
Three-panel layout: Problem → Finding → Solution
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Global Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Wong 2011 Colorblind-Safe Palette ──
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
CYAN   = "#56B4E9"
DARK   = "#2D2D2D"
GRAY   = "#888888"
LIGHT  = "#E8E8E8"
WHITE  = "#FAFAFA"
BG     = "#FFFFFF"

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def panel_problem(ax):
    """Left: The leakage mechanism — clean timeline diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Title
    ax.text(5, 5.3, "The Problem: Temporal Data Leakage",
            ha="center", fontsize=11, fontweight="bold", color=DARK)

    # ── Raw time series bar ──
    ts_y = 3.8
    rect = FancyBboxPatch((0.3, ts_y - 0.25), 9.4, 0.5, boxstyle="round,pad=0.05",
                          facecolor=LIGHT, edgecolor=GRAY, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(5, ts_y, "Raw Time Series  (T time steps, D sensors)",
            ha="center", fontsize=8, color=DARK, fontweight="bold")

    # ── Three overlapping windows ──
    win_colors = [BLUE, ORANGE, CYAN]
    win_labels  = ["Window i", "Window i+1", "Window i+2"]
    win_y = 2.3

    for j, (xc, col, lab) in enumerate(zip([0.8, 2.8, 4.8], win_colors, win_labels)):
        rect_w = FancyBboxPatch((xc, win_y - 0.3), 3.8, 0.6, boxstyle="round,pad=0.06",
                                facecolor=col, edgecolor="white", alpha=0.35, linewidth=0.6)
        ax.add_patch(rect_w)
        ax.text(xc + 1.9, win_y, lab, ha="center", fontsize=7, color=col, fontweight="bold")

    # Overlap bracket
    ax.annotate("", xy=(4.6, win_y - 0.3), xytext=(3.8, win_y + 0.45),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(6.2, win_y + 0.05, "75% overlap\nbetween adjacent\nwindows", fontsize=6.8, color=RED, fontweight="bold")

    # ── Random split annotation ──
    ax.text(1.8, 1.2, "Random split mixes overlapping\nwindows into train & test",
            ha="center", fontsize=7.5, color=RED, fontweight="bold")
    ax.text(1.8, 0.7, "→ Model trains on test data",
            ha="center", fontsize=7.5, color=RED, fontstyle="italic")

    # Right side of panel 1: arrow to panel 2
    ax.annotate("", xy=(9.8, 2.8), xytext=(9.4, 2.8),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))


def panel_finding(ax):
    """Middle: The key result — rankings reversed."""
    ax.set_title("The Finding: Rankings Reversed", fontsize=11, fontweight="bold", color=DARK, pad=8)

    methods = ["Z-score", "IsoForest", "OCSVM", "LSTM-AE", "Transf-AE"]
    x = np.arange(len(methods))
    width = 0.30

    random_auc  = [0.846, 0.683, 0.816, 0.619, 0.743]
    temporal_auc = [0.712, 0.822, 0.773, 0.870, 0.908]

    bars1 = ax.bar(x - width/2, random_auc, width, color=RED, alpha=0.6,
                   edgecolor="white", linewidth=0.3, label="Random Split (current)")
    bars2 = ax.bar(x + width/2, temporal_auc, width, color=GREEN, alpha=0.6,
                   edgecolor="white", linewidth=0.3, label="Temporal Split (honest)")

    # Rank labels
    for i in range(5):
        ax.text(x[i] - width/2, random_auc[i] + 0.018, f"#{i+1}", ha="center", fontsize=6.5, color=RED, fontweight="bold")
        temp_ranks = [4, 3, 2, 1, 0]  # Inverted ranks
        ax.text(x[i] + width/2, temporal_auc[i] + 0.018, f"#{temp_ranks[i]+1}",
                ha="center", fontsize=6.5, color=GREEN, fontweight="bold")

    # Rho annotation
    ax.annotate("ρ = −0.70\n(SWaT)",
                xy=(2, 0.908), fontsize=9.5, fontweight="bold", color=RED,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=YELLOW, alpha=0.85, edgecolor=RED, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7.5, rotation=15)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.set_ylim(0.48, 0.98)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, edgecolor=GRAY)
    ax.grid(axis="y", alpha=0.2, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_solution(ax):
    """Right: TSP — the one-line fix."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(5, 5.3, "The Fix: Temporal-Split Protocol (TSP)",
            ha="center", fontsize=11, fontweight="bold", color=GREEN)

    # ── Time series with split ──
    ts_y = 3.5
    rect = FancyBboxPatch((0.3, ts_y - 0.25), 9.4, 0.5, boxstyle="round,pad=0.05",
                          facecolor=LIGHT, edgecolor=GRAY, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(5, ts_y, "Raw Time Series",
            ha="center", fontsize=8, color=DARK, fontweight="bold")

    # Split line
    ax.axvline(x=5.5, ymin=0.28, ymax=0.78, color=GREEN, linewidth=2.5, linestyle="--")
    ax.text(5.5, 4.05, "SPLIT\nFIRST", fontsize=8.5, color=GREEN, fontweight="bold", ha="center")

    # Train section
    rect_tr = FancyBboxPatch((0.5, 1.8), 4.8, 1.0, boxstyle="round,pad=0.08",
                             facecolor=GREEN, alpha=0.12, edgecolor=GREEN, linewidth=1.5)
    ax.add_patch(rect_tr)
    ax.text(2.9, 2.55, "TRAINING (70%)", ha="center", fontsize=8.5, fontweight="bold", color=GREEN)
    ax.text(2.9, 2.05, "Generate windows\nfrom training period only", ha="center", fontsize=7, color=DARK)

    # Test section
    rect_te = FancyBboxPatch((5.5, 1.8), 4.0, 1.0, boxstyle="round,pad=0.08",
                             facecolor=BLUE, alpha=0.12, edgecolor=BLUE, linewidth=1.5)
    ax.add_patch(rect_te)
    ax.text(7.5, 2.55, "TEST (30%)", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    ax.text(7.5, 2.05, "Generate windows\nfrom test period only", ha="center", fontsize=7, color=DARK)

    # Code box
    code_box = FancyBboxPatch((0.5, 0.15), 9.0, 1.05, boxstyle="round,pad=0.08",
                              facecolor=DARK, alpha=0.92, edgecolor=DARK)
    ax.add_patch(code_box)
    ax.text(5, 0.95, "# TSP: One line, zero new infrastructure — just reorder your preprocessing",
            ha="center", fontsize=7.2, color=YELLOW, fontfamily="monospace")
    ax.text(5, 0.55, "split_idx = int(len(ts) * 0.7)\nts_train, ts_test = ts[:split_idx], ts[split_idx:]",
            ha="center", fontsize=8, color=WHITE, fontfamily="monospace")


def main():
    fig = plt.figure(figsize=(15, 6.5))

    # Three-panel grid
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.03, right=0.97, top=0.88, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_problem(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_finding(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    panel_solution(ax3)

    # Global title
    fig.suptitle("When Benchmarks Lie: Temporal Data Leakage Reverses TSAD Rankings",
                 fontsize=14, fontweight="bold", color=DARK, y=0.98)

    # ── Save ──
    for fmt in ["pdf", "png"]:
        path = f"{OUTPUT_DIR}/graphical_abstract.{fmt}"
        fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none")
        print(f"  Saved: {path}")

    plt.close(fig)
    print("Done — clean graphical abstract generated.")


if __name__ == "__main__":
    main()

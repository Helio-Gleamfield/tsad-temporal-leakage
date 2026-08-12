"""
Science Bulletin — Comprehensive Multi-Panel Figure (Zhang et al. quality)
==========================================================================
Five panels telling the complete story:
  (a) Leakage mechanism schematic
  (b) ΔAUC across 5 datasets
  (c) Ranking reversal on SWaT (ρ = −0.70)
  (d) Component ablation waterfall
  (e) Full temporal-split results matrix

Design: Wong 2011 colorblind-safe, publication typography, 300 DPI
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os, json

# ── Global Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Wong 2011 Colorblind-Safe ──
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
BG     = "#FFFFFF"

# Method colors
MCOLORS = {"Z-score": BLUE, "IsoForest": ORANGE, "OCSVM": GREEN,
           "LSTM-AE": RED, "Transformer-AE": PURPLE}

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Real Data ──
DATASETS = ["SWaT", "TEP", "MSL", "SMAP", "SMD"]
METHODS  = ["Z-score", "IsoForest", "OCSVM", "LSTM-AE", "Transformer-AE"]

# ΔAUC per method per dataset (from JSON)
DELTA_AUC = {
    "SWaT": [0.134, -0.139, 0.043, -0.251, -0.165],
    "TEP":  [0.020, 0.024, 0.028, 0.023, 0.053],
    "MSL":  [0.067, 0.038, -0.138, 0.104, 0.054],
    "SMAP": [0.016, 0.123, 0.092, -0.007, 0.018],
    "SMD":  [0.065, 0.117, 0.043, 0.091, 0.150],
}

# Temporal AUC
TEMPORAL_AUC = {
    "SWaT": [0.712, 0.822, 0.773, 0.870, 0.908],
    "TEP":  [0.793, 0.740, 0.683, 0.803, 0.777],
    "MSL":  [0.578, 0.518, 0.635, 0.537, 0.559],
    "SMAP": [0.458, 0.329, 0.360, 0.501, 0.479],
    "SMD":  [0.775, 0.446, 0.490, 0.700, 0.682],
}

# Random AUC for SWaT
RANDOM_AUC_SWAT = [0.846, 0.683, 0.816, 0.619, 0.743]

# Ablation
ABLATION_LABELS = ["6-dim\nbaseline", "+Enhanced\nstats", "+VLM\nfeatures", "+Recon\nloss", "+Patch\nTransformer"]
ABLATION_AUC    = [0.497, 0.588, 0.588, 0.588, 0.589]
ABLATION_DELTA  = [0, 0.092, 0.000, 0.000, 0.001]


# ══════════════════════════════════════════════════════════════
def panel_a_schematic(ax):
    """Leakage mechanism: random vs temporal split comparison."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("(a)  Sliding-Window Leakage Mechanism", fontweight="bold", loc="left", pad=4)

    # Raw TS bar
    ts_y = 4.2
    rect = FancyBboxPatch((0.3, ts_y-0.22), 11.4, 0.44, boxstyle="round,pad=0.04",
                          facecolor=LIGHT, edgecolor=GRAY, linewidth=0.6)
    ax.add_patch(rect)
    ax.text(6, ts_y, "Raw Multivariate Time Series", ha="center", fontsize=7, fontweight="bold", color=DARK)

    # ── Random split side (left half, y=2.0-3.5) ──
    ax.text(3, 3.6, "Current Practice: Random Split", ha="center", fontsize=7.5, color=RED, fontweight="bold")
    win_colors = [BLUE, ORANGE, CYAN]
    for j, (xc, col) in enumerate(zip([0.8, 2.8, 4.8], win_colors)):
        rect_w = FancyBboxPatch((xc, 2.15), 3.8, 0.55, boxstyle="round,pad=0.05",
                                facecolor=col, edgecolor="white", alpha=0.30, linewidth=0.5)
        ax.add_patch(rect_w)
        ax.text(xc+1.9, 2.43, f"W{j+1}", ha="center", fontsize=6.5, color=col, fontweight="bold")
    # Overlap bracket
    ax.annotate("", xy=(4.6, 2.7), xytext=(3.8, 2.15),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(5.8, 2.55, "75% overlap\n→ leakage", fontsize=6, color=RED, fontweight="bold")
    # Shuffle arrow
    ax.annotate("Shuffle", xy=(3, 1.4), xytext=(3, 1.85), ha="center", fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    # ── Temporal split side (right half, y=2.0-3.5) ──
    shift_x = 6.2
    ax.text(shift_x+3, 3.6, "Proposed: Temporal Split", ha="center", fontsize=7.5, color=GREEN, fontweight="bold")
    # Split line
    ax.axvline(x=shift_x+3, ymin=0.45, ymax=0.60, color=GREEN, linewidth=2, linestyle="--")
    # Train
    rect_tr = FancyBboxPatch((shift_x+0.3, 2.15), 2.4, 0.55, boxstyle="round,pad=0.05",
                             facecolor=GREEN, alpha=0.20, edgecolor=GREEN, linewidth=0.8)
    ax.add_patch(rect_tr)
    ax.text(shift_x+1.5, 2.43, "Train windows", ha="center", fontsize=6.5, color=GREEN, fontweight="bold")
    # Test
    rect_te = FancyBboxPatch((shift_x+3.3, 2.15), 2.7, 0.55, boxstyle="round,pad=0.05",
                             facecolor=BLUE, alpha=0.20, edgecolor=BLUE, linewidth=0.8)
    ax.add_patch(rect_te)
    ax.text(shift_x+4.65, 2.43, "Test windows", ha="center", fontsize=6.5, color=BLUE, fontweight="bold")
    ax.text(shift_x+3, 1.65, "No temporal\noverlap", ha="center", fontsize=6.5, color=GREEN, fontweight="bold")

    # Bottom code snippets
    code1 = "windows = slide(ts); X_tr,X_te = split(windows, shuffle=True)   # LEAKAGE"
    code2 = "ts_tr,ts_te = ts[:idx], ts[idx:]; X_tr = slide(ts_tr); X_te = slide(ts_te)   # TSP"
    ax.text(0.5, 0.7, code1, fontsize=5.8, fontfamily="monospace", color=RED,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF0F0", edgecolor=RED, alpha=0.5))
    ax.text(0.5, 0.1, code2, fontsize=5.8, fontfamily="monospace", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0FFF0", edgecolor=GREEN, alpha=0.5))


def panel_b_delta_auc(ax):
    """ΔAUC bar chart across 5 datasets."""
    ax.set_title("(b)  Leakage Gap |ΔAUC| Across Datasets", fontweight="bold", loc="left", pad=4)
    x = np.arange(len(DATASETS))
    widths = 0.15
    for j, method in enumerate(METHODS):
        vals = [DELTA_AUC[d][j] for d in DATASETS]
        bars = ax.bar(x + j*widths - 2*widths, vals, widths, color=MCOLORS[method],
                      alpha=0.75, edgecolor="white", linewidth=0.2, label=method)
    ax.axhline(y=0, color=DARK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=7)
    ax.set_ylabel("ΔAUC  (Random − Temporal)", fontsize=7)
    ax.legend(fontsize=5.5, ncol=3, loc="lower left", framealpha=0.8, edgecolor=GRAY)
    ax.set_ylim(-0.32, 0.22)
    ax.grid(axis="y", alpha=0.15, linestyle="-")


def panel_c_ranking(ax):
    """Ranking reversal on SWaT."""
    ax.set_title("(c)  Ranking Reversal on SWaT  (ρ = −0.70)", fontweight="bold", loc="left", pad=4)
    x = np.arange(len(METHODS))
    width = 0.32
    b1 = ax.bar(x - width/2, RANDOM_AUC_SWAT, width, color=RED, alpha=0.55,
                edgecolor="white", linewidth=0.3, label="Random Split")
    b2 = ax.bar(x + width/2, TEMPORAL_AUC["SWaT"], width, color=GREEN, alpha=0.55,
                edgecolor="white", linewidth=0.3, label="Temporal Split (TSP)")
    # Rank numbers
    rand_ranks = np.argsort(np.argsort(RANDOM_AUC_SWAT)[::-1]) + 1
    temp_ranks = np.argsort(np.argsort(TEMPORAL_AUC["SWaT"])[::-1]) + 1
    for i in range(5):
        ax.text(x[i]-width/2, RANDOM_AUC_SWAT[i]+0.015, f"#{rand_ranks[i]}", ha="center", fontsize=6, color=RED, fontweight="bold")
        ax.text(x[i]+width/2, TEMPORAL_AUC["SWaT"][i]+0.015, f"#{temp_ranks[i]}", ha="center", fontsize=6, color=GREEN, fontweight="bold")
    # Inversion arrow
    ax.annotate("Inverted!", xy=(2.5, 0.88), fontsize=8, fontweight="bold", color=RED, ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=YELLOW, alpha=0.80, edgecolor=RED, linewidth=1))
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=6.5, rotation=12)
    ax.set_ylabel("AUC-ROC", fontsize=7)
    ax.set_ylim(0.50, 0.96)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8, edgecolor=GRAY)
    ax.grid(axis="y", alpha=0.15, linestyle="-")


def panel_d_ablation(ax):
    """Component ablation waterfall."""
    ax.set_title("(d)  Component Ablation on SWaT", fontweight="bold", loc="left", pad=4)
    colors_ab = [GRAY, GREEN, RED, RED, RED]
    bars = ax.bar(range(5), ABLATION_AUC, color=colors_ab, alpha=0.65, edgecolor="white", linewidth=0.3, width=0.55)
    # Delta annotations
    for i, (auc, delta) in enumerate(zip(ABLATION_AUC, ABLATION_DELTA)):
        clr = GREEN if delta > 0 else GRAY
        label = f"+{delta:.3f}" if delta > 0 else "baseline" if delta == 0 and i == 0 else "0"
        ax.text(i, auc + 0.012, label, ha="center", fontsize=6.5, color=clr, fontweight="bold")
    # Arrow showing the only real gain
    ax.annotate("", xy=(1, 0.61), xytext=(0, 0.53),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
    ax.text(0.5, 0.62, "+0.092", fontsize=7.5, fontweight="bold", color=GREEN, ha="center")
    ax.set_xticks(range(5))
    ax.set_xticklabels(ABLATION_LABELS, fontsize=6)
    ax.set_ylabel("AUC-ROC", fontsize=7)
    ax.set_ylim(0.44, 0.64)
    ax.grid(axis="y", alpha=0.15, linestyle="-")


def panel_e_full_results(ax):
    """Heatmap of temporal-split AUC across all datasets and methods."""
    ax.set_title("(e)  Honest (Temporal-Split) AUC-ROC Matrix", fontweight="bold", loc="left", pad=4)
    data_matrix = np.array([TEMPORAL_AUC[d] for d in DATASETS])
    im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd", vmin=0.30, vmax=0.95)
    # Annotate cells
    for i in range(len(DATASETS)):
        for j in range(len(METHODS)):
            val = data_matrix[i, j]
            is_best = val == np.max(data_matrix[i, :])
            txt = ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6.5,
                          fontweight="bold" if is_best else "normal",
                          color="white" if val > 0.70 else DARK)
            if is_best:
                txt.set_bbox(dict(boxstyle="round,pad=0.15", facecolor=DARK, alpha=0.55, edgecolor="none"))
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=6.5, rotation=15)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS, fontsize=6.5)
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AUC-ROC", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)


# ══════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(16, 11))

    # Grid: 2 rows × 3 cols, panels a and b share row 1, c/d/e share row 2
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28,
                          left=0.04, right=0.98, top=0.96, bottom=0.04,
                          height_ratios=[0.42, 0.58])

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_schematic(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_delta_auc(ax_b)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_c_ranking(ax_c)

    ax_d = fig.add_subplot(gs[1, 0])
    panel_d_ablation(ax_d)

    ax_e = fig.add_subplot(gs[1, 1:])
    panel_e_full_results(ax_e)

    # Global title
    fig.suptitle("Temporal Data Leakage Reverses Time Series Anomaly Detection Rankings",
                 fontsize=13, fontweight="bold", color=DARK, y=1.0)

    for fmt in ["pdf", "png"]:
        path = f"{OUTPUT_DIR}/comprehensive_figure.{fmt}"
        fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none")
        print(f"  Saved: {path}")

    plt.close(fig)
    print("Comprehensive figure generated — Zhang et al. quality.")


if __name__ == "__main__":
    main()

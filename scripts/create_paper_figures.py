"""
Generate publication-quality figures for the Science Bulletin paper.
Produces:
  1. fig1_leakage_gap.png — Temporal leakage bar chart (random vs temporal AUC)
  2. fig2_ablation_waterfall.png — Component ablation waterfall plot
  3. fig3_causal_f1.png — Causal discovery F1 comparison
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Style Configuration ──────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

# Colorblind-safe palette (Wong 2011)
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_SKY = "#56B4E9"
CB_GREY = "#999999"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Temporal Leakage Gap
# ═══════════════════════════════════════════════════════════════════
def create_leakage_figure():
    datasets = ["SWaT", "TEP"]
    random_auc = [0.7263, 0.8647]
    temporal_auc = [0.4688, 0.8389]
    zscore_auc = [0.4325, 0.8428]
    leakage_gap = [0.2575, 0.0258]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(datasets))
    width = 0.28

    # Left: Grouped bar chart
    bars1 = ax1.bar(x - width, random_auc, width, label="Random Split (Leaky)", color=CB_RED, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x, temporal_auc, width, label="Temporal Split (Honest)", color=CB_BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars3 = ax1.bar(x + width, zscore_auc, width, label="Z-score Baseline", color=CB_GREY, alpha=0.7, edgecolor="white", linewidth=0.5, hatch="//")

    # Annotate gaps
    for i, (r, t) in enumerate(zip(random_auc, temporal_auc)):
        gap = r - t
        mid = (r + t) / 2
        ax1.annotate(f"Δ={gap:.3f}", (x[i], mid), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="darkred",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))

    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("(a) Detection Performance Under Different Splits")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="bold")
    ax1.set_ylim(0.3, 1.0)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: Leakage gap bars
    gap_colors = [CB_RED, CB_SKY]
    bars = ax2.bar(datasets, leakage_gap, color=gap_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0.14, color=CB_GREY, linestyle="--", linewidth=1.5, label=f"Mean ΔAUC = 0.142")
    ax2.set_ylabel("ΔAUC (Random − Temporal)")
    ax2.set_title("(b) Leakage Gap (ΔAUC)")
    ax2.legend(framealpha=0.9)

    # Annotate bars
    for bar, gap in zip(bars, leakage_gap):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"+{gap:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 0.35)

    fig.suptitle("Fig. 1: Temporal Data Leakage in TSAD Benchmarks", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_leakage_gap.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "fig1_leakage_gap.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Fig 1 saved: leakage gap chart")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Component Ablation Waterfall
# ═══════════════════════════════════════════════════════════════════
def create_ablation_figure():
    variants = ["6-dim stats\n(Baseline)", "+Enhanced stats\n(6→12 dim)",
                "+Random VLM\nfeatures", "+Reconstruction\nloss"]
    auc_values = [0.4967, 0.5883, 0.5883, 0.5883]
    deltas = [0.0, 0.0916, 0.0000, 0.0000]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = [CB_GREY, CB_GREEN, CB_ORANGE, CB_PURPLE]
    x_pos = np.arange(len(variants))

    # Bar chart
    bars = ax.bar(x_pos, auc_values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8, width=0.55)

    # Delta annotations
    for i, (bar, delta, auc) in enumerate(zip(bars, deltas, auc_values)):
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"AUC={auc:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        else:
            color = CB_GREEN if delta > 0 else CB_RED
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"AUC={auc:.4f}\nΔ={delta:+.4f}", ha="center", va="bottom",
                   fontweight="bold", fontsize=10, color=color)

    # Baseline reference line
    ax.axhline(y=0.4967, color=CB_GREY, linestyle="--", linewidth=1.2, alpha=0.6)

    ax.set_ylabel("AUC-ROC (Temporal Split)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants)
    ax.set_ylim(0.40, 0.65)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Waterfall connector
    for i in range(len(variants) - 1):
        ax.plot([x_pos[i] + 0.275, x_pos[i+1] - 0.275], [auc_values[i], auc_values[i+1]],
               color=CB_GREY, linewidth=1.5, linestyle=":", alpha=0.5)

    ax.set_title("Fig. 2: ViCSynAD Component Ablation — VLM Features Contribute Zero", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_ablation_waterfall.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "fig2_ablation_waterfall.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Fig 2 saved: ablation waterfall")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Causal Discovery F1 Comparison
# ═══════════════════════════════════════════════════════════════════
def create_causal_figure():
    methods = ["Correlation\nThreshold", "PC\n(no prior)", "PC\n(+ P&ID prior)", "P&ID GT\n(upper bound)"]
    f1_scores = [0.000, 0.500, 0.627, 1.000]
    precision_vals = [0.000, 0.380, 0.456, 1.000]
    recall_vals = [0.000, 0.731, 1.000, 1.000]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(methods))
    width = 0.25

    bars_f1 = ax.bar(x - width, f1_scores, width, label="F1-score", color=CB_BLUE, alpha=0.85, edgecolor="white")
    bars_p = ax.bar(x, precision_vals, width, label="Precision", color=CB_ORANGE, alpha=0.85, edgecolor="white")
    bars_r = ax.bar(x + width, recall_vals, width, label="Recall", color=CB_GREEN, alpha=0.85, edgecolor="white")

    # Annotate F1 values
    for bar, val in zip(bars_f1, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Annotation arrow showing prior improvement
    ax.annotate("", xy=(2 - width, 0.53), xytext=(1 - width, 0.53),
               arrowprops=dict(arrowstyle="->", color=CB_RED, lw=2))
    ax.text(1.5 - width, 0.56, "+25.4%\nΔF1=+0.127", ha="center", fontsize=9,
           fontweight="bold", color=CB_RED)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.set_title("Fig. 3: Causal Discovery — Domain Prior Contribution (SWaT, 12 sensors)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_causal_f1.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "fig3_causal_f1.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Fig 3 saved: causal discovery F1")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Summary — Key Findings Dashboard
# ═══════════════════════════════════════════════════════════════════
def create_summary_figure():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Leakage gap (reprise from Fig 1b)
    datasets = ["SWaT", "TEP", "Mean"]
    gaps = [0.2575, 0.0258, 0.1416]
    colors = [CB_RED, CB_SKY, CB_GREY]
    axes[0].bar(datasets, gaps, color=colors, alpha=0.85, edgecolor="white")
    for i, (d, g) in enumerate(zip(datasets, gaps)):
        axes[0].text(i, g + 0.008, f"+{g:.3f}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("ΔAUC")
    axes[0].set_title("(a) Temporal Leakage Gap", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylim(0, 0.35)

    # Panel 2: Ablation summary
    components = ["Enhanced\nStats", "VLM\nFeatures", "Recon\nLoss"]
    contributions = [0.0916, 0.0000, 0.0000]
    bar_colors = [CB_GREEN if c > 0 else CB_RED for c in contributions]
    axes[1].bar(components, contributions, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    for i, (c, val) in enumerate(zip(components, contributions)):
        axes[1].text(i, val + 0.003 if val >= 0 else val - 0.008,
                    f"{val:+.4f}", ha="center", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("ΔAUC Contribution")
    axes[1].set_title("(b) Component Contribution", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 3: Causal F1
    methods = ["Correlation", "PC\n(no prior)", "PC\n(+ prior)"]
    f1_vals = [0.000, 0.500, 0.627]
    bar_colors3 = [CB_GREY, CB_ORANGE, CB_BLUE]
    axes[2].bar(methods, f1_vals, color=bar_colors3, alpha=0.85, edgecolor="white")
    for i, (m, f) in enumerate(zip(methods, f1_vals)):
        axes[2].text(i, f + 0.02, f"{f:.3f}", ha="center", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("F1-score")
    axes[2].set_title("(c) Causal Discovery F1", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].set_ylim(0, 0.75)

    fig.suptitle("Fig. 4: Key Findings — Temporal Leakage, Component Ablation, and Causal Discovery",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_summary_dashboard.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "fig4_summary_dashboard.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Fig 4 saved: summary dashboard")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    create_leakage_figure()
    create_ablation_figure()
    create_causal_figure()
    create_summary_figure()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")

"""
P0 Figure Generation — Publication-quality figures for Science Bulletin submission.
Generates: Fig 1 (split schematic), Fig 2 (overlap ablation), Fig 3 (parallel coordinates),
Table 1 (leakage quantification), Table 2 (overlap ablation), Table 3 (ranking stability).

Usage:
    python scripts/experiment_p0_figures.py --results experiments/p0_results/p0_comprehensive_results.json
"""
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
COLORS = sns.color_palette("colorblind", 8)
METHOD_COLORS = {
    "Z-score": COLORS[0], "IsolationForest": COLORS[1], "OCSVM": COLORS[2],
    "LSTM-AE": COLORS[3], "Transformer-AE": COLORS[4],
}


def fig1_split_schematic():
    """Figure 1: Schematic comparing random vs temporal split on overlapping windows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Simplified visualization of the two split strategies
    n_windows = 12
    overlap = 0.75
    window_length = 0.8

    for ax, title, split_mode in [
        (ax1, "Random Split (Standard — Leakage)", "random"),
        (ax2, "Temporal Split (TSP — No Leakage)", "temporal")
    ]:
        # Draw windows as horizontal bars
        y_positions = np.arange(n_windows) * 0.15
        for i, y in enumerate(y_positions):
            x_start = i * (1 - overlap) * window_length
            color = COLORS[0] if split_mode == "random" else COLORS[3]
            alpha = 0.7 if split_mode == "random" else 0.9

            if split_mode == "random":
                # Color by random train/test assignment
                is_train = np.random.RandomState(42 + i).random() > 0.3
                color = COLORS[0] if is_train else COLORS[3]
            else:
                # First 70% train, last 30% test
                is_train = i < n_windows * 0.7
                color = COLORS[0] if is_train else COLORS[3]

            ax.barh(y, window_length, left=x_start, height=0.12,
                    color=color, alpha=alpha, edgecolor="white", linewidth=0.5)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time →")
        ax.set_yticks([])
        ax.set_ylim(-0.2, y_positions[-1] + 0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS[0], alpha=0.7, label="Training Windows"),
            Patch(facecolor=COLORS[3], alpha=0.7, label="Test Windows"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

        # Red arrows showing leakage in random split
        if split_mode == "random":
            for i in range(n_windows - 1):
                if i < n_windows * 0.7 and i + 1 >= n_windows * 0.7:
                    ax.annotate("⚠ Leak", xy=((i + 0.5) * (1 - overlap) * window_length, y_positions[i]),
                                xytext=((i + 1.5) * (1 - overlap) * window_length, y_positions[i + 1]),
                                arrowprops=dict(arrowstyle="->", color="red", lw=1.5), color="red", fontsize=7)

    fig.suptitle("Figure 1: Random vs. Temporal Split on Overlapping Sliding Windows",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURE_DIR / "fig1_split_schematic.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  ✅ Fig 1 saved: {out}")
    plt.close(fig)


def fig2_overlap_ablation(ablation_data):
    """Figure 2: ΔAUC vs. window overlap ratio."""
    if not ablation_data:
        print("  ⚠️  No ablation data — skipping Fig 2")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    markers = ["o", "s", "D"]
    for idx, (ds_name, ds_data) in enumerate(ablation_data.items()):
        overlaps, deltas = [], []
        for stride_key, stride_data in ds_data.get("ablation", {}).items():
            for mkey, mdata in stride_data.items():
                overlaps.append(mdata["overlap_pct"])
                deltas.append(mdata["delta_auc"])
                break  # Take first method's result per stride
        if overlaps:
            ax.plot(overlaps, deltas, marker=markers[idx % len(markers)],
                    color=COLORS[idx], label=ds_name, markersize=8, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="ΔAUC = 0 (no leakage)")
    ax.set_xlabel("Window Overlap Ratio ρ (%)")
    ax.set_ylabel("ΔAUC (Random − Temporal)")
    ax.set_title("Figure 2: Leakage Gap vs. Window Overlap", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURE_DIR / "fig2_overlap_ablation.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  ✅ Fig 2 saved: {out}")
    plt.close(fig)


def fig3_delta_auc_bar(all_data):
    """Figure 3: ΔAUC bar chart across datasets and methods."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = list(all_data.keys())
    method_names = list(next(iter(all_data.values()))["results"].keys())
    n_datasets = len(datasets)
    n_methods = len(method_names)
    bar_width = 0.15
    x = np.arange(n_datasets)

    for i, mk in enumerate(method_names):
        deltas = []
        for dn in datasets:
            if mk in all_data[dn]["results"]:
                deltas.append(all_data[dn]["results"][mk].get("delta_auc", np.nan))
            else:
                deltas.append(np.nan)
        bars = ax.bar(x + i * bar_width, deltas, bar_width,
                      label=METHODS_DISPLAY.get(mk, mk),
                      color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, deltas):
            if not np.isnan(val) and val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:+.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("ΔAUC (Random − Temporal)")
    ax.set_title("Figure 3: Temporal Leakage Gap Across Datasets and Methods", fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(framealpha=0.9, ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURE_DIR / "fig3_delta_auc_bar.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  ✅ Fig 3 saved: {out}")
    plt.close(fig)


def fig4_temporal_auc_dot(all_data):
    """Figure 4: Temporal-split AUC dot plot — all methods, all datasets."""
    fig, ax = plt.subplots(figsize=(9, 5))

    datasets = list(all_data.keys())
    method_names = list(next(iter(all_data.values()))["results"].keys())
    all_methods_display = [METHODS_DISPLAY.get(mk, mk) for mk in method_names]

    for i, mk in enumerate(method_names):
        aucs, stds = [], []
        for dn in datasets:
            if mk in all_data[dn]["results"]:
                aucs.append(all_data[dn]["results"][mk]["temporal"]["auc_roc"]["mean"])
                stds.append(all_data[dn]["results"][mk]["temporal"]["auc_roc"]["std"])
            else:
                aucs.append(np.nan)
                stds.append(np.nan)
        ax.errorbar(range(len(datasets)), aucs, yerr=stds,
                    marker="o", markersize=8, linewidth=2, capsize=4,
                    color=COLORS[i % len(COLORS)], label=all_methods_display[i])

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUC-ROC (Temporal Split)")
    ax.set_title("Figure 4: Honest (Temporal-Split) Detection Performance", fontweight="bold")
    ax.legend(framealpha=0.9, ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.95)

    plt.tight_layout()
    out = FIGURE_DIR / "fig4_temporal_auc_dot.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  ✅ Fig 4 saved: {out}")
    plt.close(fig)


# Display name mapping
METHODS_DISPLAY = {
    "zscore": "Z-score", "isolation": "IsolationForest", "ocsvm": "OCSVM",
    "lstm_ae": "LSTM-AE", "transformer_ae": "Transformer-AE",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    all_data = data.get("per_dataset", {})
    ablation = data.get("overlap_ablation", {})
    stability = data.get("ranking_stability", {})

    print(f"\n{'='*60}")
    print("  Generating P0 Figures for Science Bulletin")
    print(f"{'='*60}\n")

    fig1_split_schematic()
    fig2_overlap_ablation(ablation)
    fig3_delta_auc_bar(all_data)
    fig4_temporal_auc_dot(all_data)

    # Print ranking stability
    if stability:
        print(f"\n  Ranking Stability:")
        for ds, stab in stability.items():
            print(f"    {ds}: ρ={stab['spearman_rho']}, p={stab['p_value']}")
            print(f"      Random:   {' > '.join(stab['random_ranking'])}")
            print(f"      Temporal: {' > '.join(stab['temporal_ranking'])}")

    print(f"\n  ✅ All figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

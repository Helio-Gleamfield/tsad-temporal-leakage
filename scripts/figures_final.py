"""
Enhanced P0 Figure Generation — Science Bulletin Publication Quality.
Generates all figures and LaTeX-ready tables from experiment results.

Usage:
    python scripts/figures_final.py --results experiments/p0_results/p0_comprehensive_results.json
"""
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Science Bulletin Style ──
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
})
# Colorblind-safe palette (Wong 2011)
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]

# Method key normalization
METHOD_KEYS = {
    "z_score": "Z-score", "zscore": "Z-score",
    "isolation_forest": "Isolation Forest", "isolation": "Isolation Forest",
    "ocsvm": "OCSVM",
    "lstm_ae": "LSTM-AE",
    "transformer_ae": "Transformer-AE",
}
METHOD_DISPLAY = ["Z-score", "Isolation Forest", "OCSVM", "LSTM-AE", "Transformer-AE"]
METHOD_COLORS = dict(zip(METHOD_DISPLAY, CB_PALETTE[:5]))


def normalize_key(key):
    """Map any method key variant to display name."""
    return METHOD_KEYS.get(key.lower().replace("-", "_"), key)


def get_auc_mean(method_results, mode="temporal"):
    """Safely extract AUC-ROC mean from method results."""
    if not method_results:
        return np.nan
    auc_data = method_results.get(mode, {}).get("auc_roc", {})
    if isinstance(auc_data, dict):
        return auc_data.get("mean", np.nan)
    if isinstance(auc_data, (int, float)):
        return auc_data
    return np.nan


def get_auc_std(method_results, mode="temporal"):
    """Safely extract AUC-ROC std."""
    if not method_results:
        return 0
    auc_data = method_results.get(mode, {}).get("auc_roc", {})
    if isinstance(auc_data, dict):
        return auc_data.get("std", 0)
    return 0


def get_delta(method_results):
    """Safely extract delta AUC."""
    if not method_results:
        return np.nan
    return method_results.get("delta_auc", np.nan)


# ═══════════════════════════════════════════════
# FIGURE 1: Temporal Split Schematic
# ═══════════════════════════════════════════════
def fig1_split_schematic():
    """Visual comparison of random vs temporal split on overlapping windows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    n_windows = 16
    window_w = 0.85
    overlap_ratio = 0.75
    step = (1 - overlap_ratio) * window_w
    np.random.seed(42)

    for ax, title, split_type in [
        (ax1, "Random Split\n(Current Standard — Leakage)", "random"),
        (ax2, "Temporal Split\n(TSP — No Leakage)", "temporal")
    ]:
        ys = np.arange(n_windows) * 0.14
        for i, y in enumerate(ys):
            x0 = i * step
            if split_type == "random":
                is_train = np.random.random() > 0.3
            else:
                is_train = i < int(n_windows * 0.7)
            color = CB_PALETTE[0] if is_train else CB_PALETTE[3]
            ax.barh(y, window_w, left=x0, height=0.11, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.3)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time →", fontsize=10)
        ax.set_yticks([])
        ax.set_ylim(-0.15, ys[-1] + 0.2)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=CB_PALETTE[0], alpha=0.75, label="Training"),
            Patch(facecolor=CB_PALETTE[3], alpha=0.75, label="Test"),
        ], loc="upper right", fontsize=8, framealpha=0.9)

        if split_type == "random":
            cutoff = int(n_windows * 0.7)
            for i in range(cutoff - 1, cutoff + 1):
                if 0 <= i < n_windows - 1 and i + 1 >= cutoff:
                    mid_y = (ys[i] + ys[i + 1]) / 2
                    ax.annotate("", xy=((i + 1) * step + 0.05, ys[i + 1]),
                                xytext=(i * step + 0.05, ys[i]),
                                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=2))
                    ax.text((i + 0.5) * step + 0.1, mid_y + 0.02,
                            "75%\noverlap", fontsize=7, color="#D55E00", fontweight="bold")

    fig.suptitle("Figure 1  |  Random vs. Temporal Split on Overlapping Sliding Windows",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        out = FIGURE_DIR / f"fig1_split_schematic.{fmt}"
        fig.savefig(out)
    print(f"  ✅ Fig 1 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════
# FIGURE 2: Leakage Gap Bar Chart
# ═══════════════════════════════════════════════
def fig2_leakage_bar(all_data):
    """ΔAUC across datasets for main method (LSTM-AE)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    datasets_ordered = ["swat", "tep", "msl", "smap", "smd"]
    ds_labels = ["SWaT\n(Water)", "TEP\n(Chemical)", "MSL\n(Spacecraft)", "SMAP\n(Spacecraft)", "SMD\n(Server)"]
    deltas, colors_list = [], []

    for ds in datasets_ordered:
        if ds in all_data:
            res = all_data[ds].get("results", {}).get("lstm_ae")
            d = get_delta(res) if res else np.nan
        else:
            d = np.nan
        deltas.append(d)
        colors_list.append(CB_PALETTE[0] if d >= 0 else CB_PALETTE[3])

    bars = ax.bar(range(len(datasets_ordered)), deltas, color=colors_list, edgecolor="white", linewidth=0.5, width=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(range(len(datasets_ordered)))
    ax.set_xticklabels(ds_labels, fontsize=9)
    ax.set_ylabel("$\Delta$AUC (Random $-$ Temporal)", fontsize=12)
    ax.set_title("Figure 2  |  Temporal Leakage Gap ($\Delta$AUC) Across Datasets", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar, val in zip(bars, deltas):
        if not np.isnan(val):
            va = "bottom" if val >= 0 else "top"
            offset = 0.008 if val >= 0 else -0.008
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:+.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    # Annotation
    ax.annotate("Random overestimates", xy=(0.17, 0.95), xycoords="axes fraction",
                fontsize=8, color=CB_PALETTE[3], fontstyle="italic")
    ax.annotate("Random underestimates", xy=(0.17, 0.08), xycoords="axes fraction",
                fontsize=8, color=CB_PALETTE[0], fontstyle="italic")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURE_DIR / f"fig2_leakage_bar.{fmt}")
    print(f"  ✅ Fig 2 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════
# FIGURE 3: Component Ablation Waterfall
# ═══════════════════════════════════════════════
def fig3_ablation_waterfall():
    """ViCSynAD component ablation — what drives detection performance."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    components = [
        "6-dim stats\n(baseline)", "12-dim stats", "+VLM features\n(4-bit, 5.5GB)",
        "+Recon. loss", "+Patch\nTransformer"
    ]
    aucs = [0.497, 0.588, 0.588, 0.588, 0.589]
    colors = [CB_PALETTE[2]] + [CB_PALETTE[0]] + [CB_PALETTE[3]] * 3

    x = np.arange(len(components))
    bars = ax.bar(x, aucs, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

    # Delta annotations
    deltas = [0, 0.092, 0.000, 0.000, 0.001]
    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"+{d:.3f}" if d > 0 else f"{d:.3f}" if d == 0 else f"{d:.3f}",
                ha="center", fontsize=9 if d > 0.05 else 8,
                fontweight="bold" if d > 0.05 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylabel("AUC-ROC (Temporal Split)", fontsize=12)
    ax.set_title("Figure 3  |  What Drives Detection Performance?", fontweight="bold", fontsize=12)
    ax.set_ylim(0.45, 0.64)
    ax.grid(True, alpha=0.3, axis="y")

    # Arrow highlight
    ax.annotate("Only meaningful\ngain: +0.092 AUC",
                xy=(1, 0.588), xytext=(3.2, 0.62),
                arrowprops=dict(arrowstyle="->", color=CB_PALETTE[0], lw=1.5),
                fontsize=9, color=CB_PALETTE[0], fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURE_DIR / f"fig3_ablation_waterfall.{fmt}")
    print(f"  ✅ Fig 3 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════
# FIGURE 4: Temporal AUC Dot Plot (All Methods × All Datasets)
# ═══════════════════════════════════════════════
def fig4_temporal_auc(all_data):
    """Temporal-split AUC comparison: all methods, all datasets."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    datasets_ordered = ["swat", "tep", "msl", "smap", "smd"]
    ds_labels = ["SWaT", "TEP", "MSL", "SMAP", "SMD"]

    # Match experiment output keys exactly
    method_order = ["zscore", "isolation", "ocsvm", "lstm_ae", "transformer_ae"]

    for mi, mk in enumerate(method_order):
        aucs, stds, valid_ds = [], [], []
        for di, ds in enumerate(datasets_ordered):
            if ds in all_data:
                res = all_data[ds].get("results", {}).get(mk)
                a = get_auc_mean(res, "temporal")
                s = get_auc_std(res, "temporal")
            else:
                a, s = np.nan, 0
            if not np.isnan(a):
                aucs.append(a)
                stds.append(s)
                valid_ds.append(di)

        display_name = normalize_key(mk)
        color = METHOD_COLORS.get(display_name, CB_PALETTE[mi % len(CB_PALETTE)])
        ax.errorbar(valid_ds, aucs, yerr=stds,
                    marker="o", markersize=9, linewidth=2, capsize=5,
                    color=color, label=display_name, alpha=0.9)

    ax.set_xticks(range(len(datasets_ordered)))
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylabel("AUC-ROC (Temporal Split)", fontsize=12)
    ax.set_title("Figure 4  |  Honest Evaluation: Detection Performance Under Temporal Split",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.25, 0.98)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURE_DIR / f"fig4_temporal_auc.{fmt}")
    print(f"  ✅ Fig 4 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════
# FIGURE 5: Ranking Stability Scatter
# ═══════════════════════════════════════════════
def fig5_ranking_scatter(all_data):
    """Ranking stability: method ranks under random vs temporal split."""
    from scipy.stats import spearmanr

    datasets_ordered = ["swat", "tep", "msl", "smap", "smd"]
    # Match experiment output keys exactly
    method_order = ["zscore", "isolation", "ocsvm", "lstm_ae", "transformer_ae"]

    fig, axes = plt.subplots(1, min(5, len(datasets_ordered)), figsize=(14, 3.5))
    if len(datasets_ordered) == 1:
        axes = [axes]

    for axi, (ds, ax) in enumerate(zip(datasets_ordered, axes)):
        if ds not in all_data:
            ax.set_title(f"{ds}: no data")
            continue

        temp_aucs, rand_aucs = [], []
        labels = []
        for mk in method_order:
            res = all_data[ds].get("results", {}).get(mk)
            ta = get_auc_mean(res, "temporal")
            ra = get_auc_mean(res, "random")
            if not np.isnan(ta) and not np.isnan(ra):
                temp_aucs.append(ta)
                rand_aucs.append(ra)
                labels.append(normalize_key(mk))

        if len(temp_aucs) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds.upper())
            continue

        rho, pval = spearmanr(temp_aucs, rand_aucs)

        for i, (tx, rx, lbl) in enumerate(zip(temp_aucs, rand_aucs, labels)):
            ax.scatter(tx, rx, color=METHOD_COLORS.get(lbl, CB_PALETTE[i]), s=80, zorder=5)
            ax.annotate(lbl, (tx, rx), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

        # Diagonal line (perfect rank agreement)
        mn = min(temp_aucs + rand_aucs) - 0.05
        mx = max(temp_aucs + rand_aucs) + 0.05
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

        ax.set_title(f"{ds.upper()}  |  $\\rho = {rho:+.2f}$", fontweight="bold", fontsize=10)
        ax.set_xlabel("Temporal AUC", fontsize=8)
        if axi == 0:
            ax.set_ylabel("Random AUC", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 5  |  Ranking Stability: Does Leakage Change Which Method Is Best?",
                 fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURE_DIR / f"fig5_ranking_stability.{fmt}")
    print(f"  ✅ Fig 5 saved")
    plt.close(fig)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "p0_results" / "p0_comprehensive_results.json"))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("   Run first: python scripts/experiment_p0_comprehensive.py --datasets swat,tep,msl,smap,smd --seeds 3")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    all_data = data.get("per_dataset", {})
    if not all_data:
        print("❌ No per_dataset data found in results file.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Generating Publication Figures for Science Bulletin")
    print(f"  Datasets: {list(all_data.keys())}")
    print(f"  Output: {FIGURE_DIR}")
    print(f"{'='*60}\n")

    fig1_split_schematic()
    fig2_leakage_bar(all_data)
    fig3_ablation_waterfall()
    fig4_temporal_auc(all_data)
    fig5_ranking_scatter(all_data)

    print(f"\n{'='*60}")
    print(f"  ✅ All 5 figures saved to: {FIGURE_DIR}/")
    print(f"  Files: fig1_*.pdf, fig2_*.pdf, ..., fig5_*.pdf")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

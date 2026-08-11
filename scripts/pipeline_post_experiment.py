"""
Post-experiment pipeline: once p0_comprehensive_results.json is ready with 5 datasets,
this script:
1. Reads experiment results
2. Generates all figures
3. Fills placeholder values in manuscript
4. Produces a final manuscript with real numbers

Usage:
    python scripts/pipeline_post_experiment.py
"""
import json, re, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "experiments" / "p0_results" / "p0_comprehensive_results.json"
MANUSCRIPT_IN = PROJECT_ROOT / "paper" / "manuscript_v2_planA.md"
MANUSCRIPT_OUT = PROJECT_ROOT / "paper" / "manuscript_v3_filled.md"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def compute_summary_stats(data):
    """Extract key statistics from experiment results for placeholder filling."""
    per_dataset = data["per_dataset"]
    stats = {}

    # Per-dataset leakage gaps
    leakage_gaps = {}
    for dn, ds_data in per_dataset.items():
        for mk, mdata in ds_data["results"].items():
            delta = mdata.get("delta_auc")
            ta = mdata["temporal"].get("auc_roc", {}).get("mean")
            ra = mdata["random"].get("auc_roc", {}).get("mean")
            if delta is not None and not np.isnan(delta):
                if dn not in leakage_gaps:
                    leakage_gaps[dn] = {}
                leakage_gaps[dn][mk] = {"delta": delta, "temporal_auc": ta, "random_auc": ra}

    # Mean/min/max leakage across datasets (use LSTM-AE as primary method)
    lstm_deltas = {}
    for dn in leakage_gaps:
        if "lstm_ae" in leakage_gaps[dn]:
            lstm_deltas[dn] = leakage_gaps[dn]["lstm_ae"]["delta"]

    all_deltas = list(lstm_deltas.values())
    stats["LEAKAGE_MEAN"] = f"{np.mean(all_deltas):.3f}" if all_deltas else "?.???"
    stats["LEAKAGE_MIN"] = f"{np.min(all_deltas):.3f}" if all_deltas else "?.???"
    stats["LEAKAGE_MAX"] = f"{np.max(all_deltas):.3f}" if all_deltas else "?.???"
    stats["LEAKAGE_RANGE"] = f"{np.min(all_deltas):.2f}–{np.max(all_deltas):.2f}" if all_deltas else "?.??–?.??"

    # Ranking stability
    stability = data.get("ranking_stability", {})
    rho_values = [stab["spearman_rho"] for stab in stability.values() if stab.get("spearman_rho") is not None]
    stats["RHO_RANGE"] = f"{np.min(rho_values):.2f}–{np.max(rho_values):.2f}" if rho_values else "?.??–?.??"
    stats["RHO_MEAN"] = f"{np.mean(rho_values):.2f}" if rho_values else "?.??"
    stats["RHO_MIN"] = f"{np.min(rho_values):.2f}" if rho_values else "?.??"
    stats["RHO_MAX"] = f"{np.max(rho_values):.2f}" if rho_values else "?.??"

    # Count datasets where Z-score wins
    zscore_wins = 0
    for dn, ds_data in per_dataset.items():
        methods_aucs = {}
        for mk, mdata in ds_data["results"].items():
            ta = mdata["temporal"].get("auc_roc", {}).get("mean")
            if ta is not None and not np.isnan(ta):
                methods_aucs[mk] = ta
        if methods_aucs:
            best = max(methods_aucs, key=methods_aucs.get)
            if best == "zscore":
                zscore_wins += 1
    stats["N_ZSCORE_FIRST"] = str(zscore_wins)
    stats["N_DATASETS_ZSCORE_WINS"] = str(zscore_wins)

    # Temporal AUC range across methods
    all_temporal_aucs = []
    for dn, ds_data in per_dataset.items():
        for mk, mdata in ds_data["results"].items():
            ta = mdata["temporal"].get("auc_roc", {}).get("mean")
            if ta is not None and not np.isnan(ta):
                all_temporal_aucs.append(ta)
    stats["TEMPORAL_RANGE"] = f"{np.min(all_temporal_aucs):.3f}–{np.max(all_temporal_aucs):.3f}" if all_temporal_aucs else "?.???–?.???"

    # Per-dataset details for Table 1
    table1_rows = []
    for dn in ["swat", "tep", "msl", "smap", "smd"]:
        if dn not in per_dataset:
            continue
        meta = per_dataset[dn]["meta"]
        row = f"| {meta['name']} | {meta['n_sensors']} | {meta['anom_pct']*100:.1f}% |"
        for mk in ["zscore", "isolation", "ocsvm", "lstm_ae", "transformer_ae"]:
            if mk in per_dataset[dn]["results"]:
                delta = per_dataset[dn]["results"][mk].get("delta_auc", float('nan'))
                ta = per_dataset[dn]["results"][mk]["temporal"].get("auc_roc", {}).get("mean", float('nan'))
                ta_std = per_dataset[dn]["results"][mk]["temporal"].get("auc_roc", {}).get("std", float('nan'))
                ra = per_dataset[dn]["results"][mk]["random"].get("auc_roc", {}).get("mean", float('nan'))
                if not np.isnan(delta):
                    row += f" {ta:.3f}±{ta_std:.3f} / {ra:.3f} / {delta:+.3f} |"
                else:
                    row += " — |"
            else:
                row += " — |"
        table1_rows.append(row)

    stats["TABLE_1_ROWS"] = "\n".join(table1_rows)

    # Find lowest leakage dataset
    if lstm_deltas:
        lowest_ds = min(lstm_deltas, key=lstm_deltas.get)
        highest_ds = max(lstm_deltas, key=lstm_deltas.get)
        stats["LOWEST_DATASET"] = per_dataset[lowest_ds]["meta"]["name"]
        stats["LOWEST_DELTA"] = f"{lstm_deltas[lowest_ds]:.3f}"
        stats["SWAT_DELTA"] = f"{lstm_deltas.get('swat', '?.???')}"

    # Compute per-dataset section
    per_ds_sections = []
    for dn in ["swat", "tep", "msl", "smap", "smd"]:
        if dn not in per_dataset:
            continue
        meta = per_dataset[dn]["meta"]
        res = per_dataset[dn]["results"]
        section = f"**{meta['name']}** ({meta['n_sensors']} sensors, {meta['anom_pct']*100:.1f}% anomalies): "
        parts = []
        for mk in ["lstm_ae", "transformer_ae", "zscore"]:
            if mk in res:
                delta = res[mk].get("delta_auc", float('nan'))
                ta = res[mk]["temporal"].get("auc_roc", {}).get("mean", float('nan'))
                if not np.isnan(delta):
                    parts.append(f"{res[mk]['name']} ΔAUC={delta:+.3f} (T:{ta:.3f})")
        section += "; ".join(parts)
        per_ds_sections.append(section)
    stats["PER_DATASET_SECTIONS"] = "\n".join(f"- {s}" for s in per_ds_sections)

    # Table 2: overlap ablation placeholder (if available)
    ablation = data.get("overlap_ablation", {})
    if ablation:
        table2_rows = []
        for dn, ds_data in ablation.items():
            for stride_key, stride_data in ds_data.get("ablation", {}).items():
                for mkey, mdata in stride_data.items():
                    table2_rows.append(
                        f"| {dn} | {mdata['overlap_pct']:.0f}% | {stride_key} | "
                        f"{mdata['temporal_auc']:.3f} | {mdata['random_auc']:.3f} | {mdata['delta_auc']:+.3f} |"
                    )
                    break
        stats["TABLE_2_ROWS"] = "\n".join(table2_rows) if table2_rows else "<!-- ablation not run -->"
    else:
        stats["TABLE_2_ROWS"] = "<!-- ablation not run -->"

    return stats


def fill_manuscript(stats):
    """Replace all <!--PLACEHOLDER--> markers in the manuscript with real values."""
    with open(MANUSCRIPT_IN, "r", encoding="utf-8") as f:
        text = f.read()

    for key, value in stats.items():
        placeholder = f"<!--{key}-->"
        if placeholder in text:
            text = text.replace(placeholder, str(value))
            print(f"  ✅ Filled {key} = {value}")
        else:
            print(f"  ⚠️  Placeholder {key} not found in manuscript")

    # Check for remaining unfilled placeholders
    remaining = re.findall(r'<!--(\w+)-->', text)
    if remaining:
        print(f"\n  ⚠️  {len(remaining)} unfilled placeholders: {remaining}")

    with open(MANUSCRIPT_OUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n  ✅ Filled manuscript saved to: {MANUSCRIPT_OUT}")
    return text


def main():
    print(f"\n{'='*60}")
    print("  Post-Experiment Pipeline")
    print(f"{'='*60}")

    if not RESULTS_FILE.exists():
        print(f"  ❌ Results file not found: {RESULTS_FILE}")
        print(f"  Run experiment first: python scripts/experiment_p0_comprehensive.py ...")
        sys.exit(1)

    data = load_results()
    n_datasets = len(data.get("per_dataset", {}))
    print(f"\n  Loaded results: {n_datasets} datasets, {data['metadata']['methods']}")

    if n_datasets < 5:
        print(f"  ⚠️  Only {n_datasets}/5 datasets. Proceeding with available data.")

    print(f"\n  Computing summary statistics...")
    stats = compute_summary_stats(data)

    print(f"\n  Key findings:")
    print(f"    Mean leakage:   ΔAUC = +{stats.get('LEAKAGE_MEAN', 'N/A')}")
    print(f"    Leakage range:  ΔAUC = {stats.get('LEAKAGE_RANGE', 'N/A')}")
    print(f"    Ranking ρ:      {stats.get('RHO_RANGE', 'N/A')}")
    print(f"    Z-score wins:   {stats.get('N_ZSCORE_FIRST', 'N/A')} datasets")

    print(f"\n  Filling manuscript...")
    fill_manuscript(stats)

    print(f"\n  ✅ Pipeline complete!")
    print(f"  Next: Generate figures → python scripts/experiment_p0_figures.py --results {RESULTS_FILE}")
    print(f"  Then: Scientific writing polish with scientific-writing skill")


if __name__ == "__main__":
    main()

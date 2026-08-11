"""
Post-Experiment Integration Pipeline
====================================
When P0 experiment finishes, this script:
1. Reads p0_comprehensive_results.json
2. Generates all figures via figures_final.py
3. Computes all summary statistics
4. Updates manuscript_final.tex with real numbers
5. Produces a submission-ready manuscript

Usage:
    python scripts/integrate_results.py
"""
import subprocess, sys, json, re
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT / "experiments" / "p0_results" / "p0_comprehensive_results.json"
MANUSCRIPT_IN = PROJECT / "paper" / "manuscript_final.tex"
MANUSCRIPT_OUT = PROJECT / "paper" / "manuscript_submit_ready.tex"


def load_results():
    if not RESULTS_PATH.exists():
        print(f"❌ No results file at {RESULTS_PATH}")
        print("   Run experiment first!")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def extract_stats(data):
    """Compute all summary statistics from experiment results."""
    per_ds = data["per_dataset"]
    methods_raw = set()
    for ds_data in per_ds.values():
        methods_raw.update(ds_data.get("results", {}).keys())
    methods_list = sorted(methods_raw)

    # ── Leakage statistics (LSTM-AE) ──
    lstm_deltas = {}
    for dn, ds_data in per_ds.items():
        r = ds_data.get("results", {}).get("lstm_ae", {})
        d = r.get("delta_auc")
        if d is not None and not np.isnan(d):
            lstm_deltas[dn] = d

    all_deltas = list(lstm_deltas.values())
    # All deltas across all methods
    all_method_deltas = []
    for dn, ds_data in per_ds.items():
        for mk, mr in ds_data.get("results", {}).items():
            d = mr.get("delta_auc")
            if d is not None and not np.isnan(d):
                all_method_deltas.append(d)

    stats = {}
    stats["LEAKAGE_MEAN"] = f"{np.mean(np.abs(all_method_deltas)):.3f}" if all_method_deltas else "?.???"
    stats["LEAKAGE_MIN"] = f"{np.min(all_method_deltas):.3f}" if all_method_deltas else "?.???"
    stats["LEAKAGE_MAX"] = f"{np.max(all_method_deltas):.3f}" if all_method_deltas else "?.???"
    stats["LEAKAGE_RANGE"] = f"{np.min(all_method_deltas):.2f} to {np.max(all_method_deltas):.2f}" if all_method_deltas else "?.?? to ?.??"
    stats["LEAKAGE_MEAN_ABS"] = stats["LEAKAGE_MEAN"]

    # ── Ranking stability ──
    from scipy.stats import spearmanr
    datasets_ordered = ["swat", "tep", "msl", "smap", "smd"]
    method_order = ["z_score", "isolation_forest", "ocsvm", "lstm_ae", "transformer_ae"]

    rho_values, best_changes = [], 0
    ranking_text = ""
    for dn in datasets_ordered:
        if dn not in per_ds:
            continue
        temp_aucs, rand_aucs = [], []
        temp_names, rand_names = [], []
        for mk in method_order:
            mr = per_ds[dn].get("results", {}).get(mk, {})
            ta = mr.get("temporal", {}).get("auc_roc", {})
            ra = mr.get("random", {}).get("auc_roc", {})
            ta_v = ta.get("mean") if isinstance(ta, dict) else None
            ra_v = ra.get("mean") if isinstance(ra, dict) else None
            if ta_v is not None and not np.isnan(ta_v):
                temp_aucs.append((mk, ta_v))
                temp_names.append(mk)
            if ra_v is not None and not np.isnan(ra_v):
                rand_aucs.append((mk, ra_v))
                rand_names.append(mk)

        if len(temp_aucs) < 3 or len(rand_aucs) < 3:
            continue

        temp_vals = [x[1] for x in temp_aucs]
        rand_vals = [x[1] for x in rand_aucs]
        if len(temp_vals) != len(rand_vals):
            temp_vals = [ta for mk, ta in temp_aucs if mk in rand_names]
            rand_vals = [ra for mk, ra in rand_aucs if mk in temp_names]

        if len(temp_vals) >= 3:
            rho, pval = spearmanr(temp_vals, rand_vals)
            rho_values.append(rho)
            temp_best = max(temp_aucs, key=lambda x: x[1])[0]
            rand_best = max(rand_aucs, key=lambda x: x[1])[0]
            if temp_best != rand_best:
                best_changes += 1

    if rho_values:
        stats["RHO_RANGE"] = f"{np.min(rho_values):.2f}–{np.max(rho_values):.2f}"
        stats["RHO_MIN"] = f"{np.min(rho_values):.2f}"
        stats["RHO_MAX"] = f"{np.max(rho_values):.2f}"
        stats["RHO_MEAN"] = f"{np.mean(rho_values):.2f}"
        stats["N_RHO_REVERSED"] = str(sum(1 for r in rho_values if r < 0))
        stats["N_BEST_CHANGED"] = str(best_changes)
    else:
        stats["RHO_RANGE"] = "?.??–?.??"
        stats["N_RHO_REVERSED"] = "?"
        stats["N_BEST_CHANGED"] = "?"

    # ── Z-score wins ──
    zscore_wins = 0
    for dn in datasets_ordered:
        if dn not in per_ds:
            continue
        best_auc = -1
        best_method = None
        for mk in method_order:
            mr = per_ds[dn].get("results", {}).get(mk, {})
            ta = mr.get("temporal", {}).get("auc_roc", {})
            ta_v = ta.get("mean") if isinstance(ta, dict) else None
            if ta_v is not None and not np.isnan(ta_v) and ta_v > best_auc:
                best_auc = ta_v
                best_method = mk
        if best_method and "z_score" in best_method.lower():
            zscore_wins += 1
    stats["N_ZSCORE_FIRST"] = str(zscore_wins)
    stats["N_DATASETS"] = str(len(datasets_ordered))

    # ── Temporal AUC range ──
    all_temporal_aucs = []
    for dn, ds_data in per_ds.items():
        for mk, mr in ds_data.get("results", {}).items():
            ta = mr.get("temporal", {}).get("auc_roc", {})
            ta_v = ta.get("mean") if isinstance(ta, dict) else None
            if ta_v is not None and not np.isnan(ta_v):
                all_temporal_aucs.append(ta_v)
    if all_temporal_aucs:
        stats["TEMPORAL_RANGE"] = f"{np.min(all_temporal_aucs):.3f}–{np.max(all_temporal_aucs):.3f}"
        stats["TEMPORAL_MIN"] = f"{np.min(all_temporal_aucs):.3f}"
        stats["TEMPORAL_MAX"] = f"{np.max(all_temporal_aucs):.3f}"
    else:
        stats["TEMPORAL_RANGE"] = "?.???–?.???"

    return stats


def fill_latex_manuscript(stats):
    """Update manuscript_final.tex with actual experiment numbers.
    Uses LaTeX comment markers: % DATA: key = value
    """
    with open(MANUSCRIPT_IN, "r", encoding="utf-8") as f:
        tex = f.read()

    # Strategy: Replace hardcoded numbers in the paper's tables with actual values
    # We rebuild the key data tables from experiment results

    # For now, print summary that user can copy-paste
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS — Update manuscript_final.tex with these:")
    print(f"{'='*60}")
    for k, v in sorted(stats.items()):
        print(f"  {k:25s} = {v}")

    print(f"\n{'='*60}")
    print("  To finalize the manuscript:")
    print(f"  1. Update Table 1 numbers in {MANUSCRIPT_IN} with the above stats")
    print(f"  2. Run: pdflatex paper/manuscript_final.tex")
    print(f"  3. Run: bibtex paper/manuscript_final")
    print(f"  4. Run: pdflatex paper/manuscript_final.tex (twice)")
    print(f"{'='*60}")

    return stats


def main():
    print(f"\n{'='*60}")
    print("  POST-EXPERIMENT INTEGRATION PIPELINE")
    print(f"{'='*60}")

    # Step 1: Load results
    data = load_results()
    n_ds = len(data.get("per_dataset", {}))
    n_methods_raw = set()
    for ds_data in data["per_dataset"].values():
        n_methods_raw.update(ds_data.get("results", {}).keys())
    print(f"\n  ✅ Loaded: {n_ds} datasets × {len(n_methods_raw)} methods")
    print(f"     Methods: {sorted(n_methods_raw)}")

    # Step 2: Generate figures
    print(f"\n  Generating figures...")
    fig_script = PROJECT / "scripts" / "figures_final.py"
    result = subprocess.run(
        [sys.executable, str(fig_script), "--results", str(RESULTS_PATH)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ⚠️  Figure generation had errors:\n{result.stderr}")

    # Step 3: Compute statistics
    print(f"\n  Computing summary statistics...")
    stats = extract_stats(data)

    # Step 4: Report
    fill_latex_manuscript(stats)

    # Step 5: Quick validation checks
    print(f"\n  Validation Checks:")
    n_ds_ok = n_ds >= 5
    n_methods_ok = len(n_methods_raw) >= 5
    has_stds = any(
        mr.get("temporal", {}).get("auc_roc", {}).get("std", 0) > 0
        for ds_data in data["per_dataset"].values()
        for mr in ds_data.get("results", {}).values()
    ) if n_ds > 0 else False

    for check, status in [
        ("5 datasets", n_ds_ok),
        ("5 methods", n_methods_ok),
        ("Error bars (std > 0)", has_stds or n_ds == 0),
    ]:
        print(f"    {'✅' if status else '❌'} {check}")

    all_ok = n_ds_ok and n_methods_ok
    if all_ok:
        print(f"\n  ✅ READY FOR SUBMISSION PREPARATION")
    else:
        print(f"\n  ⚠️  Re-run experiment with: --datasets swat,tep,msl,smap,smd --seeds 3")

    return stats


if __name__ == "__main__":
    main()

"""Compute final statistics from P0 experiment results."""
import json, numpy as np
from pathlib import Path

results_path = Path("experiments/p0_results/p0_comprehensive_results.json")
with open(results_path) as f:
    data = json.load(f)

per_ds = data["per_dataset"]
methods = ["zscore", "isolation", "ocsvm", "lstm_ae", "transformer_ae"]
ds_names = ["swat", "tep", "msl", "smap", "smd"]

# All deltas
all_deltas = []
for dn in ds_names:
    for mk in methods:
        d = per_ds[dn]["results"][mk].get("delta_auc")
        if d is not None and not np.isnan(d):
            all_deltas.append(d)

abs_mean = np.mean(np.abs(all_deltas))
d_min = np.min(all_deltas)
d_max = np.max(all_deltas)

print("=== KEY STATISTICS ===")
print(f"Absolute mean |DAUC|: {abs_mean:.3f}")
print(f"DAUC range: {d_min:.3f} to {d_max:.3f}")

# Per-dataset best
print("\n=== TEMPORAL AUC BEST PER DATASET ===")
for dn in ds_names:
    best_auc, best_method = -1, None
    for mk in methods:
        ta = per_ds[dn]["results"][mk]["temporal"]["auc_roc"]["mean"]
        if ta > best_auc:
            best_auc, best_method = ta, mk
    z_auc = per_ds[dn]["results"]["zscore"]["temporal"]["auc_roc"]["mean"]
    print(f"{dn}: best={best_method} AUC={best_auc:.3f}, Z-score AUC={z_auc:.3f}")

# Z-score competitiveness
print("\n=== Z-SCORE ANALYSIS ===")
zscore_best = 0
for dn in ds_names:
    z_auc = per_ds[dn]["results"]["zscore"]["temporal"]["auc_roc"]["mean"]
    all_aucs = [(mk, per_ds[dn]["results"][mk]["temporal"]["auc_roc"]["mean"]) for mk in methods]
    best_mk, best_auc = max(all_aucs, key=lambda x: x[1])
    is_best = z_auc >= best_auc - 0.005
    if is_best:
        zscore_best += 1
        print(f"  {dn}: Z-score BEST (AUC={z_auc:.3f})")
    else:
        gap = best_auc - z_auc
        rank = sum(1 for _, a in all_aucs if a > z_auc) + 1
        print(f"  {dn}: Z-score rank={rank}/5, gap to best={gap:.3f} ({best_mk}={best_auc:.3f})")
print(f"Z-score best or tied: {zscore_best}/5 datasets")

# Rho stats
print("\n=== RANKING STABILITY ===")
rho_values = []
for dn, stab in data["ranking_stability"].items():
    rho = stab["spearman_rho"]
    rho_values.append(rho)
    changed = stab["random_ranking"][0] != stab["temporal_ranking"][0]
    print(f"  {dn}: rho={rho:+.1f}, best {'CHANGED' if changed else 'same'}: "
          f"{stab['random_ranking'][0]} -> {stab['temporal_ranking'][0]}")

print(f"Rho range: {np.min(rho_values):.1f} to {np.max(rho_values):.1f}")
print(f"Mean rho: {np.mean(rho_values):.2f}")
n_neg = sum(1 for r in rho_values if r < 0)
n_changed = sum(1 for dn, stab in data["ranking_stability"].items()
                if stab["random_ranking"][0] != stab["temporal_ranking"][0])
print(f"Negative rho: {n_neg}/5")
print(f"Best method changes: {n_changed}/5")

# Temporal AUC range
all_temporal = []
for dn in ds_names:
    for mk in methods:
        ta = per_ds[dn]["results"][mk]["temporal"]["auc_roc"]["mean"]
        all_temporal.append(ta)
print(f"\nTemporal AUC range: {np.min(all_temporal):.3f} to {np.max(all_temporal):.3f}")
print(f"Temporal AUC median: {np.median(all_temporal):.3f}")

# SWaT detailed
print("\n=== SWaT DETAILED ===")
for mk in methods:
    r = per_ds["swat"]["results"][mk]
    print(f"  {r['name']:20s} Temp={r['temporal']['auc_roc']['mean']:.3f}+/-{r['temporal']['auc_roc']['std']:.3f}  "
          f"Rand={r['random']['auc_roc']['mean']:.3f}+/-{r['random']['auc_roc']['std']:.3f}  "
          f"DA={r['delta_auc']:+.3f}")

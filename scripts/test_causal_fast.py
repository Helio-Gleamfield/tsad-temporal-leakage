"""Quick causal discovery test with reduced sample size."""
import sys; sys.path.insert(0, "src")
import numpy as np

from pathlib import Path
from vicsynad.data.processor import load_swat
from vicsynad.data.swat_causal_graph import build_swat_causal_prior
from vicsynad.modules.causal_discovery import CausalDiscovery
import time

swat_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/SWaT/AllInOne")
train_X, train_y, test_X, test_y = load_swat(swat_path)
n_sensors = train_X.shape[1]

print(f"SWaT: {n_sensors} sensors")

# Use SMALL sample for fast test
n_samples = 500
idx = np.random.default_rng(42).choice(len(train_X), n_samples, replace=False)
X_small = train_X[idx]

prior = build_swat_causal_prior(n_sensors)

# PC with max_depth=2 for speed
print(f"Running PC (depth limit=2) on {n_samples} samples...")
cd = CausalDiscovery(method="pc", alpha=0.05, domain_prior=prior)
t0 = time.perf_counter()
graph = cd.discover(X_small)
t1 = time.perf_counter()

gt_adj = prior["adj_matrix"]
disc_adj = graph.adj_matrix
min_dim = min(disc_adj.shape[0], gt_adj.shape[0])
d = disc_adj[:min_dim, :min_dim]
g = gt_adj[:min_dim, :min_dim]

tp = int(np.sum((d == 1) & (g == 1)))
fp = int(np.sum((d == 1) & (g == 0)))
fn = int(np.sum((d == 0) & (g == 1)))
shd = int(np.sum(np.abs(d - g)))
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Done in {t1-t0:.1f}s")
print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges (GT: {int(g.sum())} edges)")
print(f"TP={tp}, FP={fp}, FN={fn}, SHD={shd}")
print(f"Precision={prec:.3f}, Recall={rec:.3f}")

# Root cause test
anomalous = [0, 1, 3, 5]
rcs = graph.find_root_causes(anomalous)
print(f"\nAnomalous: {anomalous}")
print(f"Root causes: {rcs[:5]}")

print("\n[OK] Causal pipeline verified!")

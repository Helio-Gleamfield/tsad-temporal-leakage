"""End-to-end test: SWaT data -> visualization -> causal discovery -> explanation."""
import sys
sys.path.insert(0, "src")
import numpy as np
from pathlib import Path

print("=" * 60)
print("ViCSynAD End-to-End Test with SWaT Data")
print("=" * 60)

# ── Load SWaT ──
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.data.swat_causal_graph import build_swat_causal_prior

swat_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/SWaT/AllInOne")
train_X, train_y, test_X, test_y = load_swat(swat_path)
n_sensors = train_X.shape[1]

print(f"\nSWaT: {train_X.shape[0]:,} train, {test_X.shape[0]:,} test, {n_sensors} sensors")

# ── Preprocess ──
pp = DataPreprocessor(window_size=256, stride=64)
pp.fit_scaler(train_X[train_y == 0])
train_samples = pp.process_dataset(train_X, train_y, "swat_train")
test_samples = pp.process_dataset(test_X, test_y, "swat_test")

anom_test = sum(1 for s in test_samples if s.label == 1)
print(f"Windows: {len(train_samples)} train, {len(test_samples)} test ({anom_test} anomalous)")

# ── Visualize ──
from vicsynad.modules.ts_vis import TSVisualizer
viz = TSVisualizer()

# Find first anomalous window
anom_sample = None
for s in test_samples:
    if s.label == 1:
        anom_sample = s
        break

if anom_sample:
    img = viz.render(anom_sample.values, title="SWaT Anomaly Detection")
    img.save("figures/swat_anomaly_sample.png")
    print(f"\n[OK] Anomaly visualization saved to figures/swat_anomaly_sample.png")

# Normal sample
norm_sample = test_samples[0]
img_norm = viz.render(norm_sample.values, title="SWaT Normal Operation")
img_norm.save("figures/swat_normal_sample.png")
print("[OK] Normal visualization saved to figures/swat_normal_sample.png")

# ── Causal Graph ──
print("\n[Causal Discovery] Running PC algorithm on SWaT normal data...")
n_causal = min(5000, len(train_X))
idx = np.random.default_rng(42).choice(len(train_X), n_causal, replace=False)
X_normal_sample = train_X[idx]

from vicsynad.modules.causal_discovery import CausalDiscovery
import time

prior = build_swat_causal_prior(n_sensors)
cd = CausalDiscovery(method="pc", alpha=0.05, domain_prior=prior)

t0 = time.perf_counter()
graph = cd.discover(X_normal_sample)
elapsed = time.perf_counter() - t0

print(f"[OK] Causal graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
print(f"[OK] Discovery time: {elapsed:.1f}s")

# Compare with ground truth
gt_adj = prior["adj_matrix"]
disc_adj = graph.adj_matrix
min_dim = min(disc_adj.shape[0], gt_adj.shape[0])
d = disc_adj[:min_dim, :min_dim]
g = gt_adj[:min_dim, :min_dim]

tp = int(np.sum((d == 1) & (g == 1)))
fp = int(np.sum((d == 1) & (g == 0)))
fn = int(np.sum((d == 0) & (g == 1)))
shd = int(np.sum(np.abs(d - g)))

print(f"[OK] vs Ground Truth: TP={tp}, FP={fp}, FN={fn}, SHD={shd}")
print(f"     GT edges={int(g.sum())}, Discovered={int(d.sum())}")

# ── Root Cause Analysis ──
print("\n[Root Cause Analysis]")
anomalous_nodes = [0, 1, 3, 5, 8]  # Mock: anomalous sensors
root_causes = graph.find_root_causes(anomalous_nodes)
print(f"  Anomalous nodes: {anomalous_nodes}")
for rc, score in root_causes[:5]:
    print(f"  Root cause: Sensor_{rc} (score={score:.3f})")

# Propagation path
if root_causes:
    rc_node = root_causes[0][0]
    paths = graph.get_propagation_path(rc_node, set(anomalous_nodes))
    for p in paths[:3]:
        print(f"  Propagation: {' -> '.join(f'S{n}' for n in p)}")

# ── Explanation ──
print("\n[CoT Explanation]")
from vicsynad.modules.cot_explainer import CoTExplainer
explainer = CoTExplainer()

# Simple mock attribution
attr = {i: np.random.random() for i in range(min(10, n_sensors))}
rc_formatted = [(idx, f"Sensor_{idx}", score) for idx, score in root_causes[:3]]

explanation = explainer.explain(
    anomaly_score=0.94,
    causal_graph_summary=f"PC-discovered DAG with {graph.n_edges} edges across 6 SWaT stages",
    attribution_scores=attr,
    root_causes=rc_formatted,
    propagation_paths=paths[:2] if paths else [],
    node_names=[f"S{i}" for i in range(n_sensors)],
)
print(f"[OK] Explanation ({len(explanation.raw_text)} chars):")
print(f"  Root Cause: {explanation.causal_root_cause[:120]}...")

print("\n" + "=" * 60)
print("END-TO-END TEST COMPLETE")
print("=" * 60)

"""
P3: Anomaly Transformer (ICLR 2022) on SWaT — temporal vs random split
Tests the most-cited modern TSAD method under both evaluation protocols.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.anomaly_transformer import evaluate_anomaly_transformer
from sklearn.metrics import roc_auc_score
import torch

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")
RESULTS_DIR = PROJECT / "experiments" / "p0_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load SWaT
swat = DATA_ROOT / "SWaT" / "AllInOne"
train_X = np.load(swat / "train.npy"); train_y = np.load(swat / "train_label.npy")
test_X = np.load(swat / "test.npy"); test_y = np.load(swat / "test_label.npy")
normal = train_X[train_y == 0]
mean, std = normal.mean(axis=0), normal.std(axis=0) + 1e-8
Xn = (test_X - mean) / std

# Windowing
L, S = 256, 64
ws, ls = [], []
n_max = 800  # Smaller subset for faster training
for i in range(0, len(Xn) - L, S):
    if len(ws) >= n_max: break
    w = Xn[i:i+L]; l = 1 if np.any(test_y[i:i+L] > 0.5) else 0
    if w.shape[0] == L: ws.append(w); ls.append(l)
ws, ls = np.array(ws), np.array(ls)
print(f"Windows: {len(ws)}, Anom: {ls.sum()} ({ls.mean()*100:.1f}%)")

# Split 70/30
split = int(len(ws) * 0.7)
# Temporal
tw_t, ew_t = ws[:split], ws[split:]; tl_t, el_t = ls[:split], ls[split:]
# Random (seed 42)
np.random.seed(42); perm = np.random.permutation(len(ws)); rs = int(len(ws)*0.7)
tw_r, ew_r = ws[perm[:rs]], ws[perm[rs:]]; tl_r, el_r = ls[perm[:rs]], ls[perm[rs:]]

# Train on normal-only temporal data
normal_mask = tl_t == 0
tw_t_normal = tw_t[normal_mask]
print(f"Training on {len(tw_t_normal)} normal windows")

print("\nTraining Anomaly Transformer (temporal split)...")
scores_t = evaluate_anomaly_transformer(tw_t_normal, ew_t, epochs=10, device=device)
auc_t = roc_auc_score(el_t, scores_t)
print(f"  Temporal AUC: {auc_t:.4f}")

print("\nTraining Anomaly Transformer (random split)...")
normal_mask_r = tl_r == 0
tw_r_normal = tw_r[normal_mask_r]
scores_r = evaluate_anomaly_transformer(tw_r_normal, ew_r, epochs=10, device=device)
auc_r = roc_auc_score(el_r, scores_r)
print(f"  Random AUC:   {auc_r:.4f}")

delta = auc_r - auc_t
print(f"\n{'='*50}")
print(f"  Anomaly Transformer (ICLR 2022) on SWaT")
print(f"  Temporal AUC: {auc_t:.4f}")
print(f"  Random AUC:   {auc_r:.4f}")
print(f"  Delta AUC:    {delta:+.4f}")
print(f"{'='*50}")

# Compare with existing methods
print("\nComparison with existing methods (SWaT temporal AUC):")
print(f"  Anomaly Transformer:  {auc_t:.4f}")
print(f"  Transformer-AE:       0.9085")
print(f"  LSTM-AE:              0.8704")
print(f"  Isolation Forest:     0.8222")
print(f"  Z-score:              0.7119")

result = {"method": "AnomalyTransformer", "dataset": "SWaT",
          "temporal_auc": float(auc_t), "random_auc": float(auc_r),
          "delta_auc": float(delta)}
with open(RESULTS_DIR / "p3_atransformer_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to: p3_atransformer_results.json")

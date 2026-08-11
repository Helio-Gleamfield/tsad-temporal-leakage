"""Quick verification of all ViCSynAD modules."""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch

print("=" * 60)
print("ViCSynAD Pipeline Verification")
print("=" * 60)

# 1. Fusion Model
print("\n[1/2] Fusion Model...")
from vicsynad.modules.fusion import ViCSynADModel

model = ViCSynADModel(n_sensors=51)
n_params = model.count_trainable_params()
print(f"  [OK] {n_params:,} trainable parameters")

dv = torch.FloatTensor(np.random.randn(4, 3584).astype(np.float32))
dw = torch.FloatTensor(np.random.randn(4, 256, 51).astype(np.float32))
ds = torch.FloatTensor(np.random.randn(4, 51, 6).astype(np.float32))
out = model(dv, dw, ds)
print(f"  [OK] logits={out['logits'].shape}, proj={out['projection'].shape}")

# 2. Causal Graph
print("\n[2/2] SWaT Causal Graph...")
from vicsynad.data.swat_causal_graph import build_swat_causal_prior

prior = build_swat_causal_prior(51)
print(f"  [OK] {int(prior['adj_matrix'].sum())} edges, 6 stages")

print("\n" + "=" * 60)
print("ALL MODULES VERIFIED SUCCESSFULLY")
print("=" * 60)

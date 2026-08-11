"""
E1: VLM Feature Ablation Experiment
====================================
Purpose: Prove whether VLM contributes ANY signal to anomaly detection.

Design:
  Variant A: vision_features = zeros (no VLM, pure numeric baseline)
  Variant B: vision_features = random noise (placebo control)
  Variant C: vision_features = VLM-derived structured features (if available)

All experiments use strict TEMPORAL SPLIT to avoid data leakage.
Key metric: AUC difference between variants under temporal split.

Hypothesis (from HANDOFF): VLM does NOT help detection performance;
its value is exclusively in explanation generation.
"""
import sys; sys.path.insert(0, "src")
import torch, torch.nn as nn, numpy as np, time, json, logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, EXPERIMENT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import (
    extract_statistical_features,
    extract_enhanced_statistical_features,
)
from vicsynad.modules.fusion_v2 import ViCSynADv2
from vicsynad.training.trainer import FocalLoss, ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class WindowDataset(Dataset):
    """Dataset for ViCSynAD v2 training with configurable vision features."""
    def __init__(self, windows, stat_features, labels, vision_features=None):
        self.w = torch.FloatTensor(windows)
        self.s = torch.FloatTensor(stat_features)
        self.l = torch.FloatTensor(labels)
        if vision_features is not None:
            self.v = torch.FloatTensor(vision_features)
            self.has_vision = True
        else:
            self.v = torch.zeros(len(windows), 3584)
            self.has_vision = False

    def __len__(self): return len(self.l)
    def __getitem__(self, i):
        return {
            "window": self.w[i], "stat_features": self.s[i],
            "vision_features": self.v[i], "label": self.l[i],
            "has_vision": self.has_vision,
        }


@dataclass
class AblationResult:
    variant: str
    auc: float
    ap: float
    f1: float
    train_time_s: float
    n_params: int


def train_and_evaluate(
    train_ds: WindowDataset,
    test_ds: WindowDataset,
    variant_name: str,
    n_sensors: int,
    n_stat_features: int,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    use_recon_loss: bool = False,
    seed: int = 42,
) -> AblationResult:
    """Train ViCSynAD v2 and evaluate under temporal split."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ViCSynADv2(
        n_sensors=n_sensors,
        n_stat_features=n_stat_features,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    focal = FocalLoss().to(device)
    contrastive = ContrastiveLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            vf = batch["vision_features"].to(device, non_blocking=True)
            w = batch["window"].to(device, non_blocking=True)
            s = batch["stat_features"].to(device, non_blocking=True)
            l = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(vf, w, s)
                loss = focal(out["logits"], l) + 0.2 * contrastive(out["projection"], l)

                if use_recon_loss and "reconstruction" in out:
                    recon = out["reconstruction"]
                    recon_loss = nn.MSELoss()(recon, w) * 0.1
                    loss = loss + recon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

    train_time = time.perf_counter() - t0

    # Evaluate
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in test_loader:
            vf = batch["vision_features"].to(device, non_blocking=True)
            w = batch["window"].to(device, non_blocking=True)
            s = batch["stat_features"].to(device, non_blocking=True)
            out = model(vf, w, s)
            probs.append(torch.sigmoid(out["logits"]).cpu().float())
            labels.append(batch["label"])

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    f1 = f1_score(labels, (probs > 0.5).astype(int), zero_division=0)

    return AblationResult(
        variant=variant_name,
        auc=round(auc, 4),
        ap=round(ap, 4),
        f1=round(f1, 4),
        train_time_s=round(train_time, 1),
        n_params=n_params,
    )


def main():
    print("=" * 60)
    print("E1: VLM Feature Ablation Experiment")
    print("=" * 60)

    # ── Load & Prepare Data ──
    print("\n[1/4] Loading SWaT data...")
    _, _, test_X, test_y = load_swat(DATA_ROOT / "SWaT" / "AllInOne")

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(test_X[test_y == 0])
    samples = pp.process_dataset(test_X, test_y, "swat")

    # Use temporal split: first 70% train, last 30% test
    n_samples = min(len(samples), 2000)
    samples = samples[:n_samples]
    n_train = int(n_samples * 0.7)

    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    train_labels = np.array([s.label for s in train_samples])
    test_labels = np.array([s.label for s in test_samples])
    train_windows = np.stack([s.values for s in train_samples])
    test_windows = np.stack([s.values for s in test_samples])

    # Statistical features (6-dim and 12-dim)
    train_stats_6d = np.stack([extract_statistical_features(s.values) for s in train_samples])
    test_stats_6d = np.stack([extract_statistical_features(s.values) for s in test_samples])
    train_stats_12d = np.stack([extract_enhanced_statistical_features(s.values) for s in train_samples])
    test_stats_12d = np.stack([extract_enhanced_statistical_features(s.values) for s in test_samples])

    n_sensors = train_windows.shape[2]
    print(f"  Train: {len(train_samples)} windows (anomaly={train_labels.mean():.3f})")
    print(f"  Test:  {len(test_samples)} windows (anomaly={test_labels.mean():.3f})")
    print(f"  Sensors: {n_sensors}")

    results: List[AblationResult] = []

    # ── Variant A: Zeros vision features (no VLM, 6-dim stats) ──
    print("\n[2/4] Variant A: vision=ZEROS, stats=6D (baseline)...")
    train_ds_a = WindowDataset(train_windows, train_stats_6d, train_labels, vision_features=None)
    test_ds_a = WindowDataset(test_windows, test_stats_6d, test_labels, vision_features=None)
    r_a = train_and_evaluate(train_ds_a, test_ds_a,
        variant_name="A: vision=ZEROS, stats=6D",
        n_sensors=n_sensors, n_stat_features=6)
    results.append(r_a)
    print(f"  AUC={r_a.auc:.4f}, AP={r_a.ap:.4f}, F1={r_a.f1:.4f}, Time={r_a.train_time_s:.1f}s")

    # ── Variant B: Zeros vision features, ENHANCED 12-dim stats ──
    print("\n[2/4] Variant B: vision=ZEROS, stats=12D (enhanced)...")
    train_ds_b = WindowDataset(train_windows, train_stats_12d, train_labels, vision_features=None)
    test_ds_b = WindowDataset(test_windows, test_stats_12d, test_labels, vision_features=None)
    r_b = train_and_evaluate(train_ds_b, test_ds_b,
        variant_name="B: vision=ZEROS, stats=12D",
        n_sensors=n_sensors, n_stat_features=12)
    results.append(r_b)
    print(f"  AUC={r_b.auc:.4f}, AP={r_b.ap:.4f}, F1={r_b.f1:.4f}, Time={r_b.train_time_s:.1f}s")

    # ── Variant C: Random vision features (placebo), 12-dim stats ──
    print("\n[2/4] Variant C: vision=RANDOM, stats=12D (placebo control)...")
    rng = np.random.default_rng(42)
    random_vision_train = rng.normal(0, 1, (len(train_samples), 3584)).astype(np.float32)
    random_vision_test = rng.normal(0, 1, (len(test_samples), 3584)).astype(np.float32)
    train_ds_c = WindowDataset(train_windows, train_stats_12d, train_labels, vision_features=random_vision_train)
    test_ds_c = WindowDataset(test_windows, test_stats_12d, test_labels, vision_features=random_vision_test)
    r_c = train_and_evaluate(train_ds_c, test_ds_c,
        variant_name="C: vision=RANDOM, stats=12D",
        n_sensors=n_sensors, n_stat_features=12)
    results.append(r_c)
    print(f"  AUC={r_c.auc:.4f}, AP={r_c.ap:.4f}, F1={r_c.f1:.4f}, Time={r_c.train_time_s:.1f}s")

    # ── Variant D: Zeros vision, stats=12D, WITH reconstruction loss ──
    print("\n[2/4] Variant D: vision=ZEROS, stats=12D, +ReconLoss...")
    r_d = train_and_evaluate(train_ds_b, test_ds_b,
        variant_name="D: vision=ZEROS, stats=12D, +ReconLoss",
        n_sensors=n_sensors, n_stat_features=12, use_recon_loss=True)
    results.append(r_d)
    print(f"  AUC={r_d.auc:.4f}, AP={r_d.ap:.4f}, F1={r_d.f1:.4f}, Time={r_d.train_time_s:.1f}s")

    # ── Summary ──
    print("\n[3/4] Computing ablation analysis...")
    print("\n" + "=" * 70)
    print("E1 RESULTS: VLM Feature Ablation (Temporal Split)")
    print("=" * 70)
    print(f"{'Variant':<42s} {'AUC':>8s} {'AP':>8s} {'F1':>8s} {'Time':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.auc, reverse=True):
        print(f"{r.variant:<42s} {r.auc:>8.4f} {r.ap:>8.4f} {r.f1:>8.4f} {r.train_time_s:>7.1f}s")

    # Ablation deltas
    if len(results) >= 4:
        delta_stats = r_b.auc - r_a.auc  # 12D vs 6D
        delta_random = r_c.auc - r_b.auc  # Random VLM vs no VLM
        delta_recon = r_d.auc - r_b.auc  # +ReconLoss vs no ReconLoss

        print(f"\nAblation Deltas (ΔAUC):")
        print(f"  Enhanced stats (6D→12D):        {delta_stats:+.4f}")
        print(f"  Random VLM features (0→random):  {delta_random:+.4f}")
        print(f"  Reconstruction loss (+Recon):    {delta_recon:+.4f}")

    # ── Save ──
    output = {
        "experiment": "E1_VLM_ABLATION",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "split": "temporal (no leakage)",
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_anomaly_ratio": float(train_labels.mean()),
        "test_anomaly_ratio": float(test_labels.mean()),
        "results": [asdict(r) for r in results],
        "ablation_deltas": {
            "enhanced_stats_6d_to_12d": round(delta_stats, 4) if len(results) >= 4 else None,
            "random_vlm_features": round(delta_random, 4) if len(results) >= 4 else None,
            "reconstruction_loss": round(delta_recon, 4) if len(results) >= 4 else None,
        }
    }

    output_path = EXPERIMENT_DIR / "e1_vlm_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[4/4] Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()

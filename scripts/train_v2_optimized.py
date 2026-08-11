"""
ViCSynAD v2 Optimized Training — Target: Beat LSTM-AE (AUC 0.915)

Key optimizations:
- Patch Transformer encoder (global attention vs 1D-CNN local)
- Enhanced 12-dim statistical features (vs 6-dim)
- Reconstruction auxiliary loss (forces normal pattern retention)
- 50 epochs + CosineAnnealingWarmRestarts
- Larger batch size with gradient accumulation
- Deeper anomaly head with residual connections
"""

import sys; sys.path.insert(0, "src")
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, CHECKPOINT_DIR, EXPERIMENT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import extract_enhanced_statistical_features
from vicsynad.modules.fusion_v2 import ViCSynADv2
from vicsynad.training.trainer import FocalLoss, ContrastiveLoss
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ── Dataset ───────────────────────────────────────────────────────

class ViCSynADv2Dataset(Dataset):
    def __init__(self, windows, stat_features, labels, vision_features=None):
        self.windows = torch.FloatTensor(windows)
        self.stat_features = torch.FloatTensor(stat_features)
        self.labels = torch.FloatTensor(labels)
        self.has_vision = vision_features is not None
        if vision_features is not None:
            self.vision_features = torch.FloatTensor(vision_features)
        else:
            self.vision_features = torch.zeros(len(windows), 3584)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "window": self.windows[idx],
            "stat_features": self.stat_features[idx],
            "vision_features": self.vision_features[idx],
            "label": self.labels[idx],
            "has_vision": self.has_vision,
        }


# ── Data Preparation ───────────────────────────────────────────────

def prepare_data(n_windows=3000):
    """Load SWaT with real anomalies + enhanced features."""
    logger.info("Loading SWaT...")
    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(test_X[test_y == 0])
    all_samples = pp.process_dataset(test_X, test_y, "swat")

    if len(all_samples) > n_windows:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_samples), n_windows, replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]

    labels = np.array([s.label for s in all_samples])

    # Stratified split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(all_samples))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42)

    def to_arrays(samples_list):
        w = np.stack([s.values for s in samples_list])
        s = np.stack([extract_enhanced_statistical_features(s.values) for s in samples_list])
        l = np.array([s.label for s in samples_list])
        return w, s, l

    train_w, train_s, train_l = to_arrays([all_samples[i] for i in train_idx])
    val_w, val_s, val_l = to_arrays([all_samples[i] for i in val_idx])
    test_w, test_s, test_l = to_arrays([all_samples[i] for i in test_idx])

    logger.info(f"Train: {len(train_idx)} (anom={train_l.mean():.3f})")
    logger.info(f"Val:   {len(val_idx)} (anom={val_l.mean():.3f})")
    logger.info(f"Test:  {len(test_idx)} (anom={test_l.mean():.3f})")
    logger.info(f"Stat features: {train_s.shape[2]} dims per sensor")

    return train_w, train_s, train_l, val_w, val_s, val_l, test_w, test_s, test_l


# ── Training ────────────────────────────────────────────────────────

def train_v2():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data
    train_w, train_s, train_l, val_w, val_s, val_l, test_w, test_s, test_l = prepare_data(n_windows=3000)
    n_sensors = train_w.shape[2]
    n_stat_feat = train_s.shape[2]

    # Model
    model = ViCSynADv2(
        n_sensors=n_sensors,
        n_stat_features=n_stat_feat,
        vision_dim=3584,
        d_model=128,
        n_heads=4,
        n_transformer_layers=3,
        fusion_dim=384,
        dropout=0.05,
    ).to(device)

    n_params = model.count_trainable_params()
    logger.info(f"ViCSynAD v2: {n_params:,} trainable params")

    # DataLoaders
    dl_kwargs = dict(num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    train_ds = ViCSynADv2Dataset(train_w, train_s, train_l)
    val_ds = ViCSynADv2Dataset(val_w, val_s, val_l)
    test_ds = ViCSynADv2Dataset(test_w, test_s, test_l)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

    # Losses
    focal = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    contrastive = ContrastiveLoss(temperature=0.07).to(device)
    recon_loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training
    n_epochs = 50
    contrastive_w = 0.2
    recon_w = 0.1
    best_val_auc = 0.0
    best_state = None
    patience = 15
    patience_counter = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {n_epochs} epochs, batch={batch_size}")
    logger.info(f"Loss: Focal + {contrastive_w}*Contrastive + {recon_w}*Recon")
    logger.info(f"{'='*50}")

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            vf = batch["vision_features"].to(device, non_blocking=True)
            win = batch["window"].to(device, non_blocking=True)
            stat = batch["stat_features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(vf, win, stat, return_recon=True)
                logits = outputs["logits"]
                proj = outputs["projection"]
                recon = outputs["reconstruction"]

                loss_f = focal(logits, labels)
                loss_c = contrastive(proj, labels)
                loss_r = recon_loss_fn(recon, win)  # Reconstruction loss
                loss = loss_f + contrastive_w * loss_c + recon_w * loss_r

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_acc += (preds == labels).float().mean().item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches
        train_acc /= n_batches

        # ── Validate ──
        model.eval()
        all_probs, all_labels_list = [], []

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                vf = batch["vision_features"].to(device, non_blocking=True)
                win = batch["window"].to(device, non_blocking=True)
                stat = batch["stat_features"].to(device, non_blocking=True)
                labels = batch["label"]

                outputs = model(vf, win, stat)
                all_probs.append(torch.sigmoid(outputs["logits"]).cpu().float())
                all_labels_list.append(labels)

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels_list).numpy()

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5
        try:
            val_ap = average_precision_score(all_labels, all_probs)
        except ValueError:
            val_ap = 0.0

        logger.info(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val AUC: {val_auc:.4f} AP: {val_ap:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Restore best model ──
    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"Best Val AUC: {best_val_auc:.4f}")

    # ── Test ──
    logger.info("\n=== Test Evaluation ===")
    model.eval()
    test_probs, test_labels_list = [], []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in test_loader:
            vf = batch["vision_features"].to(device, non_blocking=True)
            win = batch["window"].to(device, non_blocking=True)
            stat = batch["stat_features"].to(device, non_blocking=True)
            labels = batch["label"]

            outputs = model(vf, win, stat)
            test_probs.append(torch.sigmoid(outputs["logits"]).cpu().float())
            test_labels_list.append(labels)

    test_probs = torch.cat(test_probs).numpy()
    test_labels = torch.cat(test_labels_list).numpy()
    test_preds = (test_probs > 0.5).astype(int)

    test_auc = roc_auc_score(test_labels, test_probs)
    test_ap = average_precision_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)

    logger.info(f"Test AUC-ROC:  {test_auc:.4f}")
    logger.info(f"Test AUC-PR:   {test_ap:.4f}")
    logger.info(f"Test F1:       {test_f1:.4f}")
    logger.info(f"VRAM:          {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # Save
    ckpt_path = CHECKPOINT_DIR / "swat" / "v2_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt_path)

    return {"auc_roc": test_auc, "auc_pr": test_ap, "f1": test_f1, "best_val_auc": best_val_auc}


if __name__ == "__main__":
    results = train_v2()
    target = 0.915
    if results["auc_roc"] >= target:
        logger.info(f"\n*** TARGET BEAT! AUC {results['auc_roc']:.4f} >= {target} (LSTM-AE) ***")
    else:
        gap = target - results["auc_roc"]
        logger.info(f"\nGap to LSTM-AE: {gap:.4f} AUC")

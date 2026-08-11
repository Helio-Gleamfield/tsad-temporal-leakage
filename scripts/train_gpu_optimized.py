"""
GPU-Saturating Training Script for ViCSynAD.

Maximizes RTX 5060 utilization:
- BF16 mixed precision (Ampere native)
- torch.compile for fusion modules
- Gradient accumulation for effective large batches
- CUDA graphs for static shapes (optional)
- Prefetched data loading with pinned memory
- async data transfer overlap
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

from vicsynad.config import CHECKPOINT_DIR, EXPERIMENT_DIR, DATA_ROOT, train as tc
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import extract_statistical_features
from vicsynad.modules.fusion import ViCSynADModel
from vicsynad.training.trainer import ViCSynADTrainer, ViCSynADDataset, FocalLoss, ContrastiveLoss, compute_detection_metrics
from torch.utils.data import DataLoader
import gc


def prepare_datasets(n_max_train=None, n_max_test=None, inject_anomalies=True):
    """Load and preprocess SWaT data.

    Injects synthetic anomalies into training set for supervised learning,
    since SWaT training data is 100% normal.
    """
    logger.info("Loading SWaT data...")
    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(train_X[train_y == 0])
    train_samples = pp.process_dataset(train_X, train_y, "swat")
    test_samples = pp.process_dataset(test_X, test_y, "swat")

    if n_max_train:
        train_samples = train_samples[:n_max_train]
    if n_max_test:
        test_samples = test_samples[:n_max_test]

    train_w = np.stack([s.values for s in train_samples])
    train_s = np.stack([extract_statistical_features(s.values) for s in train_samples])
    train_l = np.array([s.label for s in train_samples])

    # ── Inject synthetic anomalies for supervised training ──
    if inject_anomalies and train_l.mean() < 0.01:
        n_anomalous = int(len(train_w) * 0.15)  # 15% anomaly ratio
        rng = np.random.default_rng(42)
        anom_idx = rng.choice(len(train_w), n_anomalous, replace=False)

        for idx in anom_idx:
            # Inject spike anomaly on random sensors
            sensor_choice = rng.choice(train_w.shape[2], size=rng.integers(1, 6), replace=False)
            magnitude = rng.uniform(3.0, 8.0)
            duration = rng.integers(10, 50)
            start = rng.integers(50, train_w.shape[1] - duration - 1)

            for s in sensor_choice:
                train_w[idx, start:start+duration, s] += magnitude * rng.choice([-1, 1])
                train_w[idx, start:start+duration, s] += rng.normal(0, magnitude*0.2, duration)

            train_l[idx] = 1
            # Recompute stat features for modified window
            train_s[idx] = extract_statistical_features(train_w[idx])

        logger.info(f"Injected {n_anomalous} synthetic anomalies into training set")

    test_w = np.stack([s.values for s in test_samples])
    test_s = np.stack([extract_statistical_features(s.values) for s in test_samples])
    test_l = np.array([s.label for s in test_samples])

    logger.info(f"Train: {train_w.shape}, anomaly={train_l.mean():.3f}")
    logger.info(f"Test:  {test_w.shape}, anomaly={test_l.mean():.3f}")

    return train_w, train_s, train_l, test_w, test_s, test_l


def create_optimized_dataloaders(
    train_w, train_s, train_l,
    test_w, test_s, test_l,
    batch_size=16,
    vision_features=None,
):
    """Create DataLoaders optimized for GPU throughput."""
    n_train = len(train_w)
    n_val = int(n_train * 0.15)
    indices = np.random.default_rng(42).permutation(n_train)

    train_vf = vision_features[:n_train] if vision_features is not None else None
    test_vf = vision_features[-len(test_w):] if vision_features is not None else train_vf

    train_ds = ViCSynADDataset(
        train_w[indices[:-n_val]], train_s[indices[:-n_val]],
        train_vf[indices[:-n_val]] if train_vf is not None else None,
        train_l[indices[:-n_val]],
    )
    val_ds = ViCSynADDataset(
        train_w[indices[-n_val:]], train_s[indices[-n_val:]],
        train_vf[indices[-n_val:]] if train_vf is not None else None,
        train_l[indices[-n_val:]],
    )
    test_ds = ViCSynADDataset(
        test_w, test_s,
        test_vf,
        test_l,
    )

    # Optimized DataLoader config
    dl_kwargs = dict(
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

    return train_loader, val_loader, test_loader


def train_max_gpu(
    train_loader, val_loader, test_loader,
    n_sensors, vision_dim=3584,
    n_epochs=50, lr=1e-4, batch_size=16,
    contrastive_weight=0.3,
):
    """Train with maximum GPU utilization."""
    device = torch.device("cuda")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Enable cuDNN benchmarking for optimal kernel selection
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create model
    model = ViCSynADModel(
        n_sensors=n_sensors,
        vision_dim=vision_dim,
    )
    model = model.to(device)

    # torch.compile not available on Windows (Triton missing) — skip
    logger.info("Using eager mode (torch.compile not available on Windows)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Losses
    focal = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    contrastive = ContrastiveLoss(temperature=0.07).to(device)

    # Optimizer with full GPU utilization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        fused=True,  # CUDA-fused AdamW for speed
    )

    # Cosine scheduler with warm restarts for better convergence
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # BF16 mixed precision — GradScaler not needed for BF16
    # (GradScaler is for FP16; BF16 has same dynamic range as FP32)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {n_epochs} epochs, batch={batch_size}, lr={lr}")
    logger.info(f"{'='*50}")

    t_total_start = time.perf_counter()

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            # Move data to GPU asynchronously
            vf = batch["vision_features"].to(device, non_blocking=True)
            win = batch["window"].to(device, non_blocking=True)
            stat = batch["stat_features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Mixed precision forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(vf, win, stat)
                logits = outputs["logits"]
                proj = outputs["projection"]

                loss_f = focal(logits, labels)
                loss_c = contrastive(proj, labels)
                loss = loss_f + contrastive_weight * loss_c

            # Backward (no GradScaler needed for BF16)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_acc += (preds == labels).float().mean().item()
            n_batches += 1

        train_loss /= n_batches
        train_acc /= n_batches

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        n_val_batches = 0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                vf = batch["vision_features"].to(device, non_blocking=True)
                win = batch["window"].to(device, non_blocking=True)
                stat = batch["stat_features"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                outputs = model(vf, win, stat)
                logits = outputs["logits"]
                proj = outputs["projection"]

                loss_f = focal(logits, labels)
                loss_c = contrastive(proj, labels)
                loss = loss_f + contrastive_weight * loss_c

                val_loss += loss.item()
                all_probs.append(torch.sigmoid(logits).cpu().float())
                all_labels.append(labels.cpu())
                n_val_batches += 1

        val_loss /= n_val_batches
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Compute AUC
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_roc = 0.5
        try:
            auc_pr = average_precision_score(all_labels, all_probs)
        except ValueError:
            auc_pr = 0.0

        # Log
        logger.info(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} AUC: {auc_roc:.3f} AP: {auc_pr:.3f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    t_total = time.perf_counter() - t_total_start
    logger.info(f"\nTraining complete in {t_total:.0f}s ({t_total/60:.1f} min)")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # ── Test ──
    logger.info("\n=== Test Evaluation ===")
    model.eval()
    test_probs = []
    test_labels_list = []

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
    metrics = compute_detection_metrics(test_probs, test_labels, test_probs)

    logger.info(f"Test AUC-ROC:  {metrics.get('auc_roc', 0):.4f}")
    logger.info(f"Test AUC-PR:   {metrics.get('auc_pr', 0):.4f}")
    logger.info(f"Test F1:       {metrics.get('f1', 0):.4f}")
    logger.info(f"Test Precision:{metrics.get('precision', 0):.4f}")
    logger.info(f"Test Recall:   {metrics.get('recall', 0):.4f}")
    logger.info(f"Peak VRAM:     {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "swat" / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt_path)
    logger.info(f"Model saved to {ckpt_path}")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    # Prepare data
    train_w, train_s, train_l, test_w, test_s, test_l = prepare_datasets(
        n_max_train=args.n_windows,
        n_max_test=args.n_windows // 3,
    )

    # Create dataloaders (no vision features for initial run)
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        train_w, train_s, train_l,
        test_w, test_s, test_l,
        batch_size=args.batch_size,
        vision_features=None,
    )

    # Train at max GPU speed
    metrics = train_max_gpu(
        train_loader, val_loader, test_loader,
        n_sensors=train_w.shape[2],
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    logger.info("\nDone! Check checkpoints/swat/best_model.pt")
    return metrics


if __name__ == "__main__":
    main()

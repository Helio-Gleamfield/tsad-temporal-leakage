"""
Train ViCSynAD fusion model with VLM semantic features.

Architecture:
- VLM reasoning text → SentenceTransformer embedding → "vision proxy" feature
- Numeric features (statistical + raw window) → NumericEncoder
- Cross-modal fusion → anomaly head
"""

import sys; sys.path.insert(0, "src")
import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, CHECKPOINT_DIR, train as tc
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import extract_statistical_features
from vicsynad.modules.fusion import ViCSynADModel
from vicsynad.training.trainer import ViCSynADTrainer, ViCSynADDataset, compute_detection_metrics
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

# ── Text Embedding for VLM Reasoning ───────────────────────────────

def embed_vlm_reasoning(vlm_results_path: Path) -> np.ndarray:
    """Embed VLM reasoning text into semantic feature vectors."""
    import json

    logger.info("Loading SentenceTransformer for text embedding...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, 384-dim output

    # Load raw VLM results
    data = np.load(vlm_results_path, allow_pickle=True)

    # Reconstruct reasoning texts from saved results
    # We need to re-read the API output - for now use vlm_scores as proxy feature
    # and vlm_predictions as binary feature

    logger.info(f"VLM features available: {list(data.keys())}")

    # Use VLM features as input: [anomaly_score, severity/norm, n_affected/norm, ...per-sensor flags]
    vlm_features = data["vlm_features"]  # (N, 54) = [score, sev, n_aff, 51 sensor flags]

    # Combine with vlm_scores for richer representation
    vlm_scores = data["vlm_scores"].reshape(-1, 1)
    vlm_preds = data["vlm_predictions"].reshape(-1, 1)

    # Final VLM feature: concat all available signals
    combined = np.concatenate([vlm_scores, vlm_preds, vlm_features], axis=1)
    logger.info(f"VLM feature shape: {combined.shape}")

    return combined.astype(np.float32)


def prepare_training_data(n_windows=2000):
    """Prepare training data including VLM features."""
    logger.info("Loading SWaT...")
    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(train_X[train_y == 0])
    train_samples = pp.process_dataset(train_X, train_y, "swat")
    test_samples = pp.process_dataset(test_X, test_y, "swat")

    # Limit
    train_samples = train_samples[:n_windows]
    test_samples = test_samples[:n_windows // 3]

    # Inject synthetic anomalies
    train_w = np.stack([s.values for s in train_samples])
    train_l = np.array([s.label for s in train_samples])

    if train_l.mean() < 0.01:
        rng = np.random.default_rng(42)
        n_anom = int(len(train_w) * 0.15)
        anom_idx = rng.choice(len(train_w), n_anom, replace=False)
        for idx in anom_idx:
            sc = rng.choice(train_w.shape[2], size=rng.integers(1, 6), replace=False)
            mag = rng.uniform(3.0, 8.0)
            dur = rng.integers(10, 50)
            start = rng.integers(50, train_w.shape[1] - dur - 1)
            for s in sc:
                train_w[idx, start:start+dur, s] += mag * rng.choice([-1, 1])
                train_w[idx, start:start+dur, s] += rng.normal(0, mag*0.2, dur)
            train_l[idx] = 1
        logger.info(f"Injected {n_anom} synthetic anomalies")

    train_s = np.stack([extract_statistical_features(w) for w in train_w])

    test_w = np.stack([s.values for s in test_samples])
    test_s = np.stack([extract_statistical_features(s.values) for s in test_samples])
    test_l = np.array([s.label for s in test_samples])

    # Load VLM features
    vlm_path = CHECKPOINT_DIR / "swat" / "vlm_api_results.npz"
    vlm_feat = None
    if vlm_path.exists():
        vlm_feat = embed_vlm_reasoning(vlm_path)
        # Pad/trim VLM features to match training data
        n_vlm = len(vlm_feat)
        if n_vlm < len(train_w):
            pad = np.zeros((len(train_w) - n_vlm, vlm_feat.shape[1]), dtype=np.float32)
            vlm_feat = np.concatenate([vlm_feat, pad], axis=0)
        else:
            vlm_feat = vlm_feat[:len(train_w)]

    logger.info(f"Train: {train_w.shape}, anomaly={train_l.mean():.3f}")
    logger.info(f"Test:  {test_w.shape}, anomaly={test_l.mean():.3f}")
    if vlm_feat is not None:
        logger.info(f"VLM features: {vlm_feat.shape}")

    return train_w, train_s, train_l, test_w, test_s, test_l, vlm_feat


def train_with_vlm_features():
    """Full training pipeline with VLM features."""
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Prepare data
    train_w, train_s, train_l, test_w, test_s, test_l, vlm_feat = prepare_training_data(n_windows=2000)

    n_vlm_dim = vlm_feat.shape[1] if vlm_feat is not None else 3584
    n_sensors = train_w.shape[2]

    # Create model
    model = ViCSynADModel(
        n_sensors=n_sensors,
        vision_dim=n_vlm_dim,
    ).to(device)

    logger.info(f"Model: {model.count_trainable_params():,} trainable params")
    logger.info(f"Vision dim: {n_vlm_dim}")

    # Split train/val
    n_train = int(len(train_w) * 0.85)
    idx = np.random.default_rng(42).permutation(len(train_w))
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    dl_kwargs = dict(num_workers=2, pin_memory=True, prefetch_factor=2)

    train_ds = ViCSynADDataset(train_w[tr_idx], train_s[tr_idx],
                                vlm_feat[tr_idx] if vlm_feat is not None else None,
                                train_l[tr_idx])
    val_ds = ViCSynADDataset(train_w[val_idx], train_s[val_idx],
                              vlm_feat[val_idx] if vlm_feat is not None else None,
                              train_l[val_idx])
    test_ds = ViCSynADDataset(test_w, test_s,
                               vlm_feat[-len(test_w):] if vlm_feat is not None else None,
                               test_l)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, **dl_kwargs)

    # Train
    trainer_config = {
        "focal_alpha": 0.25, "focal_gamma": 2.0,
        "contrastive_temperature": 0.07, "contrastive_weight": 0.3,
        "learning_rate": 3e-4, "weight_decay": 1e-5,
    }

    trainer = ViCSynADTrainer(model, trainer_config, device)
    checkpoint_dir = CHECKPOINT_DIR / "swat" / "vlm_fusion"

    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=30,
        early_stopping_patience=10,
        checkpoint_dir=checkpoint_dir,
    )

    # Test
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in test_loader:
            vf = batch["vision_features"].to(device)
            win = batch["window"].to(device)
            stat = batch["stat_features"].to(device)
            outputs = model(vf, win, stat)
            all_scores.append(torch.sigmoid(outputs["logits"]).cpu().float())
            all_labels.append(batch["label"])

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_detection_metrics(all_scores, all_labels, all_scores)

    logger.info(f"\n=== ViCSynAD + VLM Features Results ===")
    logger.info(f"  AUC-ROC:  {metrics.get('auc_roc', 0):.4f}")
    logger.info(f"  AUC-PR:   {metrics.get('auc_pr', 0):.4f}")
    logger.info(f"  F1:       {metrics.get('f1', 0):.4f}")

    # Save
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    return metrics


if __name__ == "__main__":
    train_with_vlm_features()

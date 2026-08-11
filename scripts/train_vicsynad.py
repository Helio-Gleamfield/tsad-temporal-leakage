"""
Main training script for ViCSynAD.

Usage:
    python scripts/train_vicsynad.py --dataset swat --epochs 50

The script:
1. Loads & preprocesses data
2. Optionally extracts VLM vision features (if VLM is available)
3. Trains the fusion model
4. Saves checkpoints and metrics
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np
from vicsynad.config import (
    PROJECT_ROOT, CHECKPOINT_DIR, EXPERIMENT_DIR,
    model as model_cfg, train as train_cfg, hardware as hw_cfg,
)
from vicsynad.data.processor import load_swat, DataPreprocessor, extract_statistical_features
from vicsynad.modules.fusion import ViCSynADModel
from vicsynad.training.trainer import ViCSynADTrainer, ViCSynADDataset, compute_detection_metrics
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViCSynAD")
    parser.add_argument("--dataset", type=str, default="swat", help="Dataset name")
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--lr", type=float, default=train_cfg.learning_rate)
    parser.add_argument("--use_vlm", action="store_true", help="Use VLM vision features")
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--no_vlm", action="store_true", help="Train without VLM features")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def prepare_data(args):
    """Load and preprocess the dataset."""
    logger.info("=" * 60)
    logger.info("Phase 1: Data Preparation")
    logger.info("=" * 60)

    pp = DataPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
    )

    # Load SWaT
    swat_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/SWaT/AllInOne")
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    # Fit scaler on normal training data
    normal_mask = train_y == 0
    pp.fit_scaler(train_X[normal_mask])

    # Process
    train_samples = pp.process_dataset(train_X, train_y, "swat_train")
    test_samples = pp.process_dataset(test_X, test_y, "swat_test")

    # Convert to arrays
    def samples_to_arrays(samples):
        windows = np.stack([s.values for s in samples])
        labels = np.array([s.label for s in samples])
        # Extract statistical features
        stat_feats = np.stack([
            extract_statistical_features(s.values)
            for s in samples
        ])
        return windows, stat_feats, labels

    train_windows, train_stats, train_labels = samples_to_arrays(train_samples)
    test_windows, test_stats, test_labels = samples_to_arrays(test_samples)

    logger.info(f"Train: {train_windows.shape}, Anomaly: {train_labels.mean():.4f}")
    logger.info(f"Test:  {test_windows.shape}, Anomaly: {test_labels.mean():.4f}")

    return train_windows, train_stats, train_labels, test_windows, test_stats, test_labels


def extract_vision_features(windows, vlm_model_id, dataset_name):
    """Extract vision features using VLM (optional, VRAM-intensive)."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Vision Feature Extraction (VLM)")
    logger.info("=" * 60)

    from vicsynad.modules.ts_vis import TSVisualizer
    from vicsynad.modules.vlm_pipeline import VLMFeatureExtractor

    # Render images
    viz = TSVisualizer()
    logger.info(f"Rendering {len(windows)} time series images...")

    images = viz.render_batch(windows)

    # Extract features
    logger.info(f"Loading VLM: {vlm_model_id}")
    extractor = VLMFeatureExtractor(model_id=vlm_model_id)

    logger.info(f"Extracting vision features for {len(images)} images...")
    features = extractor.extract_vision_features(images, batch_size=4)

    logger.info(f"Vision features: {features.shape}")
    logger.info(f"VRAM usage: {extractor.vram_usage_gb:.1f} GB")

    return features.numpy()


def train(args, train_windows, train_stats, train_labels,
          test_windows, test_stats, test_labels, vision_features=None):
    """Train the ViCSynAD fusion model."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Model Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    n_sensors = train_windows.shape[2]
    logger.info(f"Input dimensions: T={train_windows.shape[1]}, D={n_sensors}")

    # Create model
    vision_dim = vision_features.shape[1] if vision_features is not None else 3584
    model = ViCSynADModel(
        n_sensors=n_sensors,
        vision_dim=vision_dim,
    )
    logger.info(f"Model trainable params: {model.count_trainable_params():,}")

    # Split train/val
    n_train = int(len(train_windows) * 0.85)
    indices = np.random.default_rng(42).permutation(len(train_windows))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Create datasets
    train_dataset = ViCSynADDataset(
        train_windows[train_idx],
        train_stats[train_idx],
        vision_features[train_idx] if vision_features is not None else None,
        train_labels[train_idx],
    )
    val_dataset = ViCSynADDataset(
        train_windows[val_idx],
        train_stats[val_idx],
        vision_features[val_idx] if vision_features is not None else None,
        train_labels[val_idx],
    )
    test_dataset = ViCSynADDataset(
        test_windows,
        test_stats,
        vision_features[-len(test_windows):] if vision_features is not None else None,
        test_labels,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
    )

    # Trainer
    trainer_config = {
        "focal_alpha": train_cfg.focal_alpha,
        "focal_gamma": train_cfg.focal_gamma,
        "contrastive_temperature": train_cfg.contrastive_temperature,
        "contrastive_weight": train_cfg.contrastive_weight,
        "learning_rate": args.lr,
        "weight_decay": train_cfg.weight_decay,
    }

    trainer = ViCSynADTrainer(model, trainer_config, device)

    # Train
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR / args.dataset
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=train_cfg.early_stopping_patience,
        checkpoint_dir=output_dir,
    )

    # Test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Evaluation")
    logger.info("=" * 60)

    trainer.model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            vision_emb = batch["vision_features"].to(device)
            window = batch["window"].to(device)
            stat = batch["stat_features"].to(device)
            labels = batch["label"]

            outputs = trainer.model(vision_emb, window, stat)
            scores = torch.sigmoid(outputs["logits"])
            all_scores.append(scores.cpu())
            all_labels.append(labels)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_detection_metrics(all_scores, all_labels, all_scores)
    logger.info("\nTest Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    import json
    results_path = output_dir / "test_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in metrics.items()}, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return metrics, history


def main():
    args = parse_args()
    logger.info("ViCSynAD Training Pipeline")
    logger.info(f"Config: {args}")

    # Prepare data
    train_w, train_s, train_l, test_w, test_s, test_l = prepare_data(args)

    # Extract vision features (if VLM available)
    vision_features = None
    if args.use_vlm and not args.no_vlm:
        # Combine train + test for feature extraction
        all_windows = np.concatenate([train_w[:500], test_w[:200]], axis=0)  # Subset for testing
        vision_features = extract_vision_features(
            all_windows, args.vlm_model_id, args.dataset
        )

    # Train
    metrics, history = train(
        args,
        train_w, train_s, train_l,
        test_w, test_s, test_l,
        vision_features,
    )

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()

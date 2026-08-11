"""
Training pipeline for ViCSynAD.

Trains the fusion module (Layer 2) with a frozen VLM vision encoder.
Uses Focal Loss + Contrastive Loss for handling class imbalance and
learning discriminative normal/anomaly representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Loss Functions ──────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in anomaly detection.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = torch.where(
            targets == 1, self.alpha, 1 - self.alpha
        )
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss to separate normal and anomalous embeddings.
    Pulls normal embeddings together, pushes anomalous apart from normal.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized projections
            labels: (B,) binary labels (0=normal, 1=anomalous)
        """
        B = embeddings.shape[0]

        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Positive pairs: same label
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)

        # Negative pairs: different labels
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)

        # InfoNCE-style loss
        # For each sample, compute: -log( sum(pos_exp) / (sum(pos_exp) + sum(neg_exp)) )
        exp_sim = torch.exp(sim)

        # Don't include self (use mask, NOT in-place)
        diag_mask = 1.0 - torch.eye(B, device=embeddings.device)
        exp_sim = exp_sim * diag_mask

        pos_exp = (exp_sim * pos_mask).sum(dim=1)
        neg_exp = (exp_sim * neg_mask).sum(dim=1)

        # Only compute for samples that have at least one positive pair
        valid = pos_mask.sum(dim=1) > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = -torch.log(
            pos_exp[valid] / (pos_exp[valid] + neg_exp[valid] + 1e-8)
        ).mean()

        return loss


# ── Dataset ────────────────────────────────────────────────────────

class ViCSynADDataset(Dataset):
    """Dataset for ViCSynAD training with pre-extracted vision features."""

    def __init__(
        self,
        windows: np.ndarray,       # (N, T, D)
        stat_features: np.ndarray,  # (N, D, 6)
        vision_features: Optional[np.ndarray],  # (N, vision_dim) or None
        labels: np.ndarray,         # (N,)
    ):
        self.windows = torch.FloatTensor(windows)
        self.stat_features = torch.FloatTensor(stat_features)
        self.labels = torch.FloatTensor(labels)

        if vision_features is not None:
            self.vision_features = torch.FloatTensor(vision_features)
            self.has_vision = True
        else:
            # Placeholder for VLM-offline mode
            self.vision_features = torch.zeros(len(windows), 3584)
            self.has_vision = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "window": self.windows[idx],
            "stat_features": self.stat_features[idx],
            "vision_features": self.vision_features[idx],
            "label": self.labels[idx],
            "has_vision": self.has_vision,
        }


# ── Trainer ────────────────────────────────────────────────────────

class ViCSynADTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Losses
        self.focal_loss = FocalLoss(
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get("contrastive_temperature", 0.07),
        )
        self.contrastive_weight = config.get("contrastive_weight", 0.3)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.scheduler = None  # Set up in fit() based on steps

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

        logger.info(f"Trainer initialized on {device}")
        logger.info(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss for a batch."""
        vision_emb = batch["vision_features"].to(self.device)
        window = batch["window"].to(self.device)
        stat = batch["stat_features"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.model(vision_emb, window, stat)
        logits = outputs["logits"]
        proj = outputs["projection"]

        # Focal loss for classification
        focal = self.focal_loss(logits, labels)

        # Contrastive loss for embedding separation
        contrastive = self.contrastive_loss(proj, labels)

        # Combined
        total = focal + self.contrastive_weight * contrastive

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == labels).float().mean()

        return total, {
            "total": total.item(),
            "focal": focal.item(),
            "contrastive": contrastive.item(),
            "acc": acc.item(),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "focal": 0.0, "contrastive": 0.0, "acc": 0.0}
        n_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += metrics[k]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{metrics['total']:.4f}",
                "acc": f"{metrics['acc']:.3f}",
            })

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        self.train_losses.append(epoch_losses["total"])
        return epoch_losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        val_losses = {"total": 0.0, "focal": 0.0, "contrastive": 0.0, "acc": 0.0}
        n_batches = 0

        all_probs = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Validating"):
            loss, metrics = self._compute_loss(batch)

            vision_emb = batch["vision_features"].to(self.device)
            window = batch["window"].to(self.device)
            stat = batch["stat_features"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(vision_emb, window, stat)
            probs = torch.sigmoid(outputs["logits"])

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            for k in val_losses:
                val_losses[k] += metrics[k]
            n_batches += 1

        for k in val_losses:
            val_losses[k] /= max(n_batches, 1)

        # Compute full metrics
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            val_losses["auc_roc"] = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        except ValueError:
            val_losses["auc_roc"] = 0.0
        try:
            val_losses["auc_pr"] = average_precision_score(all_labels.numpy(), all_probs.numpy())
        except ValueError:
            val_losses["auc_pr"] = 0.0

        self.val_losses.append(val_losses["total"])
        return val_losses

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict:
        """Full training loop with early stopping."""
        patience_counter = 0
        history = {"train": [], "val": []}

        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            logger.info(
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"Val AUC-ROC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"Val AUC-PR: {val_metrics.get('auc_pr', 0):.4f}"
            )

            # Early stopping
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0

                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.best_model_state,
                        checkpoint_dir / "best_model.pt",
                    )
                    logger.info(f"  ✓ Checkpoint saved")
            else:
                patience_counter += 1
                logger.info(
                    f"  No improvement ({patience_counter}/{early_stopping_patience})"
                )

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return history


# ── Metrics ────────────────────────────────────────────────────────

def compute_detection_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive detection metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix,
    )

    preds_binary = (predictions > 0.5).astype(int)

    metrics = {}
    metrics["precision"] = precision_score(labels, preds_binary, zero_division=0)
    metrics["recall"] = recall_score(labels, preds_binary, zero_division=0)
    metrics["f1"] = f1_score(labels, preds_binary, zero_division=0)

    try:
        metrics["auc_roc"] = roc_auc_score(labels, scores)
    except ValueError:
        metrics["auc_roc"] = 0.0
    try:
        metrics["auc_pr"] = average_precision_score(labels, scores)
    except ValueError:
        metrics["auc_pr"] = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp

    return metrics


if __name__ == "__main__":
    print("ViCSynAD Trainer module loaded successfully.")
    print(f"  FocalLoss: α=0.25, γ=2.0")
    print(f"  ContrastiveLoss: τ=0.07")

"""
Global configuration for ViCSynAD.
All hyperparameters, paths, and hardware-specific settings are centralized here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# DATA_ROOT can be overridden via environment variable TSAD_DATA_ROOT
_DATA_ROOT_ENV = os.environ.get("TSAD_DATA_ROOT", "")
DATA_ROOT = Path(_DATA_ROOT_ENV) if _DATA_ROOT_ENV else PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT

DATASET_PATHS = {
    "swat": DATA_ROOT / "SWaT" / "AllInOne",
    "skab": DATA_ROOT / "SKAB" / "data",
    "tep": DATA_ROOT / "TEP" / "new_tep_datasets-main",
    "mtsbench": DATA_ROOT / "mTSBench",
    "tab": DATA_ROOT / "TAB-main",
}

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
FIGURE_DIR = OUTPUT_ROOT / "figures"
EXPERIMENT_DIR = OUTPUT_ROOT / "experiments"
PAPER_DIR = OUTPUT_ROOT / "paper"


# ── Hardware ─────────────────────────────────────────────────────────
@dataclass
class HardwareConfig:
    gpu_name: str = "NVIDIA GeForce RTX 5060 Laptop GPU"
    vram_total_gb: float = 8.0
    vram_safety_margin_gb: float = 0.5
    cuda_version: str = "12.8"
    torch_dtype: str = "bfloat16"  # BF16 for Ampere+
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


# ── VLM Model ───────────────────────────────────────────────────────
@dataclass
class VLMConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    vision_encoder_id: Optional[str] = None  # Uses model's own vision encoder
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_model_vram_gb: float = 5.5  # Target VRAM for Qwen2-VL-7B 4-bit


# ── Visualization Engine ─────────────────────────────────────────────
@dataclass
class VisConfig:
    image_size: tuple = (448, 448)  # H x W, matches Qwen2-VL optimal
    window_size: int = 256
    stride: int = 64
    dpi: int = 100
    line_width: float = 0.5
    color_palette: str = "colorblind"  # Wong 2011 colorblind-safe
    include_heatmap: bool = True
    include_spectrum: bool = False  # Optional STFT spectrogram
    heatmap_size_ratio: float = 0.3  # Fraction of image height for heatmap


# ── Model Architecture ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Numeric encoder
    numeric_hidden_dim: int = 256
    numeric_num_layers: int = 3
    numeric_kernel_size: int = 7
    numeric_dropout: float = 0.1

    # Statistical features
    n_stat_features: int = 6  # mean, std, min, max, skew, kurtosis

    # Cross-modal fusion
    fusion_num_heads: int = 4
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.1

    # Anomaly head
    anomaly_head_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Latent dimensions
    vision_feature_dim: int = 3584  # Qwen2-VL-7B vision output dim
    numeric_output_dim: int = 256
    joint_embedding_dim: int = 256


# ── Training ────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_epochs: int = 50
    early_stopping_patience: int = 10

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.3  # Weight for contrastive loss vs focal loss

    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4


# ── Causal Discovery ────────────────────────────────────────────────
@dataclass
class CausalConfig:
    # PC algorithm
    pc_alpha: float = 0.05  # Conditional independence test threshold
    pc_test: str = "fisherz"
    pc_stable: bool = True

    # LiNGAM
    lingam_measure: str = "pwling"

    # Domain prior (SWaT P&ID topology)
    use_domain_prior: bool = True
    prior_edge_confidence: float = 0.95

    # Root cause analysis
    top_k_causes: int = 5
    attribution_samples: int = 100  # Monte Carlo samples for counterfactual


# ── CoT Explanation ─────────────────────────────────────────────────
@dataclass
class ExplainConfig:
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    cot_steps: List[str] = field(default_factory=lambda: [
        "pattern_identification",
        "anomaly_characterization",
        "causal_root_cause",
        "propagation_path",
        "recommendation",
    ])


# ── Datasets ────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    name: str = "swat"
    window_size: int = 256
    stride: int = 64
    normalize: str = "zscore"  # "zscore" | "minmax" | "none"
    anomaly_ratio_threshold: float = 0.01  # Window is anomalous if >1% points are
    min_train_normal_ratio: float = 0.99  # Training set purity requirement


# ── Evaluation ──────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    # Detection metrics
    detection_metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1", "auc_roc", "auc_pr",
        "adjusted_f1", "vus_roc", "vus_pr"
    ])
    # Causal metrics
    causal_metrics: List[str] = field(default_factory=lambda: [
        "shd", "sid", "f1_causal", "top_k_recall"
    ])
    # Explanation metrics
    explanation_metrics: List[str] = field(default_factory=lambda: [
        "bleu", "rouge_l", "bert_score", "causal_faithfulness"
    ])
    # Efficiency metrics
    efficiency_metrics: List[str] = field(default_factory=lambda: [
        "inference_latency_ms", "peak_vram_gb", "throughput_wps"
    ])


# ── Instantiate configs ─────────────────────────────────────────────
hardware = HardwareConfig()
vlm = VLMConfig()
vis = VisConfig()
model = ModelConfig()
train = TrainConfig()
causal = CausalConfig()
explain = ExplainConfig()
dataset = DatasetConfig()
eval_cfg = EvalConfig()

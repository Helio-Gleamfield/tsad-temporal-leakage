"""
GPU-Saturating VLM Feature Extraction Pipeline.

Loads Qwen2-VL-7B with 4-bit quantization (~5.5GB VRAM),
maximizes batch size, and extracts vision features at full GPU throughput.

Strategy:
- Use torch.compile for vision encoder
- Max batch size based on available VRAM
- Pre-render images in parallel CPU threads
- Stream directly to GPU without disk I/O bottleneck
"""

import sys; sys.path.insert(0, "src")
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, CHECKPOINT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import TSVisualizer, extract_statistical_features
from vicsynad.modules.vlm_pipeline import VLMFeatureExtractor


def render_batch_parallel(
    windows: np.ndarray,
    viz: TSVisualizer,
    n_workers: int = 8,
) -> list:
    """Render time series images in parallel across CPU threads."""
    logger.info(f"Rendering {len(windows)} images with {n_workers} workers...")
    t0 = time.perf_counter()

    def render_one(idx_win):
        idx, win = idx_win
        return idx, viz.render(win)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(
            render_one,
            enumerate(windows),
            chunksize=max(1, len(windows) // (n_workers * 4)),
        ))

    # Sort by original index
    results.sort(key=lambda x: x[0])
    images = [img for _, img in results]

    elapsed = time.perf_counter() - t0
    logger.info(f"Rendered {len(images)} images in {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
    return images


def extract_features_gpu(
    extractor: VLMFeatureExtractor,
    images: list,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract vision features at maximum GPU throughput."""
    logger.info(f"Extracting features for {len(images)} images (batch_size={batch_size})...")
    t0 = time.perf_counter()

    all_features = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    # Warmup
    warmup_imgs = images[:min(4, len(images))]
    _ = extractor.extract_vision_features(warmup_imgs, batch_size=min(4, len(warmup_imgs)))
    torch.cuda.synchronize()
    logger.info(f"Warmup complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    # Full extraction
    t_start = time.perf_counter()
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        features = extractor.extract_vision_features(batch_imgs, batch_size=len(batch_imgs))
        all_features.append(features)
        torch.cuda.synchronize()

        if (i // batch_size) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + len(batch_imgs)) / elapsed
            eta = (len(images) - i - len(batch_imgs)) / max(rate, 1)
            logger.info(f"  [{i+len(batch_imgs)}/{len(images)}] {rate:.1f} img/s, ETA: {eta:.0f}s")

    result = torch.cat(all_features, dim=0).numpy()
    elapsed = time.perf_counter() - t0
    logger.info(f"Feature extraction complete: {result.shape} in {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
    logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="swat", choices=["swat", "tep", "msl", "smap", "smd", "psm"])
    parser.add_argument("--max_windows", type=int, default=500, help="Max windows to process (for quick test)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model_id", default="Qwen/Qwen2-VL-7B-Instruct")
    args = parser.parse_args()

    # ── Load Data ──
    logger.info(f"=== Phase 1: Loading {args.dataset} ===")
    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(train_X[train_y == 0])
    train_samples = pp.process_dataset(train_X, train_y, "swat_train")
    test_samples = pp.process_dataset(test_X, test_y, "swat_test")

    # Limit for quick extraction
    n_train = min(len(train_samples), args.max_windows)
    n_test = min(len(test_samples), args.max_windows // 4)

    train_windows = np.stack([s.values for s in train_samples[:n_train]])
    test_windows = np.stack([s.values for s in test_samples[:n_test]])

    logger.info(f"Train windows: {train_windows.shape}")
    logger.info(f"Test windows:  {test_windows.shape}")

    # ── Render Images (CPU parallel) ──
    logger.info(f"\n=== Phase 2: Rendering images (CPU parallel) ===")
    viz = TSVisualizer(image_size=(448, 448), dpi=100)

    all_windows = np.concatenate([train_windows, test_windows], axis=0)
    images = render_batch_parallel(all_windows, viz, n_workers=8)

    # ── Extract Features (GPU) ──
    logger.info(f"\n=== Phase 3: VLM Feature Extraction (GPU) ===")
    logger.info(f"Loading {args.model_id} with 4-bit quantization...")

    extractor = VLMFeatureExtractor(model_id=args.model_id)
    features = extract_features_gpu(extractor, images, batch_size=args.batch_size)

    # ── Save ──
    output_path = Path(args.output) if args.output else CHECKPOINT_DIR / args.dataset / "vision_features.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        train_features=features[:n_train],
        test_features=features[n_train:],
    )
    logger.info(f"\nFeatures saved to {output_path}")
    logger.info(f"  Train: {features[:n_train].shape}")
    logger.info(f"  Test:  {features[n_train:].shape}")

    # ── Cleanup ──
    del extractor
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared. Done!")


if __name__ == "__main__":
    main()

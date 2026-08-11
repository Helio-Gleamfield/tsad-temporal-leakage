"""
VLM API Feature Extraction via Alibaba Cloud DashScope.

Strategy: Use qwen-vl-max as zero-shot anomaly detector + semantic feature extractor.
- Send TS window images to VLM API
- Get structured JSON: anomaly_score, anomaly_type, affected_sensors, reasoning
- Embed text descriptions into feature vectors
- Train lightweight fusion model with these enriched features

API: DashScope OpenAI-compatible endpoint
Model: qwen-vl-max (best vision model)
"""

import sys; sys.path.insert(0, "src")
import os
import base64
import json
import time
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, CHECKPOINT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import TSVisualizer, extract_statistical_features

# ── API Config ─────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "DASHSCOPE_API_KEY not set. Copy .env.example to .env and fill in your API key, "
        "or set the environment variable: $env:DASHSCOPE_API_KEY='your_key'"
    )

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    timeout=60.0,
)

# ── Prompt Templates ───────────────────────────────────────────────

TSAD_SYSTEM_PROMPT = """You are an expert time series anomaly detection system specialized in cyber-physical systems.
Analyze the multivariate time series visualization and output a structured JSON response.

The time series shows multiple sensors from a water treatment testbed (SWaT).
Normal behavior: smooth curves with minor fluctuations, gradual changes, periodic patterns.
Anomalous behavior: sudden spikes, level shifts, unusual oscillations, sensor freezing, or simultaneous deviations across multiple sensors.

For each analysis, identify:
1. Whether any anomaly exists
2. Which sensors appear anomalous
3. Anomaly type (spike, shift, drift, oscillation, noise, or normal)
4. Severity (0-10)
5. Brief reasoning (one sentence)"""

TSAD_ANALYSIS_PROMPT = """Analyze this multivariate time series window from a water treatment system.
Output ONLY valid JSON (no markdown formatting, no code fences):

{
  "is_anomalous": true/false,
  "anomaly_score": 0.0-1.0,
  "affected_sensors": [list of sensor indices that show anomalies, empty if normal],
  "anomaly_types": {"sensor_N": "spike|shift|drift|oscillation|noise"},
  "severity": 0-10,
  "reasoning": "one sentence describing the key observation"
}"""


def encode_image_b64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 data URI."""
    buf = BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm_analysis(image: Image.Image, model: str = "qwen-vl-max") -> Dict:
    """Send TS image to VLM API for anomaly analysis."""
    img_b64 = encode_image_b64(image)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TSAD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        },
                        {"type": "text", "text": TSAD_ANALYSIS_PROMPT}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code fences if any)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}, raw={raw[:200]}...")
        return {"is_anomalous": False, "anomaly_score": 0.0, "affected_sensors": [],
                "anomaly_types": {}, "severity": 0, "reasoning": f"Parse error: {raw[:100]}"}
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {"is_anomalous": False, "anomaly_score": 0.0, "affected_sensors": [],
                "anomaly_types": {}, "severity": 0, "reasoning": f"API error: {str(e)[:100]}"}


def call_vlm_batch(
    images: List[Image.Image],
    model: str = "qwen-vl-max",
    max_workers: int = 8,
    rate_limit_delay: float = 0.2,
) -> List[Dict]:
    """Send batch of TS images to VLM API with concurrency."""
    logger.info(f"Calling VLM API for {len(images)} images ({max_workers} concurrent, model={model})...")

    results = [None] * len(images)
    t_start = time.perf_counter()
    n_completed = 0

    def process_one(idx_img):
        idx, img = idx_img
        time.sleep(idx * rate_limit_delay % 1.0)  # Stagger requests slightly
        return idx, call_vlm_analysis(img, model=model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_one, (i, img)) for i, img in enumerate(images)]

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            n_completed += 1
            if n_completed % 50 == 0 or n_completed == len(images):
                elapsed = time.perf_counter() - t_start
                rate = n_completed / elapsed
                eta = (len(images) - n_completed) / max(rate, 0.01)
                logger.info(f"  [{n_completed}/{len(images)}] {rate:.1f} img/s, ETA: {eta:.0f}s")

    elapsed = time.perf_counter() - t_start
    logger.info(f"VLM batch complete: {len(images)} images in {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
    return results


def results_to_features(vlm_results: List[Dict], n_sensors: int = 51) -> np.ndarray:
    """Convert VLM API results to feature vectors for training."""
    features = []

    for r in vlm_results:
        feat = [
            float(r.get("anomaly_score", 0.0)),
            float(r.get("severity", 0)) / 10.0,
            float(len(r.get("affected_sensors", []))) / max(n_sensors, 1),
        ]

        # Per-sensor anomaly indicator (sparse encoding)
        affected = set(r.get("affected_sensors", []))
        sensor_flags = [1.0 if i in affected else 0.0 for i in range(min(n_sensors, 51))]
        feat.extend(sensor_flags)

        features.append(np.array(feat, dtype=np.float32))

    return np.stack(features)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=200, help="Number of windows to analyze via VLM")
    parser.add_argument("--model", default="qwen-vl-max", help="DashScope VLM model ID")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # ── Load Data ──
    logger.info("=" * 60)
    logger.info("Phase 1: Loading & Preprocessing Data")
    logger.info("=" * 60)

    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(train_X[train_y == 0])
    train_samples = pp.process_dataset(train_X, train_y, "swat")
    test_samples = pp.process_dataset(test_X, test_y, "swat")

    # Select diverse samples: mix of normal and anomalous
    rng = np.random.default_rng(42)
    anom_test = [s for s in test_samples if s.label == 1]
    norm_test = [s for s in test_samples if s.label == 0]

    n_anom = min(len(anom_test), args.n_windows // 2)
    n_norm = min(len(norm_test), args.n_windows - n_anom)

    selected = (
        list(rng.choice(anom_test, n_anom, replace=False)) +
        list(rng.choice(norm_test, n_norm, replace=False))
    )
    rng.shuffle(selected)

    test_windows = np.stack([s.values for s in selected])
    test_labels = np.array([s.label for s in selected])
    logger.info(f"Selected {len(selected)} windows (anomalous={n_anom}, normal={n_norm})")

    # ── Render Images ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Rendering Time Series Images")
    logger.info("=" * 60)

    viz = TSVisualizer(image_size=(448, 448), dpi=100)
    images = []
    for i, s in enumerate(selected):
        img = viz.render(s.values, title=f"Window {i} (Label={s.label})")
        images.append(img)

    # Save sample
    images[0].save(str(CHECKPOINT_DIR / "swat" / "vlm_sample_0.png"))
    logger.info(f"Sample saved. Total: {len(images)} images")

    # ── VLM API Call ──
    logger.info("\n" + "=" * 60)
    logger.info(f"Phase 3: VLM API Analysis ({args.model})")
    logger.info("=" * 60)

    vlm_results = call_vlm_batch(
        images,
        model=args.model,
        max_workers=args.max_workers,
    )

    # ── Convert to Features ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Feature Conversion & Evaluation")
    logger.info("=" * 60)

    vlm_predictions = np.array([1 if r.get("is_anomalous", False) else 0 for r in vlm_results])
    vlm_scores = np.array([r.get("anomaly_score", 0.0) for r in vlm_results])

    # Compare with ground truth
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

    acc = accuracy_score(test_labels, vlm_predictions)
    try:
        auc = roc_auc_score(test_labels, vlm_scores)
    except ValueError:
        auc = 0.5
    try:
        ap = average_precision_score(test_labels, vlm_scores)
    except ValueError:
        ap = 0.0
    f1 = f1_score(test_labels, vlm_predictions, zero_division=0)

    logger.info(f"\nVLM Zero-Shot Results (qwen-vl-max):")
    logger.info(f"  Accuracy:     {acc:.4f}")
    logger.info(f"  AUC-ROC:      {auc:.4f}")
    logger.info(f"  AUC-PR:       {ap:.4f}")
    logger.info(f"  F1:           {f1:.4f}")
    logger.info(f"  Anomaly rate: {vlm_predictions.mean():.3f} (GT: {test_labels.mean():.3f})")

    # ── Save ──
    output_path = Path(args.output) if args.output else CHECKPOINT_DIR / "swat" / "vlm_api_results.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vlm_features = results_to_features(vlm_results)
    np.savez_compressed(
        output_path,
        vlm_predictions=vlm_predictions,
        vlm_scores=vlm_scores,
        vlm_features=vlm_features,
        ground_truth=test_labels,
        window_ids=np.arange(len(selected)),
    )
    logger.info(f"\nResults saved to {output_path}")

    # Sample outputs
    logger.info(f"\nSample VLM outputs (first 5):")
    for i in range(min(5, len(vlm_results))):
        r = vlm_results[i]
        logger.info(f"  [{i}] GT={test_labels[i]}, VLM={r.get('is_anomalous')}, "
                    f"score={r.get('anomaly_score', 0):.2f}, "
                    f"reason={r.get('reasoning', 'N/A')[:80]}")


if __name__ == "__main__":
    main()

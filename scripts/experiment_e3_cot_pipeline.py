"""
E3: Full CoT Pipeline Activation (VLM ONLINE)
==============================================
Purpose: Generate real end-to-end Chain-of-Thought explanations for 50 anomalous windows.

Pipeline:
  1. Load SWaT anomalous windows
  2. Render TS visualization images
  3. Run causal discovery → root cause + propagation path
  4. Call qwen-vl-max API for CoT explanation generation
  5. Parse structured 5-step explanation
  6. Save all outputs for paper & human evaluation (E5)

This is the EXPLANATION GENERATION experiment — the VLM's value is here, not in detection.
"""
import sys; sys.path.insert(0, "src")
import os, base64, json, time, numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from openai import OpenAI
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, EXPERIMENT_DIR, CHECKPOINT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.data.swat_causal_graph import build_swat_causal_prior, get_swat_labeled_sensors
from vicsynad.modules.ts_vis import TSVisualizer
from vicsynad.modules.causal_discovery import CausalDiscovery, CounterfactualRCA
from vicsynad.modules.cot_explainer import CoTExplainer, AnomalyExplanation

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
    timeout=120.0,
)


def encode_image_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm_cot_explanation(
    image: Image.Image,
    cot_prompt: str,
    system_prompt: str,
    model: str = "qwen-vl-max",
    max_retries: int = 3,
) -> str:
    """Call VLM API with CoT prompt, with exponential backoff on rate limit."""
    img_b64 = encode_image_b64(image)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            },
                            {"type": "text", "text": cot_prompt}
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.3,
                top_p=0.9,
                timeout=90.0,  # 90s timeout per request
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)[:200]
            if "rate" in err_msg.lower() or "429" in err_msg or "limit" in err_msg.lower():
                wait = 5 * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
                continue
            elif attempt < max_retries - 1:
                wait = 3 * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt+1}/{max_retries}): {err_msg}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                logger.error(f"VLM API call failed after {max_retries} attempts: {err_msg}")
                return f"[API ERROR: {err_msg}]"

    return "[API ERROR: max retries exceeded]"


@dataclass
class CoTWindowResult:
    window_idx: int
    ground_truth: int  # 0=normal, 1=anomalous
    anomaly_score: float
    root_causes: List[Dict]
    propagation_paths: List[List[int]]
    explanation: Dict[str, str]  # 5-step structured output
    vlm_raw_response: str
    image_path: str


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=50, help="Number of windows to explain")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--delay", type=float, default=3.0, help="API rate limit delay (seconds) — use 3s+ to avoid QPM limits")
    args = parser.parse_args()

    print("=" * 70)
    print("E3: Full CoT Pipeline Activation (VLM ONLINE)")
    print("=" * 70)

    # ── [1] Load Data ──
    print("\n[1/5] Loading SWaT data...")
    _, _, test_X, test_y = load_swat(DATA_ROOT / "SWaT" / "AllInOne")

    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(test_X[test_y == 0])
    test_samples = pp.process_dataset(test_X, test_y, "swat")

    # Select anomalous windows
    anomalous = [s for s in test_samples if s.label == 1]
    normal = [s for s in test_samples if s.label == 0]
    rng = np.random.default_rng(42)

    n_anom = min(len(anomalous), args.n_windows)
    selected = list(rng.choice(anomalous, n_anom, replace=False))
    rng.shuffle(selected)

    print(f"  Selected {len(selected)} anomalous windows from {len(test_samples)} total")

    # ── [2] Load Causal Graph (P&ID Ground Truth) ──
    print("\n[2/5] Loading causal graph from P&ID ground truth...")
    train_X, train_y, _, _ = load_swat(DATA_ROOT / "SWaT" / "AllInOne")
    X_normal = train_X[train_y == 0]

    gt_prior = build_swat_causal_prior(X_normal.shape[1])
    # Use P&ID ground truth directly — full 51-var PC is O(d²·2^depth) and infeasible
    # (see E4 experiment: 51-var PC >10min for depth 4 only)
    from vicsynad.modules.causal_discovery import CausalGraph
    adj = gt_prior["adj_matrix"].copy()
    # Add edge confidences: 0.95 for P&ID known edges, 0 for non-edges
    confidence = np.where(adj == 1, 0.95, 0.0).astype(np.float32)
    graph = CausalGraph(
        adj_matrix=adj,
        node_names=[f"Sensor_{i}" for i in range(51)],
        confidence=confidence,
    )
    print(f"  Causal graph (P&ID GT): {graph.n_nodes} nodes, {graph.n_edges:.0f} edges")

    # Build sensor name map
    known_sensors = get_swat_labeled_sensors()
    sensor_names = [f"S{i}" for i in range(51)]
    for idx, tag, desc in known_sensors:
        if idx < 51:
            sensor_names[idx] = tag

    # ── [3] Render Images ──
    print(f"\n[3/5] Rendering {len(selected)} TS visualizations...")
    viz = TSVisualizer(image_size=(448, 448), dpi=100)

    output_dir = CHECKPOINT_DIR / "swat" / "cot_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    image_paths = []
    for i, s in enumerate(selected):
        img = viz.render(s.values, title=f"Window {s.window_idx}")
        images.append(img)
        img_path = output_dir / f"cot_window_{i:03d}.png"
        img.save(str(img_path))
        image_paths.append(str(img_path))
    print(f"  Saved {len(images)} images to {output_dir}")

    # ── [4] Run CoT Pipeline ──
    print(f"\n[4/5] Running CoT explanation pipeline ({args.model})...")
    explainer = CoTExplainer()
    results: List[CoTWindowResult] = []

    t_start = time.perf_counter()

    for i, sample in enumerate(selected):
        window = sample.values

        # Compute simple anomaly score (Z-score based)
        mu = window.mean(axis=0)
        sigma = window.std(axis=0) + 1e-8
        z_scores = np.abs((window - mu) / sigma)
        anomaly_score = float(np.max(z_scores))
        anomalous_nodes = list(np.where(z_scores.max(axis=0) > 2.0)[0])

        # Find root causes via causal graph
        root_causes = graph.find_root_causes(anomalous_nodes) if anomalous_nodes else []
        paths = []
        for rc, _ in root_causes[:2]:
            p = graph.get_propagation_path(rc, set(anomalous_nodes))
            paths.extend(p)

        # Compute attribution
        attribution = {}
        for node in range(min(51, window.shape[1])):
            attr = float(z_scores[:, node].max())
            attribution[node] = attr

        # Format for CoT
        rc_formatted = [
            (idx, sensor_names[idx] if idx < len(sensor_names) else f"S{idx}", score)
            for idx, score in root_causes[:5]
        ]

        # Build CoT prompt
        graph_summary = (
            f"SWaT Water Treatment Process: P1 → P2 → P3 → P4 → P5 → P6\n"
            f"Discovered causal edges: {graph.n_edges:.0f} among {graph.n_nodes} sensors\n"
            f"Stage flow is strictly sequential (no backflow possible)"
        )

        cot_prompt = explainer.build_prompt(
            anomaly_score=anomaly_score,
            causal_graph_summary=graph_summary,
            attribution_scores=attribution,
            root_causes=rc_formatted,
            propagation_paths=paths,
            node_names=sensor_names,
        )

        # Call VLM API
        logger.info(f"  [{i+1}/{len(selected)}] Window {sample.window_idx} "
                    f"(anomalous_nodes={len(anomalous_nodes)}, root_causes={len(root_causes)})")

        vlm_raw = call_vlm_cot_explanation(
            images[i], cot_prompt, explainer.system_prompt, model=args.model
        )

        # Parse structured explanation
        explanation = explainer.parse_explanation(vlm_raw)

        results.append(CoTWindowResult(
            window_idx=int(sample.window_idx),
            ground_truth=int(sample.label),
            anomaly_score=round(anomaly_score, 2),
            root_causes=[{"idx": idx, "name": name, "score": round(score, 4)}
                        for idx, name, score in rc_formatted],
            propagation_paths=[[int(n) for n in p] for p in paths[:5]],
            explanation=explanation.to_dict(),
            vlm_raw_response=vlm_raw,
            image_path=image_paths[i],
        ))

        # Rate limiting
        time.sleep(args.delay)

        # Save intermediate results every 10 windows (safety checkpoint)
        if (i + 1) % 10 == 0 or (i + 1) == len(selected):
            checkpoint_path = EXPERIMENT_DIR / "e3_cot_checkpoint.json"
            checkpoint = {
                "experiment": "E3_COT_CHECKPOINT",
                "progress": f"{i+1}/{len(selected)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results_so_far": [asdict(r) for r in results],
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"  [CHECKPOINT] {i+1}/{len(selected)} results saved")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Completed {len(results)} explanations in {elapsed:.1f}s "
          f"({elapsed/len(results):.1f}s per window)")

    # ── [5] Save ──
    print("\n[5/5] Saving results...")

    # Summary statistics
    sections_found = {"pattern_identification": 0, "anomaly_characterization": 0,
                      "causal_root_cause": 0, "propagation_path": 0, "recommendation": 0}
    for r in results:
        for section in sections_found:
            exp_text = r.explanation.get(section, "")
            if exp_text and exp_text not in ("Not extracted", "", "[VLM offline - prompt generated]"):
                sections_found[section] += 1

    n_total = len(results)
    print(f"\n  Explanation Section Coverage ({n_total} windows):")
    for section, count in sections_found.items():
        print(f"    {section}: {count}/{n_total} ({count/n_total*100:.0f}%)")

    output = {
        "experiment": "E3_FULL_COT_PIPELINE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_windows": len(results),
        "causal_graph": {
            "n_nodes": graph.n_nodes,
            "n_edges": int(graph.n_edges),
        },
        "section_coverage": {k: f"{v}/{n_total}" for k, v in sections_found.items()},
        "elapsed_seconds": round(elapsed, 1),
        "results": [asdict(r) for r in results],
    }

    output_path = Path(args.output) if args.output else EXPERIMENT_DIR / "e3_cot_pipeline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full results saved to {output_path}")

    # Also save a human-readable sample
    sample_path = EXPERIMENT_DIR / "e3_cot_samples.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results[:5]):
            f.write(f"{'='*60}\n")
            f.write(f"Sample {i+1}: Window {r.window_idx}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Anomaly Score: {r.anomaly_score}\n")
            f.write(f"Root Causes: {r.root_causes}\n\n")
            for section, text in r.explanation.items():
                if section != "raw_text":
                    f.write(f"\n### {section}\n{text}\n")
            f.write(f"\n\n[VLM RAW RESPONSE]\n{r.vlm_raw_response}\n\n")
    print(f"  Sample outputs saved to {sample_path}")


if __name__ == "__main__":
    main()

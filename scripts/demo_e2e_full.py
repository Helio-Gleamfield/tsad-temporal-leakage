"""
ViCSynAD Full End-to-End Demo: Detection → Causal Graph → RCA → CoT Explanation

Demonstrates the complete ViCSynAD pipeline on SWaT data.
"""
import sys; sys.path.insert(0, "src")
import numpy as np
import json
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.data.processor import load_swat
from vicsynad.data.swat_causal_graph import build_swat_causal_prior
from vicsynad.modules.causal_discovery import CausalGraph
from vicsynad.modules.cot_explainer import CoTExplainer

def demo():
    # ── Step 1: Load SWaT ──
    swat_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/SWaT/AllInOne")
    train_X, train_y, test_X, test_y = load_swat(swat_path)
    n_sensors = train_X.shape[1]

    logger.info(f"SWaT: {n_sensors} sensors, train={train_X.shape}, test={test_X.shape}")

    # ── Step 2: Build Causal Graph from SWaT P&ID ──
    logger.info("\n[Step 2] Building SWaT causal graph from P&ID...")
    prior = build_swat_causal_prior(n_sensors)

    graph = CausalGraph(
        adj_matrix=prior["adj_matrix"],
        node_names=[f"S{i}" for i in range(n_sensors)],
        confidence=prior["adj_matrix"].astype(np.float32),
    )

    logger.info(f"  Causal graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    logger.info(f"  Stage flow: {prior['stage_order']}")

    # ── Step 3: Detect anomalous sensor ──
    logger.info("\n[Step 3] Detecting anomalous window...")

    # Find a truly anomalous test window
    anom_indices = np.where(test_y == 1)[0]
    anom_idx = anom_indices[len(anom_indices)//2]  # Midpoint anomalous window
    normal_idx = anom_indices[0] - 1

    # Compute per-sensor z-scores relative to train distribution
    train_mean = train_X[train_y == 0].mean(axis=0)
    train_std = train_X[train_y == 0].std(axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    window = test_X[anom_idx:anom_idx+256]  # Take 256-point window around anomaly
    z_scores = np.abs((window - train_mean) / train_std).mean(axis=0)

    # Sensors with z > 2 are anomalous
    threshold = 2.0
    anomalous_sensors = sorted(np.where(z_scores > threshold)[0].tolist(),
                                key=lambda i: z_scores[i], reverse=True)

    logger.info(f"  Window at index {anom_idx} (GT: anomalous)")
    logger.info(f"  Anomalous sensors: {anomalous_sensors[:10]}")
    logger.info(f"  Top-5 z-scores: {[(i, z_scores[i]) for i in anomalous_sensors[:5]]}")

    # ── Step 4: Causal Root Cause Analysis ──
    logger.info("\n[Step 4] Causal Root Cause Analysis...")

    root_causes = graph.find_root_causes(anomalous_sensors[:15])

    logger.info(f"  Root cause candidates (most upstream anomalous):")
    for rc_idx, score in root_causes[:5]:
        stage = prior["sensor_stage_map"].get(rc_idx, "?")
        logger.info(f"    S{rc_idx} (Stage {stage}): causal_rc_score={score:.3f}, z={z_scores[rc_idx]:.1f}")

    # ── Step 5: Propagation Path ──
    logger.info("\n[Step 5] Tracing anomaly propagation paths...")

    propagation_paths = []
    if root_causes:
        for rc_idx, _ in root_causes[:2]:
            paths = graph.get_propagation_path(rc_idx, set(anomalous_sensors[:15]))
            propagation_paths.extend(paths)
            for p in paths[:3]:
                path_str = " -> ".join(
                    f"S{n}(Stage{prior['sensor_stage_map'].get(n,'?')})"
                    for n in p
                )
                logger.info(f"    {path_str}")

    # ── Step 6: Build Causal Attribution ──
    logger.info("\n[Step 6] Computing causal attributions...")

    attributions = {}
    for i in anomalous_sensors[:20]:
        # Combine z-score anomaly magnitude + causal upstream score
        z_norm = min(z_scores[i] / 10.0, 1.0)
        has_upstream = any(
            graph.adj_matrix[p, i] == 1
            for p in anomalous_sensors[:20] if p != i
        )
        causal_bonus = 0.3 if has_upstream else 0.7  # Root-like if no anomalous parent
        attributions[i] = float(z_norm * causal_bonus)

    # ── Step 7: Generate CoT Explanation ──
    logger.info("\n[Step 7] Generating Chain-of-Thought explanation...")

    explainer = CoTExplainer()

    # Format causal graph summary
    stage_edges = []
    stage_order = prior["stage_order"]
    for k in range(len(stage_order) - 1):
        upstream = stage_order[k]
        downstream = stage_order[k+1]
        stage_edges.append(f"{upstream} -> {downstream}")
    graph_summary = (
        f"SWaT Water Treatment Process: " + " -> ".join(stage_edges) + "\n"
        f"Discovered causal edges: {graph.n_edges} among {graph.n_sensors if hasattr(graph, 'n_sensors') else graph.n_nodes} sensors\n"
        f"Stage flow is strictly sequential (no backflow possible)"
    )

    rc_formatted = [
        (idx, f"S{idx}", score)
        for idx, score in root_causes[:3]
    ]

    explanation = explainer.explain(
        anomaly_score=float(z_scores[anomalous_sensors[0]] / 10.0) if anomalous_sensors else 0.9,
        causal_graph_summary=graph_summary,
        attribution_scores=attributions,
        root_causes=rc_formatted,
        propagation_paths=propagation_paths[:3],
        node_names=[f"S{i}" for i in range(n_sensors)],
    )

    logger.info("\n" + "=" * 60)
    logger.info("ViCSynAD FULL PIPELINE OUTPUT")
    logger.info("=" * 60)
    logger.info(explanation.to_markdown())

    # ── Save ──
    output = {
        "dataset": "SWaT",
        "n_sensors": n_sensors,
        "causal_graph": {
            "n_nodes": graph.n_nodes,
            "n_edges": graph.n_edges,
            "stage_flow": stage_order,
        },
        "detection": {
            "window_index": int(anom_idx),
            "ground_truth": "anomalous",
            "anomalous_sensors": anomalous_sensors[:20],
            "top_z_scores": {str(i): float(z_scores[i]) for i in anomalous_sensors[:10]},
        },
        "root_cause_analysis": [
            {"sensor": int(idx), "score": float(score)}
            for idx, score in root_causes[:5]
        ],
        "propagation_paths": [
            " -> ".join(f"S{n}" for n in p)
            for p in propagation_paths[:5]
        ],
        "explanation": explanation.to_dict(),
    }

    output_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/Science Bulletin/experiments/demo_e2e.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nFull pipeline output saved to: {output_path}")
    logger.info("ViCSynAD End-to-End Demo Complete!")

    return output

if __name__ == "__main__":
    demo()

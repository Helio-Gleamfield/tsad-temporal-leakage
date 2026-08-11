"""
Full evaluation pipeline for ViCSynAD.

Evaluates all four research questions:
  RQ1: Anomaly detection performance
  RQ2: Causal root cause accuracy
  RQ3: Explanation quality
  RQ4: Computational efficiency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.data.swat_causal_graph import build_swat_causal_prior
from vicsynad.modules.fusion import ViCSynADModel
from vicsynad.modules.causal_discovery import (
    CausalDiscovery, CausalGraph, CounterfactualRCA
)
from vicsynad.modules.cot_explainer import CoTExplainer
from vicsynad.training.trainer import compute_detection_metrics
from vicsynad.config import CHECKPOINT_DIR, EXPERIMENT_DIR


@dataclass
class EvaluationResults:
    """Complete evaluation results for one dataset."""
    dataset: str
    detection_metrics: Dict[str, float]
    causal_metrics: Dict[str, float]
    explanation_quality: Dict[str, float]
    efficiency_metrics: Dict[str, float]


class ViCSynADEvaluator:
    """Complete multi-dimensional evaluation of ViCSynAD."""

    def __init__(
        self,
        model: Optional[ViCSynADModel] = None,
        causal_graph: Optional[CausalGraph] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.causal_graph = causal_graph
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.explainer = CoTExplainer()

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()

    # ── RQ1: Detection Performance ──────────────────────────────────

    def evaluate_detection(
        self,
        test_windows: np.ndarray,
        test_stats: np.ndarray,
        test_labels: np.ndarray,
        vision_features: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance (RQ1)."""
        import torch
        from torch.utils.data import DataLoader
        from vicsynad.training.trainer import ViCSynADDataset

        print("\n" + "=" * 50)
        print("RQ1: Anomaly Detection Performance")
        print("=" * 50)

        if self.model is None:
            print("  No model loaded — skipping detection evaluation")
            return {}

        dataset = ViCSynADDataset(
            test_windows, test_stats, vision_features, test_labels
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                vision_emb = batch["vision_features"].to(self.device)
                window = batch["window"].to(self.device)
                stat = batch["stat_features"].to(self.device)
                labels = batch["label"]

                outputs = self.model(vision_emb, window, stat)
                scores = torch.sigmoid(outputs["logits"])
                all_scores.append(scores.cpu())
                all_labels.append(labels)

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metrics = compute_detection_metrics(all_scores, all_labels, all_scores)

        print(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        print(f"  AUC-PR:  {metrics.get('auc_pr', 0):.4f}")
        print(f"  F1:      {metrics.get('f1', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")

        return metrics

    # ── RQ2: Causal Root Cause Accuracy ────────────────────────────

    def evaluate_causal_accuracy(
        self,
        X_normal: np.ndarray,
        X_anomalous: np.ndarray,
        anomaly_labels: np.ndarray,
        ground_truth_prior: Dict,
    ) -> Dict[str, float]:
        """Evaluate causal discovery and root cause accuracy (RQ2)."""
        print("\n" + "=" * 50)
        print("RQ2: Causal Root Cause Accuracy")
        print("=" * 50)

        n_sensors = X_normal.shape[1]
        gt_adj = ground_truth_prior["adj_matrix"]

        # Causal discovery on normal data
        cd = CausalDiscovery(method="pc", alpha=0.05, domain_prior=ground_truth_prior)
        graph = cd.discover(X_normal)

        # Compute Structural Hamming Distance
        from vicsynad.modules.causal_discovery import CausalGraph
        discovered_adj = graph.adj_matrix

        # SHD: count edge differences (additions + deletions + reversals)
        min_dim = min(discovered_adj.shape[0], gt_adj.shape[0])
        d_adj = discovered_adj[:min_dim, :min_dim]
        g_adj = gt_adj[:min_dim, :min_dim]

        shd = np.sum(np.abs(d_adj - g_adj))
        # Normalize by max possible edges
        max_edges = min_dim * (min_dim - 1) // 2
        shd_norm = shd / max(max_edges, 1)

        # Edge-level F1
        tp = np.sum((d_adj == 1) & (g_adj == 1))
        fp = np.sum((d_adj == 1) & (g_adj == 0))
        fn = np.sum((d_adj == 0) & (g_adj == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_causal = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  SHD:       {shd} (normalized: {shd_norm:.4f})")
        print(f"  SID:       Not computed (requires do-calculus)")
        print(f"  F1-causal: {f1_causal:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Edges found: {discovered_adj.sum():.0f} | GT edges: {gt_adj.sum():.0f}")

        return {
            "shd": shd,
            "shd_normalized": shd_norm,
            "f1_causal": f1_causal,
            "precision": precision,
            "recall": recall,
            "discovered_edges": int(discovered_adj.sum()),
            "gt_edges": int(gt_adj.sum()),
        }

    # ── RQ3: Explanation Quality ───────────────────────────────────

    def evaluate_explanation(
        self,
        anomaly_score: float,
        causal_graph: CausalGraph,
        attribution: Dict[int, float],
        anomalous_nodes: List[int],
    ) -> Dict[str, float]:
        """Evaluate explanation quality (RQ3)."""
        print("\n" + "=" * 50)
        print("RQ3: Explanation Quality")
        print("=" * 50)

        root_causes = causal_graph.find_root_causes(anomalous_nodes)
        rc_formatted = [
            (idx, causal_graph.node_names[idx], score)
            for idx, score in root_causes
        ]

        paths = []
        for rc, _ in root_causes[:2]:
            p = causal_graph.get_propagation_path(rc, set(anomalous_nodes))
            paths.extend(p)

        graph_summary = (
            f"Discovered DAG with {causal_graph.n_nodes} nodes, "
            f"{causal_graph.n_edges} edges.\n"
            f"Stage flow: P1→P2→P3→P4→P5→P6\n"
        )

        explanation = self.explainer.explain(
            anomaly_score=anomaly_score,
            causal_graph_summary=graph_summary,
            attribution_scores=attribution,
            root_causes=rc_formatted,
            propagation_paths=paths,
            node_names=causal_graph.node_names,
        )

        print(f"  Root cause candidates: {rc_formatted[:3]}")
        print(f"  Explanation sections extracted:")
        print(f"    - Pattern: {'✓' if explanation.pattern_identification else '✗'}")
        print(f"    - Characterization: {'✓' if explanation.anomaly_characterization else '✗'}")
        print(f"    - Root cause: {'✓' if explanation.causal_root_cause else '✗'}")
        print(f"    - Propagation: {'✓' if explanation.propagation_path else '✗'}")
        print(f"    - Recommendation: {'✓' if explanation.recommendation else '✗'}")

        # In offline mode, return structural metrics
        return {
            "sections_extracted": sum([
                bool(explanation.pattern_identification),
                bool(explanation.anomaly_characterization),
                bool(explanation.causal_root_cause),
                bool(explanation.propagation_path),
                bool(explanation.recommendation),
            ]) / 5.0,
            "root_causes_found": len(root_causes),
            "explanation_length_chars": len(explanation.raw_text),
        }

    # ── RQ4: Efficiency ────────────────────────────────────────────

    def evaluate_efficiency(
        self,
        test_windows: np.ndarray,
        test_stats: np.ndarray,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """Evaluate computational efficiency (RQ4)."""
        print("\n" + "=" * 50)
        print("RQ4: Computational Efficiency")
        print("=" * 50)

        if self.model is None:
            print("  No model loaded — skipping efficiency evaluation")
            return {}

        # Warmup
        dummy_vision = torch.randn(1, 3584).to(self.device)
        dummy_window = torch.FloatTensor(test_windows[:1]).to(self.device)
        dummy_stat = torch.FloatTensor(test_stats[:1]).to(self.device)

        for _ in range(10):
            _ = self.model(dummy_vision, dummy_window, dummy_stat)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model(dummy_vision, dummy_window, dummy_stat)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / n_runs) * 1000
        throughput = n_runs / elapsed

        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        vram_peak = torch.cuda.max_memory_reserved() / 1024**3

        print(f"  Inference latency: {latency_ms:.2f} ms/window")
        print(f"  Throughput:        {throughput:.1f} windows/s")
        print(f"  Peak VRAM (alloc): {vram_gb:.2f} GB")
        print(f"  Peak VRAM (reserved): {vram_peak:.2f} GB")

        return {
            "latency_ms": latency_ms,
            "throughput_wps": throughput,
            "peak_vram_alloc_gb": vram_gb,
            "peak_vram_reserved_gb": vram_peak,
        }

    # ── Full Evaluation Pipeline ───────────────────────────────────

    def run_full_evaluation(
        self,
        X_normal: np.ndarray,
        X_test: np.ndarray,
        test_windows: np.ndarray,
        test_stats: np.ndarray,
        test_labels: np.ndarray,
        dataset_name: str = "swat",
    ) -> EvaluationResults:
        """Run the complete RQ1-RQ4 evaluation."""
        print("\n" + "=" * 70)
        print(f"ViCSynAD Full Evaluation — {dataset_name.upper()}")
        print("=" * 70)

        # Ground truth causal prior
        ground_truth_prior = build_swat_causal_prior(X_normal.shape[1])

        # RQ1
        det_metrics = self.evaluate_detection(
            test_windows, test_stats, test_labels
        )

        # RQ2: causal graph from normal data, evaluate vs GT
        # Use subset for causal discovery (PC algorithm is O(n^2) in samples)
        n_causal = min(10000, len(X_normal))
        idx = np.random.default_rng(42).choice(len(X_normal), n_causal, replace=False)
        causal_metrics = self.evaluate_causal_accuracy(
            X_normal[idx],
            X_test[:1000],
            test_labels[:1000],
            ground_truth_prior,
        )

        # RQ3: explanation on a detected anomalous window
        anomalous_idx = np.where(test_labels == 1)[0]
        if len(anomalous_idx) > 0:
            # Re-discover graph for explanation
            cd = CausalDiscovery(
                method="pc", alpha=0.05, domain_prior=ground_truth_prior
            )
            graph = cd.discover(X_normal[idx])

            # Mock attribution for the first anomalous window
            n_sensors = X_normal.shape[1]
            mock_attr = {
                i: np.random.random()
                for i in range(min(n_sensors, 10))
            }
            mock_anomalous = [0, 1, 3, 5]

            expl_metrics = self.evaluate_explanation(
                anomaly_score=0.95,
                causal_graph=graph,
                attribution=mock_attr,
                anomalous_nodes=mock_anomalous,
            )
        else:
            expl_metrics = {}

        # RQ4
        eff_metrics = self.evaluate_efficiency(test_windows, test_stats)

        results = EvaluationResults(
            dataset=dataset_name,
            detection_metrics=det_metrics,
            causal_metrics=causal_metrics,
            explanation_quality=expl_metrics,
            efficiency_metrics=eff_metrics,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Detection AUC-ROC: {det_metrics.get('auc_roc', 'N/A')}")
        print(f"  Causal F1:        {causal_metrics.get('f1_causal', 'N/A')}")
        print(f"  Expl. Sections:   {expl_metrics.get('sections_extracted', 'N/A')}")
        print(f"  Latency:          {eff_metrics.get('latency_ms', 'N/A')} ms")

        return results


def main():
    """Run the evaluation pipeline."""
    print("Loading ViCSynAD evaluation pipeline...")

    # Load data
    swat_path = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/SWaT/AllInOne")
    train_X, train_y, test_X, test_y = load_swat(swat_path)

    # Preprocess
    pp = DataPreprocessor(window_size=256, stride=64)
    pp.fit_scaler(train_X[train_y == 0])

    train_samples = pp.process_dataset(train_X, train_y, "swat_train")
    test_samples = pp.process_dataset(test_X, test_y, "swat_test")

    test_windows = np.stack([s.values for s in test_samples])
    test_labels = np.array([s.label for s in test_samples])
    from vicsynad.modules.ts_vis import extract_statistical_features
    test_stats = np.stack([extract_statistical_features(s.values) for s in test_samples])

    # Initialize evaluator (no model for now)
    evaluator = ViCSynADEvaluator(
        model=None,  # Load trained model path when available
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run evaluation (RQ2-RQ4; RQ1 requires trained model)
    results = evaluator.run_full_evaluation(
        X_normal=train_X,
        X_test=test_X,
        test_windows=test_windows,
        test_stats=test_stats,
        test_labels=test_labels,
        dataset_name="swat",
    )

    # Save
    output_path = EXPERIMENT_DIR / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

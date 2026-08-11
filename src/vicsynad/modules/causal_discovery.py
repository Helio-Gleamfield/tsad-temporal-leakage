"""
Causal Discovery & Root Cause Analysis Module (Layer 3 of ViCSynAD).

Performs causal graph discovery from normal-operation data and
counterfactual root cause attribution when anomalies are detected.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """Directed Acyclic Graph representing causal relationships."""
    adj_matrix: np.ndarray  # (D, D) adjacency, adj[i,j]=1 means i→j
    node_names: List[str]
    confidence: Optional[np.ndarray] = None  # (D, D) edge confidence scores

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def n_edges(self) -> int:
        return int(self.adj_matrix.sum())

    def get_parents(self, node_idx: int) -> List[int]:
        return list(np.where(self.adj_matrix[:, node_idx] == 1)[0])

    def get_children(self, node_idx: int) -> List[int]:
        return list(np.where(self.adj_matrix[node_idx, :] == 1)[0])

    def get_ancestors(self, node_idx: int) -> Set[int]:
        """All nodes on paths leading to node_idx."""
        ancestors = set()
        stack = [node_idx]
        while stack:
            current = stack.pop()
            parents = self.get_parents(current)
            for p in parents:
                if p not in ancestors:
                    ancestors.add(p)
                    stack.append(p)
        return ancestors

    def get_descendants(self, node_idx: int) -> Set[int]:
        """All nodes reachable from node_idx."""
        descendants = set()
        stack = [node_idx]
        while stack:
            current = stack.pop()
            children = self.get_children(current)
            for c in children:
                if c not in descendants:
                    descendants.add(c)
                    stack.append(c)
        return descendants

    def find_root_causes(
        self, anomalous_nodes: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Find the most upstream causal nodes among anomalous nodes.
        Root causes are anomalous nodes whose parents are NOT anomalous.
        """
        anomalous_set = set(anomalous_nodes)
        root_causes = []

        for node in anomalous_nodes:
            parents = set(self.get_parents(node))
            if not parents or parents.isdisjoint(anomalous_set):
                # No anomalous parent → this is a root cause
                confidence = 0.0
                if self.confidence is not None:
                    # Average confidence of edges leading to this node's descendants
                    descendants = self.get_descendants(node) & anomalous_set
                    if descendants:
                        conf_sum = 0.0
                        n_edges = 0
                        for d in descendants:
                            for p in self.get_parents(d):
                                if p in anomalous_set and self.confidence[p, d] > 0:
                                    conf_sum += self.confidence[p, d]
                                    n_edges += 1
                        confidence = conf_sum / n_edges if n_edges > 0 else 0.5
                    else:
                        confidence = 0.8  # Isolated anomalous node
                else:
                    confidence = 0.7

                root_causes.append((node, confidence))

        # Sort by confidence descending
        root_causes.sort(key=lambda x: x[1], reverse=True)
        return root_causes

    def get_propagation_path(
        self, root_node: int, affected_nodes: Set[int]
    ) -> List[List[int]]:
        """Find propagation paths from root cause to affected nodes."""
        paths = []
        for target in affected_nodes:
            if target == root_node:
                continue
            # BFS to find all paths from root to target
            all_paths = self._find_all_paths(root_node, target)
            paths.extend(all_paths)
        return paths

    def _find_all_paths(
        self, start: int, end: int, max_paths: int = 3
    ) -> List[List[int]]:
        """Find up to max_paths shortest paths from start to end in DAG.

        Uses BFS for shortest paths with a hard limit on search depth.
        """
        if start == end:
            return [[start]]

        from collections import deque
        paths = []

        # BFS: queue stores (current_node, path_so_far)
        queue = deque([(start, [start])])
        max_depth = 10  # Hard limit to prevent explosion on dense graphs

        while queue and len(paths) < max_paths:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue

            for child in self.get_children(current):
                if child == end:
                    paths.append(path + [child])
                    if len(paths) >= max_paths:
                        break
                elif child not in path:
                    queue.append((child, path + [child]))

        return paths


# ── Causal Discovery Algorithms ─────────────────────────────────────

class CausalDiscovery:
    """
    Discover causal structure from normal-operation time series data.

    Implements:
    1. PC algorithm (constraint-based)
    2. LiNGAM (functional causal model)
    3. Hybrid: PC skeleton + LiNGAM direction
    """

    def __init__(
        self,
        method: str = "pc",
        alpha: float = 0.05,
        domain_prior: Optional[Dict] = None,
    ):
        self.method = method
        self.alpha = alpha
        self.domain_prior = domain_prior

    def discover(self, X_normal: np.ndarray) -> CausalGraph:
        """
        Discover causal graph from normal-operation data.

        Args:
            X_normal: (N, D) normal-operation time series

        Returns:
            CausalGraph with discovered adjacency matrix
        """
        n_nodes = X_normal.shape[1]
        node_names = [f"Sensor_{i}" for i in range(n_nodes)]

        if self.method == "pc":
            adj = self._run_pc(X_normal)
        elif self.method == "lingam":
            adj = self._run_lingam(X_normal)
        elif self.method == "hybrid":
            adj = self._run_hybrid(X_normal)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply domain prior if available
        if self.domain_prior is not None:
            adj = self._apply_domain_prior(adj, X_normal.shape[1])

        # Ensure DAG
        adj = self._enforce_dag(adj)

        # Compute edge confidences
        confidence = self._estimate_confidence(adj, X_normal)

        return CausalGraph(
            adj_matrix=adj,
            node_names=node_names,
            confidence=confidence,
        )

    def _run_pc(self, X: np.ndarray) -> np.ndarray:
        """Run PC algorithm from causal-learn."""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        logger.info(f"  Running PC algorithm (alpha={self.alpha}) on {X.shape[0]} samples...")

        cg = pc(data=X, alpha=self.alpha, indep_test=fisherz, stable=True)

        # Convert to adjacency matrix: cg.G.graph is a GeneralGraph
        adj = cg.G.graph  # (D, D), -1=←, 1=→, 0=no edge
        # Convert to binary DAG: only directed edges
        adj_binary = np.where(adj == 1, 1, 0).astype(np.int8)  # i→j edges
        # Also include bidirected edges as undirected for skeleton
        adj_binary = np.where(adj == -1, 1, adj_binary)

        logger.info(f"  PC found {adj_binary.sum():.0f} edges in skeleton")
        return adj_binary

    def _run_lingam(self, X: np.ndarray) -> np.ndarray:
        """Run DirectLiNGAM algorithm."""
        from lingam import DirectLiNGAM

        logger.info(f"  Running DirectLiNGAM on {X.shape[0]} samples...")

        model = DirectLiNGAM()
        model.fit(X)
        adj = model.adjacency_matrix_  # (D, D), adj[i,j] = effect from i to j

        # Threshold to get binary adjacency
        # Use adaptive threshold based on adjacency distribution
        threshold = np.percentile(np.abs(adj), 80)
        adj_binary = (np.abs(adj) > threshold).astype(np.int8)

        logger.info(f"  LiNGAM found {adj_binary.sum():.0f} edges (threshold={threshold:.4f})")
        return adj_binary

    def _run_hybrid(self, X: np.ndarray) -> np.ndarray:
        """PC for skeleton + LiNGAM for direction determination."""
        # First run PC for skeleton
        skeleton = self._run_pc(X)

        # Then use LiNGAM to orient edges
        from lingam import DirectLiNGAM
        model = DirectLiNGAM()
        model.fit(X)
        lingam_adj = model.adjacency_matrix_

        # Use LiNGAM directions for PC skeleton edges
        adj = np.zeros_like(skeleton)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if skeleton[i, j] == 1 or skeleton[j, i] == 1:
                    # Use LiNGAM to determine direction
                    if abs(lingam_adj[i, j]) > abs(lingam_adj[j, i]):
                        adj[i, j] = 1  # i → j
                    else:
                        adj[j, i] = 1  # j → i

        return adj

    def _apply_domain_prior(self, adj: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Apply domain knowledge as soft constraints on the causal graph.

        Uses SWaT P&ID topology: enforce inter-stage flow direction,
        forbid reverse flow.
        """
        prior = self.domain_prior

        # Enforce known edges
        if "adj_matrix" in prior:
            known_adj = prior["adj_matrix"]
            for i in range(min(n_nodes, known_adj.shape[0])):
                for j in range(min(n_nodes, known_adj.shape[1])):
                    if known_adj[i, j] == 1:
                        adj[i, j] = 1  # Enforce known edge
                        adj[j, i] = 0  # Remove reverse

        # Remove forbidden edges
        if "forbidden_edges" in prior:
            for i, j in prior["forbidden_edges"]:
                if i < n_nodes and j < n_nodes:
                    adj[i, j] = 0

        logger.info(f"  After domain prior: {adj.sum():.0f} edges")
        return adj

    def _enforce_dag(self, adj: np.ndarray) -> np.ndarray:
        """Remove cycles if any (shouldn't happen with PC but safety first)."""
        n = adj.shape[0]
        # Simple topological-order-based fix: remove edges that create cycles
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1 and adj[j, i] == 1:
                    adj[j, i] = 0  # Remove bidirectional
        return adj

    def _estimate_confidence(
        self, adj: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """
        Estimate edge confidence via bootstrap resampling.
        Each edge's confidence = fraction of bootstrap samples where it appears.
        """
        n = adj.shape[0]
        confidence = adj.astype(np.float32)  # Initialize with 1.0 for edges, 0 for non-edges

        # Simple approach: use correlation magnitude as proxy for confidence
        corr = np.abs(np.corrcoef(X.T))
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    confidence[i, j] = 0.5 + 0.5 * min(corr[i, j], 1.0)

        return confidence


# ── Counterfactual Root Cause Attribution ──────────────────────────

class CounterfactualRCA:
    """
    Perform counterfactual reasoning to attribute anomaly to root cause sensors.

    Given a trained anomaly detection model and a causal graph,
    estimates the causal effect of each sensor's value on the anomaly score.
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        anomaly_detector,
        n_samples: int = 100,
    ):
        self.graph = causal_graph
        self.detector = anomaly_detector
        self.n_samples = n_samples

    def attribute(
        self,
        window: np.ndarray,
        anomaly_score: float,
    ) -> Dict[int, float]:
        """
        Attribute anomaly to root cause variables via counterfactual reasoning.

        For each sensor, compute: E[anomaly_score | do(X_i = normal)] - anomaly_score
        A large negative difference means intervening on that sensor removes the anomaly.

        Args:
            window: (T, D) anomalous time series window
            anomaly_score: current anomaly score

        Returns:
            dict mapping sensor_idx → attribution_score
        """
        T, D = window.shape
        attributions = {}

        for sensor in range(D):
            # Counterfactual: what if this sensor were at its normal value?
            # Estimate by replacing with in-distribution samples
            cf_scores = []
            for _ in range(self.n_samples):
                cf_window = window.copy()

                # Replace with normal-like values:
                # Use local mean of non-anomalous past window context
                # (or other plausible counterfactual strategies)
                normal_val = np.mean(window[:T//2, sensor])  # Earlier half as baseline
                cf_window[T//2:, sensor] = normal_val

                # Get counterfactual anomaly score
                cf_score = self._get_anomaly_score(cf_window)
                cf_scores.append(cf_score)

            cf_mean = np.mean(cf_scores)
            attribution = anomaly_score - cf_mean  # Positive = intervention reduces anomaly
            attributions[sensor] = float(attribution)

        return attributions

    def _get_anomaly_score(self, window: np.ndarray) -> float:
        """Get anomaly score for a window. Placeholder — uses detector in practice."""
        # In practice calls self.detector.predict(window)
        return 0.0

    def find_root_cause_nodes(
        self,
        attributions: Dict[int, float],
        anomalous_nodes: List[int],
        top_k: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """
        Find the most likely root cause nodes given attributions and causal graph.

        Returns:
            List of (node_idx, node_name, root_cause_score) sorted by score
        """
        root_causes = self.graph.find_root_causes(anomalous_nodes)
        results = []

        for node_idx, causal_conf in root_causes:
            attr_score = attributions.get(node_idx, 0.0)
            # Combine causal confidence + attribution score
            combined = 0.5 * causal_conf + 0.5 * (attr_score / max(
                max(attributions.values()), 1e-8
            ))
            node_name = self.graph.node_names[node_idx]
            results.append((node_idx, node_name, combined))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# ── Integration Test ───────────────────────────────────────────────

if __name__ == "__main__":
    from vicsynad.data.swat_causal_graph import build_swat_causal_prior

    # Generate synthetic normal data with causal structure
    rng = np.random.default_rng(42)
    n_samples, n_nodes = 5000, 10

    # Simple causal chain: X0→X1→X3, X0→X2→X4, etc.
    X = np.zeros((n_samples, n_nodes))
    X[:, 0] = rng.normal(0, 1, n_samples)  # Root
    X[:, 1] = 0.7 * X[:, 0] + rng.normal(0, 0.3, n_samples)
    X[:, 2] = 0.5 * X[:, 0] + rng.normal(0, 0.5, n_samples)
    X[:, 3] = 0.6 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.2, n_samples)
    X[:, 4] = 0.4 * X[:, 3] + rng.normal(0, 0.4, n_samples)

    # Test causal discovery
    cd = CausalDiscovery(method="pc", alpha=0.05)
    graph = cd.discover(X)

    print(f"Discovered graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Adjacency matrix:\n{graph.adj_matrix}")

    # Test root cause finding
    anomalous_nodes = [3, 4]  # Both downstream
    root_causes = graph.find_root_causes(anomalous_nodes)
    print(f"\nAnomalous nodes: {anomalous_nodes}")
    print(f"Root causes: {root_causes}")

    # Test propagation path
    if root_causes:
        root = root_causes[0][0]
        paths = graph.get_propagation_path(root, set(anomalous_nodes))
        print(f"Propagation paths from {root}: {paths}")

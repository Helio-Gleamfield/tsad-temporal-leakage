"""
Chain-of-Thought Explanation Generator (Layer 4 of ViCSynAD).

Generates structured natural language explanations for detected anomalies
by combining visual evidence, causal reasoning, and attribution scores.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ── CoT Prompt Templates ───────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert industrial anomaly analyst specializing in cyber-physical systems and water treatment processes. Your task is to analyze time series anomalies detected by an automated system and provide a structured, causal explanation.

Follow this chain-of-thought reasoning process:
1. PATTERN IDENTIFICATION: Describe what visual patterns in the time series indicate anomalous behavior
2. ANOMALY CHARACTERIZATION: Identify which sensors show abnormal readings and how they deviate from normal
3. CAUSAL ROOT CAUSE: Using the provided causal graph, identify the most likely root cause sensor(s)
4. PROPAGATION PATH: Trace how the anomaly propagated through the system
5. RECOMMENDATION: Suggest actionable steps to address the anomaly

Be specific, technical, and grounded in the data provided. Do not speculate beyond the evidence."""

ANOMALY_ANALYSIS_PROMPT = """[SYSTEM CONTEXT]
You are analyzing a water treatment cyber-physical system (SWaT testbed) with 6 stages:
P1 (Raw Water Intake) → P2 (Chemical Dosing) → P3 (UF Feed) → P4 (Ultrafiltration) → P5 (Reverse Osmosis) → P6 (Distribution).

[DETECTION RESULT]
Anomaly detected with confidence score: {anomaly_score:.3f}

[VISUAL EVIDENCE]
The attached image shows the time series data for the anomalous window (256 time steps).
Please examine the visual patterns carefully.

[CAUSAL GRAPH]
The known causal relationships between sensors are:
{causal_graph}

[ATTRIBUTION SCORES]
The causal attribution scores for the most affected sensors are:
{attribution_scores}

[ROOT CAUSE ANALYSIS]
The automated causal analysis identified these potential root cause sensors:
{root_cause}

[PROPAGATION PATH]
The likely anomaly propagation path is:
{propagation_path}

Please provide a structured analysis following the chain-of-thought reasoning:
1. PATTERN IDENTIFICATION:
2. ANOMALY CHARACTERIZATION:
3. CAUSAL ROOT CAUSE:
4. PROPAGATION PATH:
5. RECOMMENDATION:"""


# ── Explanation Generator ──────────────────────────────────────────

@dataclass
class AnomalyExplanation:
    """Structured output of the CoT explanation."""
    pattern_identification: str
    anomaly_characterization: str
    causal_root_cause: str
    propagation_path: str
    recommendation: str
    raw_text: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "pattern_identification": self.pattern_identification,
            "anomaly_characterization": self.anomaly_characterization,
            "causal_root_cause": self.causal_root_cause,
            "propagation_path": self.propagation_path,
            "recommendation": self.recommendation,
            "raw_text": self.raw_text,
        }

    def to_markdown(self) -> str:
        return f"""## Anomaly Explanation

### 1. Pattern Identification
{self.pattern_identification}

### 2. Anomaly Characterization
{self.anomaly_characterization}

### 3. Causal Root Cause
{self.causal_root_cause}

### 4. Propagation Path
{self.propagation_path}

### 5. Recommendation
{self.recommendation}
"""


class CoTExplainer:
    """
    Chain-of-Thought anomaly explainer.

    Generates structured explanations combining:
    - Visual evidence from TS images
    - Causal reasoning from discovered causal graph
    - Attribution scores from counterfactual analysis
    """

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.analysis_prompt_template = ANOMALY_ANALYSIS_PROMPT

    def build_prompt(
        self,
        anomaly_score: float,
        causal_graph_summary: str,
        attribution_scores: Dict[int, float],
        root_causes: List[Tuple[int, str, float]],
        propagation_paths: List[List[int]],
        node_names: List[str],
    ) -> str:
        """
        Construct the CoT prompt with all available evidence.

        Args:
            anomaly_score: model's anomaly detection score
            causal_graph_summary: text summary of causal graph
            attribution_scores: dict of sensor_idx → attribution
            root_causes: list of (idx, name, score) root cause candidates
            propagation_paths: list of propagation paths
            node_names: list of sensor names
        """
        # Format causal graph
        cg_text = causal_graph_summary

        # Format attribution scores (top 10)
        sorted_attr = sorted(
            attribution_scores.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        attr_text = "\n".join(
            f"  Sensor {idx} ({node_names[idx] if idx < len(node_names) else 'Unknown'}): "
            f"attribution = {val:.4f}"
            for idx, val in sorted_attr
        )

        # Format root causes
        rc_text = "\n".join(
            f"  #{i+1}: Sensor {idx} ({name}) - score: {score:.3f}"
            for i, (idx, name, score) in enumerate(root_causes)
        )

        # Format propagation paths
        pp_text = ""
        for i, path in enumerate(propagation_paths):
            path_str = " → ".join(
                node_names[n] if n < len(node_names) else f"Sensor_{n}"
                for n in path
            )
            pp_text += f"  Path {i+1}: {path_str}\n"

        prompt = self.analysis_prompt_template.format(
            anomaly_score=anomaly_score,
            causal_graph=cg_text,
            attribution_scores=attr_text,
            root_cause=rc_text,
            propagation_path=pp_text if pp_text else "  No clear propagation path identified",
        )

        return prompt

    def parse_explanation(self, raw_text: str) -> AnomalyExplanation:
        """Parse the VLM-generated text into structured sections."""
        sections = {
            "pattern_identification": "",
            "anomaly_characterization": "",
            "causal_root_cause": "",
            "propagation_path": "",
            "recommendation": "",
        }

        current_section = None
        section_markers = {
            "pattern identification": "pattern_identification",
            "anomaly characterization": "anomaly_characterization",
            "causal root cause": "causal_root_cause",
            "propagation path": "propagation_path",
            "recommendation": "recommendation",
        }

        for line in raw_text.split("\n"):
            line_lower = line.strip().lower()

            # Check for section headers
            for marker, key in section_markers.items():
                if marker in line_lower and (
                    line_lower.startswith("###") or
                    line_lower.startswith(str(list(section_markers.keys()).index(marker) + 1) + ".") or
                    marker.upper() in line_lower[:30]
                ):
                    current_section = key
                    # Clean up the header
                    remaining = line.split(":", 1)[-1].strip() if ":" in line else ""
                    if remaining:
                        sections[current_section] += remaining + " "
                    break
            else:
                if current_section:
                    sections[current_section] += line + " "

        # Clean up whitespace
        for key in sections:
            sections[key] = sections[key].strip()

        return AnomalyExplanation(
            pattern_identification=sections["pattern_identification"] or "Not extracted",
            anomaly_characterization=sections["anomaly_characterization"] or "Not extracted",
            causal_root_cause=sections["causal_root_cause"] or "Not extracted",
            propagation_path=sections["propagation_path"] or "Not extracted",
            recommendation=sections["recommendation"] or "Not extracted",
            raw_text=raw_text,
        )

    def explain(
        self,
        anomaly_score: float,
        causal_graph_summary: str,
        attribution_scores: Dict[int, float],
        root_causes: List[Tuple[int, str, float]],
        propagation_paths: List[List[int]],
        node_names: List[str],
        vlm_extractor=None,
        anomaly_image=None,
    ) -> AnomalyExplanation:
        """
        Full explanation pipeline: build prompt → VLM inference → parse output.

        If vlm_extractor is None, returns the prompt only (offline mode).
        """
        prompt = self.build_prompt(
            anomaly_score=anomaly_score,
            causal_graph_summary=causal_graph_summary,
            attribution_scores=attribution_scores,
            root_causes=root_causes,
            propagation_paths=propagation_paths,
            node_names=node_names,
        )

        if vlm_extractor is not None and anomaly_image is not None:
            causal_info = {
                "graph_summary": causal_graph_summary,
                "attribution": "\n".join(
                    f"Sensor {k}: {v:.4f}" for k, v in
                    sorted(attribution_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                ),
                "root_cause": root_causes[0][1] if root_causes else "Unknown",
                "propagation": "\n".join(
                    "→".join(node_names[n] if n < len(node_names) else f"S{n}"
                             for n in path)
                    for path in propagation_paths[:3]
                ) if propagation_paths else "Unknown",
            }
            raw_text = vlm_extractor.generate_explanation(
                image=anomaly_image,
                causal_info=causal_info,
                prompt_template=prompt,
            )
            return self.parse_explanation(raw_text)
        else:
            # Offline mode: return placeholder with prompt
            return AnomalyExplanation(
                pattern_identification="[VLM offline - prompt generated]",
                anomaly_characterization="[VLM offline - prompt generated]",
                causal_root_cause=f"Root cause candidates: {root_causes[:3] if root_causes else 'None'}",
                propagation_path=f"Paths: {propagation_paths[:2] if propagation_paths else 'None'}",
                recommendation="[VLM offline - prompt generated]",
                raw_text=prompt,
            )


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    explainer = CoTExplainer()

    # Mock data
    attribution = {0: 0.87, 1: 0.43, 3: 0.31, 4: 0.15}
    root_causes = [(0, "FIT101", 0.87), (1, "LIT101", 0.43)]
    paths = [[0, 1, 3], [0, 2, 3]]

    explanation = explainer.explain(
        anomaly_score=0.92,
        causal_graph_summary="P1→P2→P3→P4→P5→P6 (linear water treatment flow)",
        attribution_scores=attribution,
        root_causes=root_causes,
        propagation_paths=paths,
        node_names=[f"Sensor_{i}" for i in range(10)],
    )

    print(explanation.to_markdown())

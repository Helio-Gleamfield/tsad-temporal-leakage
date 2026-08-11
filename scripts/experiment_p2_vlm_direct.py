"""
P2 VLM API Direct Vision Experiment
====================================
Tests whether qwen3-vl-plus can directly detect anomalies in multivariate TS plots.
Compares three approaches:
  1. VLM direct zero-shot visual judgment (structured CoT prompt)
  2. VLM direct zero-shot (simple yes/no prompt)
  3. Pure statistical baseline (Z-score)

Key literature grounding:
  - VisualTimeAnomaly (Xu et al., WWW'26): First systematic benchmark of MLLM TSAD
  - VLM4TS (He et al., 2025): Two-stage VLM pipeline, zero-shot F1 improvement 24.6%
  - AnomSeer (Zhang et al., ICML'26): Qwen2.5-VL-7B > GPT-4o on TSAD with RL training
  - VisLit-VLM-Eval (Pandey et al., EuroVis'25): Anomaly detection only 25-30% across VLMs

Usage:
    python scripts/experiment_p2_vlm_direct.py --n_windows 100 --model qwen3-vl-plus
"""
import sys, json, time, base64, argparse, warnings
from pathlib import Path
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              accuracy_score, confusion_matrix)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "p0_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Alibaba Cloud DashScope — API key from environment variable
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")  # Latest strongest VL model

if not API_KEY:
    raise RuntimeError(
        "DASHSCOPE_API_KEY not set. Copy .env.example to .env and fill in your API key, "
        "or set the environment variable: $env:DASHSCOPE_API_KEY='your_key'"
    )

# ── Plot style ──
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]


def render_ts_window(values, title="Time Series Window"):
    """Render multivariate time series as an RGB image. Returns base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    T, D = values.shape
    n_plot = min(D, 7)
    time_axis = np.arange(T)

    # Left: Multi-channel line chart
    for i in range(n_plot):
        ax1.plot(time_axis, values[:, i], color=CB_PALETTE[i % len(CB_PALETTE)],
                 linewidth=0.8, alpha=0.8)

    ax1.set_title(f"{title} — {D} channels", fontsize=10)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Value (Z-score norm.)")

    # Right: Correlation heatmap
    corr = np.corrcoef(values[:, :min(D, 20)].T)
    sns.heatmap(corr, ax=ax2, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.7}, square=True)
    ax2.set_title("Sensor Correlation Matrix", fontsize=10)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_ts_simple(values, title="Time Series"):
    """Render a simpler single-panel line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    T, D = values.shape
    time_axis = np.arange(T)
    n_plot = min(D, 10)
    for i in range(n_plot):
        ax.plot(time_axis, values[:, i], color=CB_PALETTE[i % len(CB_PALETTE)],
                linewidth=0.7, alpha=0.8)
    ax.set_title(f"{title} — {D} sensors, {T} time steps", fontsize=11)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════
# PROMPT STRATEGIES (informed by literature)
# ═══════════════════════════════════════════════

SIMPLE_PROMPT = """You are looking at a multivariate time series line chart. Each line is a different sensor.

Question: Is there an anomaly in this time series window? An anomaly means any unusual pattern, sudden spike, drop, or deviation from normal behavior.

Answer with ONLY ONE WORD: "YES" or "NO"."""

COT_PROMPT = """You are analyzing a multivariate time series from an industrial cyber-physical system.
Each colored line represents a different sensor reading over time.

Perform this structured analysis (5 steps):

Step 1 — PATTERN: Describe the overall pattern. Do you see any sudden spikes, drops, level shifts, or erratic oscillations?
Step 2 — CHARACTERIZATION: If any unusual patterns exist, characterize their magnitude, duration, and which sensors are involved.
Step 3 — ROOT CAUSE HYPOTHESIS: Which sensor appears to deviate first? Does the deviation seem to propagate to other sensors?
Step 4 — ANOMALY JUDGMENT: Based on your analysis, is this window ANOMALOUS or NORMAL?
Step 5 — CONFIDENCE: Rate your confidence: HIGH / MEDIUM / LOW.

FORMAT YOUR ANSWER EXACTLY AS:
ANOMALY: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [One sentence summary]"""


# ═══════════════════════════════════════════════
# VLM INFERENCE
# ═══════════════════════════════════════════════

def vlm_direct_judgment(client, img_b64, prompt_type="cot"):
    """Send image to VLM, get anomaly judgment."""
    prompt = COT_PROMPT if prompt_type == "cot" else SIMPLE_PROMPT

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=512,
            temperature=0.0  # Deterministic for evaluation
        )
        text = response.choices[0].message.content.strip()
        return parse_vlm_response(text)
    except Exception as e:
        return {"prediction": None, "confidence": None, "raw": str(e), "error": True}


def parse_vlm_response(text):
    """Parse VLM response to extract anomaly judgment."""
    text_upper = text.upper()
    is_anomaly = None
    confidence = None

    # Parse ANOMALY: YES/NO
    if "ANOMALY: YES" in text_upper or "ANOMALY:YES" in text_upper:
        is_anomaly = 1
    elif "ANOMALY: NO" in text_upper or "ANOMALY:NO" in text_upper:
        is_anomaly = 0
    elif "ANOMALOUS" in text_upper and "NORMAL" not in text_upper:
        is_anomaly = 1
    elif text_upper.strip() in ["YES", "NO"]:
        is_anomaly = 1 if text_upper.strip() == "YES" else 0
    else:
        # Fallback: look for YES/NO anywhere in first line
        first_line = text_upper.split("\n")[0].strip()
        if first_line.startswith("YES"):
            is_anomaly = 1
        elif first_line.startswith("NO"):
            is_anomaly = 0

    # Parse CONFIDENCE
    if "CONFIDENCE: HIGH" in text_upper:
        confidence = "HIGH"
    elif "CONFIDENCE: MEDIUM" in text_upper:
        confidence = "MEDIUM"
    elif "CONFIDENCE: LOW" in text_upper:
        confidence = "LOW"

    return {
        "prediction": is_anomaly,
        "confidence": confidence,
        "raw": text[:500],
        "error": False
    }


# ═══════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=100, help="Number of windows to test")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--prompt", type=str, default="cot", choices=["cot", "simple"])
    args = parser.parse_args()

    print("=" * 70)
    print(f"  P2 VLM DIRECT VISION EXPERIMENT")
    print(f"  Model: {args.model}  |  Prompt: {args.prompt}  |  N={args.n_windows}")
    print("=" * 70)

    # ── Load SWaT data ──
    print("\n[1] Loading SWaT data...")
    swat_path = DATA_ROOT / "SWaT" / "AllInOne"
    train_X = np.load(swat_path / "train.npy")
    train_y = np.load(swat_path / "train_label.npy")
    test_X = np.load(swat_path / "test.npy")
    test_y = np.load(swat_path / "test_label.npy")

    # Use test set only (train has 0 anomalies per HANDOFF.md)
    # Fit scaler on normal training data
    normal_mask = train_y == 0
    mean = train_X[normal_mask].mean(axis=0)
    std = train_X[normal_mask].std(axis=0) + 1e-8
    X_norm = (test_X - mean) / std

    print(f"  Test set: {X_norm.shape}, Anomaly ratio: {test_y.mean():.3f}")

    # ── Generate windows ──
    print(f"\n[2] Generating {args.n_windows} windows (L=256, S=128)...")
    L, S = 256, 128
    windows, labels = [], []
    seen_anom, seen_normal = 0, 0
    target_each = args.n_windows // 2

    for i in range(0, len(X_norm) - L, S):
        window = X_norm[i:i+L]
        label = 1 if np.any(test_y[i:i+L] > 0.5) else 0

        if label == 1 and seen_anom < target_each:
            windows.append(window)
            labels.append(label)
            seen_anom += 1
        elif label == 0 and seen_normal < target_each:
            windows.append(window)
            labels.append(label)
            seen_normal += 1

        if seen_anom >= target_each and seen_normal >= target_each:
            break

    print(f"  Selected: {len(windows)} windows ({sum(labels)} anomalous, {len(labels)-sum(labels)} normal)")

    # ── Z-score baseline ──
    print(f"\n[3] Computing Z-score baseline...")
    z_scores = np.max(np.abs(windows - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1), axis=(1, 2))
    z_auc = roc_auc_score(labels, z_scores)
    print(f"  Z-score AUC: {z_auc:.4f}")

    # ── VLM Direct Judgment ──
    print(f"\n[4] Running VLM direct visual judgment ({args.model}, {args.prompt} prompt)...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    predictions = []
    results = []
    for idx, (window, true_label) in enumerate(zip(windows, labels)):
        # Render window as image
        img_b64 = render_ts_simple(window, f"Window #{idx}")

        # VLM judgment
        result = vlm_direct_judgment(client, img_b64, prompt_type=args.prompt)

        predictions.append(result["prediction"])
        results.append({
            "idx": idx,
            "true_label": int(true_label),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_response": result["raw"][:200] if not result["error"] else "ERROR",
        })

        status = "✓" if result["prediction"] == true_label else "✗"
        if result["prediction"] is None:
            status = "?"
        print(f"  [{idx+1:3d}/{len(windows)}] {status} True={true_label}, Pred={result['prediction']}, "
              f"Conf={result['confidence'] or 'N/A'}")

        # Rate limiting
        time.sleep(0.3)

    # ── Evaluation ──
    valid_idx = [i for i, p in enumerate(predictions) if p is not None]
    valid_preds = [predictions[i] for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]

    print(f"\n{'='*70}")
    print(f"  RESULTS: VLM Direct Visual Anomaly Detection")
    print(f"{'='*70}")
    print(f"  Valid predictions: {len(valid_preds)}/{len(predictions)} ({len(valid_preds)/len(predictions)*100:.0f}%)")

    if len(valid_preds) > 0:
        acc = accuracy_score(valid_labels, valid_preds)
        prec = precision_score(valid_labels, valid_preds, zero_division=0)
        rec = recall_score(valid_labels, valid_preds, zero_division=0)
        f1 = f1_score(valid_labels, valid_preds, zero_division=0)
        try:
            auc = roc_auc_score(valid_labels, valid_preds)
        except:
            auc = float('nan')
        cm = confusion_matrix(valid_labels, valid_preds)

        print(f"\n  Accuracy:       {acc:.4f}")
        print(f"  Precision:      {prec:.4f}")
        print(f"  Recall:         {rec:.4f}")
        print(f"  F1:             {f1:.4f}")
        print(f"  AUC (binary):   {auc:.4f}")
        print(f"  Confusion:      TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

        # Confidence analysis
        high_conf = [i for i in valid_idx if results[i]["confidence"] == "HIGH"]
        if high_conf:
            high_preds = [predictions[i] for i in high_conf]
            high_labels = [labels[i] for i in high_conf]
            high_acc = accuracy_score(high_labels, high_preds)
            print(f"\n  High-confidence subset (n={len(high_conf)}): Acc={high_acc:.4f}")

    print(f"\n  Z-score baseline AUC: {z_auc:.4f}")
    print(f"  VLM accuracy:         {acc:.4f}" if len(valid_preds) > 0 else "  VLM: insufficient data")

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: VLM Direct vs Feature Extraction vs Statistical")
    print(f"{'='*70}")
    print(f"  {'Method':<40s} {'Metric':<10s} {'Value':>8s}")
    print(f"  {'-'*58}")
    print(f"  {'Z-score (zero-param statistical)':<40s} {'AUC':<10s} {z_auc:>8.4f}")
    print(f"  {'qwen3-vl-plus direct vision (CoT)':<40s} {'F1':<10s} {f1:>8.4f}" if len(valid_preds) > 0 else "")
    if len(valid_preds) > 0:
        print(f"  {'qwen3-vl-plus direct vision (CoT)':<40s} {'Acc':<10s} {acc:>8.4f}")
    print(f"  {'Previously: ViCSynAD VLM features':<40s} {'DAUC':<10s} {'0.0000':>8s}")

    # ── Save results ──
    output = {
        "experiment": "P2_VLM_Direct_Vision",
        "model": args.model,
        "prompt_type": args.prompt,
        "n_windows": len(windows),
        "n_anomalous": int(sum(labels)),
        "zscore_auc": float(z_auc),
        "vlm_accuracy": float(acc) if len(valid_preds) > 0 else None,
        "vlm_precision": float(prec) if len(valid_preds) > 0 else None,
        "vlm_recall": float(rec) if len(valid_preds) > 0 else None,
        "vlm_f1": float(f1) if len(valid_preds) > 0 else None,
        "vlm_valid_rate": len(valid_preds) / len(predictions) if predictions else 0,
        "per_window_results": results,
    }

    out_path = RESULTS_DIR / "p2_vlm_direct_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()

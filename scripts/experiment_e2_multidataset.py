"""
E2: Multi-Dataset Temporal Leakage Quantification
==================================================
Purpose: Quantify temporal leakage prevalence across SWaT, TEP, and MSL.

For each dataset, compare:
  - Random split AUC (standard in the field — LEAKY)
  - Temporal split AUC (our proposed honest evaluation)

Key metric: ΔAUC = AUC_random - AUC_temporal (the "leakage gap")
Hypothesis: Random split inflates AUC by 0.20-0.35 across all datasets.
"""
import sys; sys.path.insert(0, "src")
import torch, torch.nn as nn, numpy as np, time, json, logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.WARNING)

from vicsynad.config import DATA_ROOT, EXPERIMENT_DIR
from vicsynad.data.processor import load_swat, load_tep, DataPreprocessor
from vicsynad.modules.ts_vis import extract_enhanced_statistical_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LSTM-AE Baseline ─────────────────────────────────────────────────
class LSTM_AE(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.enc = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.dec = nn.LSTM(h * 2, h, batch_first=True)
        self.out = nn.Linear(h, d)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        h = h.permute(1, 0, 2).reshape(x.shape[0], 1, -1).repeat(1, x.shape[1], 1)
        o, _ = self.dec(h)
        return self.out(o)


@dataclass
class DatasetResult:
    dataset: str
    n_samples: int
    n_sensors: int
    anomaly_ratio: float
    random_split: Dict[str, float]   # AUC, AP for LSTM-AE under random split
    temporal_split: Dict[str, float]  # AUC, AP for LSTM-AE under temporal split
    leakage_gap: float                # ΔAUC = random - temporal
    zscore_temporal_auc: float        # Simple Z-score baseline
    isolforest_temporal_auc: float    # Isolation Forest baseline


def temporal_split(samples: List, train_ratio: float = 0.7):
    """Create strict temporal split (first 70% train, last 30% test)."""
    n = len(samples)
    n_train = int(n * train_ratio)
    return samples[:n_train], samples[n_train:]


def random_split(samples: List, train_ratio: float = 0.7, seed: int = 42):
    """Create random split (standard in the field — CAUSES LEAKAGE)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    n_train = int(len(samples) * train_ratio)
    return [samples[i] for i in indices[:n_train]], [samples[i] for i in indices[n_train:]]


def evaluate_lstm_ae(
    train_w: np.ndarray, train_l: np.ndarray,
    test_w: np.ndarray, test_l: np.ndarray,
    epochs: int = 15, hidden: int = 64, lr: float = 1e-3,
) -> Dict[str, float]:
    """Train LSTM-AE and evaluate."""
    d = train_w.shape[2]
    model = LSTM_AE(d, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.FloatTensor(train_w).to(device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = Xtr[perm[i:i+64]]
            loss = nn.MSELoss()(model(b), b)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    Xte = torch.FloatTensor(test_w).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        scores = nn.MSELoss(reduction="none")(model(Xte), Xte).mean(dim=(1, 2)).cpu().numpy()

    return {
        "auc": round(roc_auc_score(test_l, scores), 4),
        "ap": round(average_precision_score(test_l, scores), 4),
    }


def evaluate_zscore(test_w: np.ndarray, test_l: np.ndarray, train_w: np.ndarray) -> float:
    """Simple Z-score baseline: max(|x - mu| / sigma) per window."""
    mu = train_w.mean(axis=(0, 1))
    sigma = train_w.std(axis=(0, 1)) + 1e-8
    z_scores = np.max(np.abs((test_w - mu) / sigma), axis=(1, 2))
    return round(roc_auc_score(test_l, z_scores), 4)


def evaluate_isolation_forest(
    train_w: np.ndarray, train_l: np.ndarray,
    test_w: np.ndarray, test_l: np.ndarray,
) -> float:
    """Isolation Forest on flattened windows."""
    Xtr = train_w.reshape(len(train_w), -1)
    Xte = test_w.reshape(len(test_w), -1)
    # PCA reduce to handle high-dimensional data
    n_comp = min(30, Xtr.shape[1])
    pca = PCA(n_comp, random_state=42)
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(Xtr_pca)
    scores = -iso.score_samples(Xte_pca)
    return round(roc_auc_score(test_l, scores), 4)


def run_dataset_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    window_size: int = 256,
    stride: int = 64,
    max_samples: int = 1500,
) -> DatasetResult:
    """Run full leakage quantification for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | shape={X.shape} | anomaly={y.mean():.3f}")
    print(f"{'='*60}")

    pp = DataPreprocessor(window_size=window_size, stride=stride)
    pp.fit_scaler(X[y == 0])
    samples = pp.process_dataset(X, y, dataset_name)

    # Cap samples for fair comparison
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"  Windows: {len(samples)}, Anomaly ratio: {np.mean([s.label for s in samples]):.3f}")

    # ── Temporal Split ──
    tr_t, te_t = temporal_split(samples)
    tr_w_t = np.stack([s.values for s in tr_t])
    tr_l_t = np.array([s.label for s in tr_t])
    te_w_t = np.stack([s.values for s in te_t])
    te_l_t = np.array([s.label for s in te_t])
    print(f"  Temporal: train={len(tr_t)}(a={tr_l_t.mean():.3f}), test={len(te_t)}(a={te_l_t.mean():.3f})")

    temporal_metrics = evaluate_lstm_ae(tr_w_t, tr_l_t, te_w_t, te_l_t)
    zscore_auc = evaluate_zscore(te_w_t, te_l_t, tr_w_t)
    iso_auc = evaluate_isolation_forest(tr_w_t, tr_l_t, te_w_t, te_l_t)
    print(f"  Temporal LSTM-AE: AUC={temporal_metrics['auc']:.4f}")
    print(f"  Temporal Z-score: AUC={zscore_auc:.4f}")
    print(f"  Temporal IsolForest: AUC={iso_auc:.4f}")

    # ── Random Split ──
    tr_r, te_r = random_split(samples)
    tr_w_r = np.stack([s.values for s in tr_r])
    tr_l_r = np.array([s.label for s in tr_r])
    te_w_r = np.stack([s.values for s in te_r])
    te_l_r = np.array([s.label for s in te_r])
    print(f"  Random:   train={len(tr_r)}(a={tr_l_r.mean():.3f}), test={len(te_r)}(a={te_l_r.mean():.3f})")

    random_metrics = evaluate_lstm_ae(tr_w_r, tr_l_r, te_w_r, te_l_r)
    print(f"  Random LSTM-AE:  AUC={random_metrics['auc']:.4f}")

    leakage_gap = random_metrics["auc"] - temporal_metrics["auc"]
    print(f"  LEAKAGE GAP (ΔAUC): {leakage_gap:+.4f}")

    return DatasetResult(
        dataset=dataset_name,
        n_samples=len(samples),
        n_sensors=X.shape[1],
        anomaly_ratio=round(float(y.mean()), 4),
        random_split=random_metrics,
        temporal_split=temporal_metrics,
        leakage_gap=round(leakage_gap, 4),
        zscore_temporal_auc=zscore_auc,
        isolforest_temporal_auc=iso_auc,
    )


def main():
    print("=" * 70)
    print("E2: Multi-Dataset Temporal Leakage Quantification")
    print("=" * 70)

    results: List[DatasetResult] = []

    # ── SWaT ──
    print("\n[1/3] Loading SWaT...")
    _, _, test_X, test_y = load_swat(DATA_ROOT / "SWaT" / "AllInOne")
    results.append(run_dataset_experiment(test_X, test_y, "SWaT"))

    # ── TEP ──
    print("\n[2/3] Loading TEP...")
    try:
        tep_X, tep_y = load_tep(DATA_ROOT / "TEP" / "new_tep_datasets-main")
        results.append(run_dataset_experiment(tep_X, tep_y, "TEP"))
    except Exception as e:
        print(f"  TEP load failed: {e}")
        print("  Trying alternative TEP path...")
        # Try other TEP variants
        tep_dir = DATA_ROOT / "TEP" / "new_tep_datasets-main"
        for csv_file in sorted(tep_dir.glob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if df.shape[1] > 2:
                    X = df.iloc[:, :-1].values.astype(np.float32)
                    y = (df.iloc[:, -1].values != 0).astype(np.int8)
                    results.append(run_dataset_experiment(X, y, f"TEP ({csv_file.name})"))
                    break
            except Exception:
                continue

    # ── MSL (from mTSBench) ──
    print("\n[3/3] Loading MSL...")
    try:
        import pandas as pd
        msl_dir = DATA_ROOT / "mTSBench" / "MSL"
        if msl_dir.exists():
            csv_files = sorted(msl_dir.glob("*.csv"))
            if csv_files:
                # Combine all MSL sub-datasets
                all_X, all_y = [], []
                for csv_file in csv_files[:3]:  # First 3 MSL files
                    df = pd.read_csv(csv_file)
                    # Find label column (usually "label" or last column)
                    if "label" in [c.lower() for c in df.columns]:
                        label_col = [c for c in df.columns if c.lower() == "label"][0]
                        X_cols = [c for c in df.columns if c != label_col]
                        X = df[X_cols].values.astype(np.float32)
                        y = df[label_col].values.astype(np.int8)
                    else:
                        X = df.iloc[:, :-1].values.astype(np.float32)
                        y = df.iloc[:, -1].values.astype(np.int8)
                    all_X.append(X)
                    all_y.append(y)
                if all_X:
                    # Use the longest sub-dataset
                    longest_idx = np.argmax([len(x) for x in all_X])
                    results.append(run_dataset_experiment(
                        all_X[longest_idx], all_y[longest_idx],
                        f"MSL ({csv_files[longest_idx].stem})"
                    ))
    except Exception as e:
        print(f"  MSL load failed: {e}")

    # ── Summary Table ──
    print("\n\n" + "=" * 80)
    print("E2 RESULTS: Temporal Leakage Across Datasets")
    print("=" * 80)
    header = f"{'Dataset':<25s} {'N':>6s} {'Dim':>5s} {'Anom%':>7s} "
    header += f"{'Rand AUC':>9s} {'Temp AUC':>9s} {'ΔAUC':>8s} "
    header += f"{'Zsc AUC':>8s} {'IF AUC':>8s}"
    print(header)
    print("-" * 80)

    total_gap = 0.0
    for r in results:
        print(
            f"{r.dataset:<25s} {r.n_samples:>6d} {r.n_sensors:>5d} {r.anomaly_ratio:>7.3f} "
            f"{r.random_split['auc']:>9.4f} {r.temporal_split['auc']:>9.4f} {r.leakage_gap:>+8.4f} "
            f"{r.zscore_temporal_auc:>8.4f} {r.isolforest_temporal_auc:>8.4f}"
        )
        total_gap += r.leakage_gap

    if results:
        avg_gap = total_gap / len(results)
        print("-" * 80)
        print(f"  Average leakage gap (ΔAUC) across {len(results)} datasets: {avg_gap:+.4f}")
        print(f"  Conclusion: Random split inflates AUC by {avg_gap:.2f} on average.")

    # ── Save ──
    output = {
        "experiment": "E2_MULTIDATASET_TEMPORAL_LEAKAGE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "method": "LSTM-AE (hidden=64, epochs=15) + Z-score + Isolation Forest",
        "leakage_gap_average": round(avg_gap, 4) if results else None,
        "results": [asdict(r) for r in results],
    }

    output_dir = EXPERIMENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e2_multidataset_leakage.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

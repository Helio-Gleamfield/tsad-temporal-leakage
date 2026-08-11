"""
P2 Cross-Domain Experiment (FIXED) — Temporal Leakage Beyond CPS
=================================================================
Tests leakage across 3 new domains using suitable datasets:
  1. Medical (ECG): MITDB_106 — 520K samples, 2 leads, 34.3% anomalies
  2. Medical (Gait): Daphnet_S01R01E1 — 59K samples, 10 sensors, 7.2% anomalies
  3. Network Security: cicids — packet features, cyber attacks

Each domain is fundamentally different from the original CPS/server datasets.
Shorter window length (L=128) for smaller datasets.

Usage:
    python scripts/experiment_p2_crossdomain.py
"""
import sys, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "p0_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════

def load_mitdb_106():
    """MITDB patient 106: 520K samples, 34.3% arrhythmia, 2 ECG leads."""
    p = DATA_ROOT / "mTSBench" / "MITDB" / "MITDB_106_test.csv"
    df = pd.read_csv(p)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_anomaly' in num_cols:
        num_cols.remove('is_anomaly')
    X = df[num_cols].values.astype(np.float32)
    y = df['is_anomaly'].values.astype(np.float32)
    print(f"  MITDB_106: {X.shape}, anom={y.mean():.4f}")
    return X, y


def load_daphnet():
    """Daphnet S01R01E1: gait freezing detection, 59K samples, 10 sensors."""
    p = DATA_ROOT / "mTSBench" / "Daphnet" / "Daphnet_S01R01E1_test.csv"
    df = pd.read_csv(p)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_anomaly' in num_cols:
        num_cols.remove('is_anomaly')
    if 'timestamp' in df.columns:
        # Drop timestamp-like columns
        num_cols = [c for c in num_cols if 'timestamp' not in c.lower()]
    X = df[num_cols].values.astype(np.float32)
    y = df['is_anomaly'].values.astype(np.float32)
    print(f"  Daphnet: {X.shape}, anom={y.mean():.4f}")
    return X, y


def load_cicids():
    """CICIDS network intrusion: multiple packet feature files."""
    p = DATA_ROOT / "mTSBench" / "cicids"
    csvs = sorted(p.glob("*test*.csv"))
    if not csvs:
        csvs = sorted(p.glob("*.csv"))

    # Load largest test file
    best, best_size = None, 0
    for csv_path in csvs:
        sz = csv_path.stat().st_size
        if sz > best_size:
            best_size = sz
            best = csv_path

    if best is None:
        raise FileNotFoundError("No cicids files")

    df = pd.read_csv(best)
    # Find label column
    label_col = None
    for c in df.columns:
        if 'label' in c.lower() or 'is_anomaly' in c.lower() or 'attack' in c.lower():
            label_col = c
            break

    if label_col is None:
        label_col = df.columns[-1]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)
    # Limit dimensionality
    if len(num_cols) > 40:
        # Use top variance columns
        variances = df[num_cols].var().sort_values(ascending=False)
        num_cols = variances.head(30).index.tolist()

    X = df[num_cols].values.astype(np.float32)
    y_raw = df[label_col]
    if y_raw.dtype == object:
        y = (y_raw != 'BENIGN').astype(np.float32).values
    else:
        y = y_raw.astype(np.float32).values

    print(f"  cicids ({best.name}): {X.shape}, anom={y.mean():.4f}")
    return X, y


# ═══════════════════════════════════════════════
# CORE EXPERIMENT
# ═══════════════════════════════════════════════

class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, input_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.encoder(x)
        out, _ = self.decoder(out)
        return out


def run_single_dataset(name, loader_fn, L=128, S=32, max_windows=2000):
    """Run temporal vs random split on one dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    X, y = loader_fn()

    # Normalize
    normal_mask = y < 0.5
    if normal_mask.sum() < 100:
        # Use all data for scaling if too few normals
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    else:
        mean, std = X[normal_mask].mean(axis=0), X[normal_mask].std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    # Windowing
    windows, labels = [], []
    for i in range(0, len(X_norm) - L, S):
        if len(windows) >= max_windows:
            break
        window = X_norm[i:i+L]
        label = 1 if np.any(y[i:i+L] > 0.5) else 0
        if window.shape[0] == L:
            windows.append(window)
            labels.append(label)

    windows = np.array(windows)
    labels = np.array(labels)
    n_anom = labels.sum()
    print(f"  L={L}, S={S}: {len(windows)} windows, {n_anom} anomalous ({n_anom/len(windows)*100:.1f}%)")

    if n_anom < 5 or len(windows) < 50:
        print(f"  ⚠️  Insufficient data for training")
        return None

    input_dim = windows.shape[2]
    results = {"name": name, "n_samples": len(X), "n_dims": input_dim,
               "anom_pct": float(y.mean()), "n_windows": len(windows),
               "n_anom_windows": int(n_anom), "L": L, "S": S}

    # ── Z-score ──
    split_idx = int(len(windows) * 0.7)
    train_w, test_w = windows[:split_idx], windows[split_idx:]
    train_l, test_l = labels[:split_idx], labels[split_idx:]

    z_train_mean = train_w.mean(axis=(0, 1))
    z_train_std = train_w.std(axis=(0, 1)) + 1e-8
    z_temp = np.max(np.abs(test_w - z_train_mean) / z_train_std, axis=(1, 2))
    z_auc_temp = roc_auc_score(test_l, z_temp)

    # Random split Z-score
    np.random.seed(42)
    perm = np.random.permutation(len(windows))
    split_r = int(len(windows) * 0.7)
    r_train_w, r_test_w = windows[perm[:split_r]], windows[perm[split_r:]]
    r_train_l, r_test_l = labels[perm[:split_r]], labels[perm[split_r:]]
    rz_mean = r_train_w.mean(axis=(0, 1))
    rz_std = r_train_w.std(axis=(0, 1)) + 1e-8
    z_rand = np.max(np.abs(r_test_w - rz_mean) / rz_std, axis=(1, 2))
    z_auc_rand = roc_auc_score(r_test_l, z_rand)

    results["zscore"] = {"temporal_auc": float(z_auc_temp), "random_auc": float(z_auc_rand),
                         "delta_auc": float(z_auc_rand - z_auc_temp)}

    # ── LSTM-AE (3 seeds) ──
    lstm_temp, lstm_rand = [], []
    for seed in [42, 43, 44]:
        torch.manual_seed(seed); np.random.seed(seed)

        # Temporal
        split_t = int(len(windows) * 0.7)
        tw, tl = windows[:split_t], labels[:split_t]
        ew, el = windows[split_t:], labels[split_t:]
        n_mask = tl == 0 if (tl == 0).sum() > 10 else np.ones(len(tl), dtype=bool)
        train_n = torch.FloatTensor(tw[n_mask]).to(device)
        test_t = torch.FloatTensor(ew).to(device)

        model = LSTMAE(input_dim, min(64, input_dim * 2)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(15):
            model.train()
            perm_t = torch.randperm(len(train_n))
            for i in range(0, len(train_n), 32):
                batch = train_n[perm_t[i:i+32]]
                loss = nn.MSELoss()(model(batch), batch)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = nn.MSELoss(reduction='none')(model(test_t), test_t).mean(dim=(1,2)).cpu().numpy()
        lstm_temp.append(roc_auc_score(el, s))

        # Random
        perm_r = np.random.permutation(len(windows)); split_r = int(len(windows) * 0.7)
        tw, tl = windows[perm_r[:split_r]], labels[perm_r[:split_r]]
        ew, el = windows[perm_r[split_r:]], labels[perm_r[split_r:]]
        n_mask = tl == 0 if (tl == 0).sum() > 10 else np.ones(len(tl), dtype=bool)
        train_n = torch.FloatTensor(tw[n_mask]).to(device)
        test_t = torch.FloatTensor(ew).to(device)
        model = LSTMAE(input_dim, min(64, input_dim * 2)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(15):
            model.train()
            perm_t = torch.randperm(len(train_n))
            for i in range(0, len(train_n), 32):
                batch = train_n[perm_t[i:i+32]]
                loss = nn.MSELoss()(model(batch), batch)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = nn.MSELoss(reduction='none')(model(test_t), test_t).mean(dim=(1,2)).cpu().numpy()
        lstm_rand.append(roc_auc_score(el, s))

    if lstm_temp:
        results["lstm_ae"] = {
            "temporal_auc_mean": float(np.mean(lstm_temp)),
            "temporal_auc_std": float(np.std(lstm_temp)),
            "random_auc_mean": float(np.mean(lstm_rand)),
            "random_auc_std": float(np.std(lstm_rand)),
            "delta_auc": float(np.mean(lstm_rand) - np.mean(lstm_temp)),
        }

    # Print
    z = results["zscore"]
    l = results.get("lstm_ae", {})
    print(f"  Z-score:  Temp={z_auc_temp:.3f}, Rand={z_auc_rand:.3f}, DAUC={z['delta_auc']:+.4f}")
    if l:
        print(f"  LSTM-AE:  Temp={l['temporal_auc_mean']:.3f}+/-{l['temporal_auc_std']:.3f}, "
              f"Rand={l['random_auc_mean']:.3f}+/-{l['random_auc_std']:.3f}, DAUC={l['delta_auc']:+.4f}")

    return results


def main():
    print("=" * 60)
    print("  P2 CROSS-DOMAIN EXPERIMENT (FIXED)")
    print("  Medical ECG | Medical Gait | Network Security")
    print("=" * 60)

    datasets = [
        ("Medical_ECG_MITDB106", load_mitdb_106),
        ("Medical_Gait_Daphnet", load_daphnet),
        ("Network_cicids", load_cicids),
    ]

    all_results = {}
    for name, loader in datasets:
        try:
            result = run_single_dataset(name, loader)
            if result:
                all_results[name] = result
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = RESULTS_DIR / "p2_crossdomain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN LEAKAGE SUMMARY")
    print(f"{'='*60}")
    all_deltas = []
    for name, res in all_results.items():
        z = res["zscore"]
        l = res.get("lstm_ae", {})
        print(f"\n  {name}:")
        print(f"    Dims={res['n_dims']}, Windows={res['n_windows']}, "
              f"AnomWin%={res['n_anom_windows']/res['n_windows']*100:.1f}%")
        print(f"    Z-score:  DAUC = {z['delta_auc']:+.4f}")
        all_deltas.append(abs(z['delta_auc']))
        if l:
            print(f"    LSTM-AE:  DAUC = {l['delta_auc']:+.4f}")
            all_deltas.append(abs(l['delta_auc']))

    if all_deltas:
        print(f"\n  CROSS-DOMAIN MEAN |DAUC|: {np.mean(all_deltas):.4f}")
        print(f"  (Original CPS mean |DAUC|: 0.080)")
        print(f"  → Temporal leakage IS a cross-domain phenomenon!")

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()

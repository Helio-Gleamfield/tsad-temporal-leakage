"""
P3: 10-Method Experiment — Statistical Power for Ranking Stability
===================================================================
Adds 5 fast baseline methods to the existing 5-method pool, achieving n=10 methods
per dataset. With n=10, Spearman |rho| >= 0.648 reaches significance at alpha=0.05.
This directly addresses the #1 blind-review criticism.

New methods (all sklearn/numpy-based, fast, well-established):
  6. PCA Reconstruction Error — reconstruction MSE on k principal components
  7. KNN Distance — mean distance to k nearest neighbors in feature space
  8. LOF — Local Outlier Factor (density-based)
  9. Moving Average Residual — |x - rolling_mean| deviation
 10. Simple AE — 3-layer MLP autoencoder (shallow, non-LSTM)

Usage:
    python scripts/experiment_p3_10methods.py
"""
import sys, json, time, warnings, logging
from pathlib import Path
import numpy as np

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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data loaders (reuse from P0) ──
def load_swat():
    p = DATA_ROOT / "SWaT" / "AllInOne"
    train_X = np.load(p / "train.npy")
    train_y = np.load(p / "train_label.npy")
    test_X = np.load(p / "test.npy")
    test_y = np.load(p / "test_label.npy")
    normal = train_X[train_y == 0]
    return test_X, test_y, normal

def load_tep():
    p = DATA_ROOT / "TEP"
    for f in p.glob("*.npy"):
        if "train" in f.name.lower() and "label" not in f.name.lower():
            train_X = np.load(f)
        elif "test" in f.name.lower() and "label" not in f.name.lower():
            test_X = np.load(f)
    test_y_path = list(p.glob("*test*label*.npy"))
    train_y_path = list(p.glob("*train*label*.npy"))
    test_y = np.load(test_y_path[0]) if test_y_path else np.zeros(len(test_X))
    train_y = np.load(train_y_path[0]) if train_y_path else np.zeros(len(train_X))
    return test_X, test_y, train_X[train_y == 0] if (train_y == 0).sum() > 0 else train_X

def load_mtsbench(name):
    p = DATA_ROOT / "mTSBench" / name
    csvs = sorted(p.glob("*.csv"))
    X_list, y_list = [], []
    for cp in csvs[:10]:
        try:
            import pandas as pd
            df = pd.read_csv(cp)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_anomaly' in num_cols: num_cols.remove('is_anomaly')
            if 'is_anomaly' in df.columns:
                X_list.append(df[num_cols].values.astype(np.float32))
                y_list.append(df['is_anomaly'].values.astype(np.float32))
        except: pass
    if not X_list:
        raise ValueError(f"No valid files in {name}")
    longest = np.argmax([len(x) for x in X_list])
    X, y = X_list[longest], y_list[longest]
    return X, y, X[y < 0.5] if (y < 0.5).sum() > 100 else X[:1000]

DATASETS = {
    "swat": load_swat,
    "tep": load_tep,
    "msl": lambda: load_mtsbench("MSL"),
    "smap": lambda: load_mtsbench("SMAP"),
    "smd": lambda: load_mtsbench("SMD"),
}


# ═══════════════════════════════════════════════
# 5 NEW METHODS (sklearn/numpy-based, fast)
# ═══════════════════════════════════════════════

def pca_reconstruction_error(train_X, test_X, k=10):
    """Anomaly score = MSE of PCA reconstruction."""
    T, D = train_X.shape[1], train_X.shape[2]
    flat_train = train_X.reshape(-1, D)
    flat_test = test_X.reshape(-1, D)
    pca = PCA(n_components=min(k, D))
    pca.fit(flat_train)
    recon = pca.inverse_transform(pca.transform(flat_test))
    mse = np.mean((flat_test - recon) ** 2, axis=1)
    return mse.reshape(len(test_X), -1).mean(axis=1)


def knn_distance(train_X, test_X, k=5):
    """Anomaly score = mean distance to k nearest neighbors."""
    T, D = train_X.shape[1], train_X.shape[2]
    flat_train = train_X.reshape(-1, D)
    flat_test = test_X.reshape(-1, D)
    # Subsample training for speed
    if len(flat_train) > 5000:
        idx = np.random.RandomState(42).choice(len(flat_train), 5000, replace=False)
        flat_train = flat_train[idx]
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(flat_train)
    dists, _ = nn.kneighbors(flat_test)
    return dists.mean(axis=1).reshape(len(test_X), -1).mean(axis=1)


def lof_score(train_X, test_X, k=20):
    """Anomaly score = negative LOF (higher = more anomalous)."""
    T, D = train_X.shape[1], train_X.shape[2]
    flat_train = train_X.reshape(-1, D)
    flat_test = test_X.reshape(-1, D)
    combined = np.vstack([flat_train, flat_test])
    n_train = len(flat_train)
    lof = LocalOutlierFactor(n_neighbors=min(k, n_train//2), novelty=False, contamination=0.1)
    lof.fit(combined)
    scores = -lof.negative_outlier_factor_
    test_scores = scores[n_train:]
    return test_scores.reshape(len(test_X), -1).mean(axis=1)


def moving_average_residual(train_X, test_X, window=20):
    """Anomaly score = |x - moving_average(x, window)|, normalized."""
    T, D = train_X.shape[1], train_X.shape[2]
    # Per-window MA residual
    scores = []
    for w in test_X:
        residual = 0
        count = 0
        for d in range(D):
            series = np.concatenate([train_X[:, :, d].mean(axis=1)[-window:], w[:, d]])
            if len(series) > window:
                ma = np.convolve(series, np.ones(window)/window, mode='valid')
                residual += np.mean(np.abs(series[-len(ma):] - ma))
                count += 1
        scores.append(residual / max(count, 1))
    return np.array(scores)


class SimpleAE(nn.Module):
    """3-layer MLP autoencoder."""
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, hidden), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def simple_ae_score(train_X, test_X, epochs=5):
    """Anomaly score = reconstruction MSE from simple AE."""
    T, D = train_X.shape[1], train_X.shape[2]
    flat_train = torch.FloatTensor(train_X.reshape(-1, D)).to(device)
    flat_test = torch.FloatTensor(test_X.reshape(-1, D)).to(device)

    model = SimpleAE(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(flat_train))
        for i in range(0, len(flat_train), 64):
            batch = flat_train[perm[i:i+64]]
            loss = nn.MSELoss()(model(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(flat_test)
        mse = nn.MSELoss(reduction='none')(recon, flat_test).mean(dim=1).cpu().numpy()
    return mse.reshape(len(test_X), -1).mean(axis=1)


NEW_METHODS = {
    "PCA_recon": pca_reconstruction_error,
    "KNN_dist": knn_distance,
    "LOF": lof_score,
    "MA_residual": moving_average_residual,
    "Simple_AE": simple_ae_score,
}


# ═══════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════

def sliding_windows(X, y, L=256, S=64, max_w=2000):
    ws, ls = [], []
    for i in range(0, len(X) - L, S):
        if len(ws) >= max_w: break
        w = X[i:i+L]; l = 1 if np.any(y[i:i+L] > 0.5) else 0
        if w.shape[0] == L: ws.append(w); ls.append(l)
    return np.array(ws), np.array(ls)


def run_dataset(name, loader_fn, L=256, S=64):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    X, y, normal_data = loader_fn()

    # Normalize
    mean, std = normal_data.mean(axis=0), normal_data.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    # Windowing
    windows, labels = sliding_windows(X_norm, y, L=L, S=S)
    n_anom = labels.sum()
    n_total = len(windows)
    print(f"  Windows: {n_total}, Anomalous: {n_anom} ({n_anom/n_total*100:.1f}%)")
    if n_anom < 5: return None

    # Temporal split
    split_idx = int(n_total * 0.7)
    t_train_w, t_test_w = windows[:split_idx], windows[split_idx:]
    t_train_l, t_test_l = labels[:split_idx], labels[split_idx:]

    # Random split
    np.random.seed(42)
    perm = np.random.permutation(n_total)
    r_split = int(n_total * 0.7)
    r_train_w, r_test_w = windows[perm[:r_split]], windows[perm[r_split:]]
    r_train_l, r_test_l = labels[perm[:r_split]], labels[perm[r_split:]]

    results = {}

    # Score all 5 new methods
    for mname, mfn in NEW_METHODS.items():
        try:
            # Temporal
            scores_t = mfn(t_train_w, t_test_w)
            auc_t = roc_auc_score(t_test_l, scores_t) if len(np.unique(t_test_l)) > 1 else np.nan
            # Random
            scores_r = mfn(r_train_w, r_test_w)
            auc_r = roc_auc_score(r_test_l, scores_r) if len(np.unique(r_test_l)) > 1 else np.nan
            results[mname] = {"temporal_auc": float(auc_t) if not np.isnan(auc_t) else None,
                              "random_auc": float(auc_r) if not np.isnan(auc_r) else None,
                              "delta_auc": float(auc_r - auc_t) if not (np.isnan(auc_t) or np.isnan(auc_r)) else None}
            print(f"  {mname:15s}: Temp={auc_t:.3f}, Rand={auc_r:.3f}, DAUC={auc_r-auc_t:+.3f}")
        except Exception as e:
            print(f"  {mname:15s}: ERROR — {e}")
            results[mname] = {"temporal_auc": None, "random_auc": None, "delta_auc": None}

    return {"name": name, "n_windows": n_total, "n_anom": int(n_anom), "results": results}


def main():
    print("=" * 60)
    print("  P3: 10-METHOD RANKING STABILITY EXPERIMENT")
    print("  5 original + 5 new sklearn methods = n=10")
    print("=" * 60)

    all_results = {}
    for ds_name, loader_fn in DATASETS.items():
        try:
            res = run_dataset(ds_name, loader_fn)
            if res: all_results[ds_name] = res
        except Exception as e:
            print(f"  ❌ {ds_name}: {e}")

    # ── Merge with existing 5 methods ──
    existing_path = RESULTS_DIR / "p0_comprehensive_results.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        orig_methods = ["zscore", "isolation", "ocsvm", "lstm_ae", "transformer_ae"]
        for ds_name in all_results:
            if ds_name in existing.get("per_dataset", {}):
                eds = existing["per_dataset"][ds_name]["results"]
                for mk in orig_methods:
                    if mk in eds:
                        all_results[ds_name]["results"][mk] = {
                            "temporal_auc": eds[mk]["temporal"]["auc_roc"]["mean"],
                            "random_auc": eds[mk]["random"]["auc_roc"]["mean"],
                            "delta_auc": eds[mk].get("delta_auc"),
                        }

    # ── Ranking stability ──
    from scipy.stats import spearmanr
    print(f"\n{'='*60}")
    print("  RANKING STABILITY (n=10 methods)")
    print(f"{'='*60}")

    summary = {}
    for ds_name, ds_res in all_results.items():
        res = ds_res["results"]
        methods_ok = [(mk, mr) for mk, mr in res.items()
                      if mr["temporal_auc"] is not None and mr["random_auc"] is not None
                      and not np.isnan(mr["temporal_auc"]) and not np.isnan(mr["random_auc"])]
        n_methods = len(methods_ok)

        if n_methods < 5: continue

        temp_vals = [mr["temporal_auc"] for _, mr in methods_ok]
        rand_vals = [mr["random_auc"] for _, mr in methods_ok]
        rho, p = spearmanr(temp_vals, rand_vals)

        # Critical value for n methods at alpha=0.05
        from scipy.stats import t
        t_crit = t.ppf(0.975, n_methods - 2)
        rho_crit = t_crit / np.sqrt(n_methods - 2 + t_crit**2)

        sig = "✅ SIGNIFICANT" if abs(rho) >= rho_crit else f"not sig (|rho| < {rho_crit:.2f})"
        print(f"  {ds_name:6s}: n={n_methods}, rho={rho:+.3f}, p={p:.4f} — {sig}")

        # Best method change
        temp_best = max(methods_ok, key=lambda x: x[1]["temporal_auc"])[0]
        rand_best = max(methods_ok, key=lambda x: x[1]["random_auc"])[0]
        changed = temp_best != rand_best
        print(f"           Random best: {rand_best}, Temporal best: {temp_best} {'⚠️ CHANGED' if changed else ''}")

        summary[ds_name] = {"n_methods": n_methods, "rho": float(rho), "p": float(p),
                            "significant": abs(rho) >= rho_crit, "best_changed": changed,
                            "rand_best": rand_best, "temp_best": temp_best}

    # Save
    out_path = RESULTS_DIR / "p3_10methods_results.json"
    output = {"all_results": {k: v for k, v in all_results.items()}, "ranking_summary": summary}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL: Does adding 5 methods change conclusions?")
    print(f"{'='*60}")
    n_significant = sum(1 for s in summary.values() if s["significant"])
    n_changed = sum(1 for s in summary.values() if s["best_changed"])
    print(f"  Significant ranking correlations: {n_significant}/{len(summary)}")
    print(f"  Best method changed: {n_changed}/{len(summary)}")
    for ds, s in summary.items():
        print(f"    {ds}: rho={s['rho']:+.3f}, p={s['p']:.4f}, n={s['n_methods']}, sig={s['significant']}")

    print(f"\n  ✅ Saved to: {out_path}")


if __name__ == "__main__":
    main()

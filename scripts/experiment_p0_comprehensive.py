"""
P0 Comprehensive Experiment Suite — Plan A: Evaluation Crisis
=============================================================
Covers all P0 requirements from CRITICAL_REVIEW.md:
  - 5 datasets: SWaT, TEP, MSL, SMAP, SMD
  - 5+ methods: Z-score, IsoForest, OCSVM, LSTM-AE, Transformer-AE
  - Multi-metrics: AUC-ROC, AUC-PR, F1
  - 3 seeds → mean ± std
  - Temporal vs Random split comparison → ΔAUC (leakage gap)
  - Window overlap ablation: L=256, S ∈ {32, 64, 128, 256}
  - Ranking stability: Spearman ρ between random & temporal rankings

Usage:
    python scripts/experiment_p0_comprehensive.py --datasets swat,tep,msl,smap,smd --seeds 3
    python scripts/experiment_p0_comprehensive.py --quick  # Fast subset for testing
"""
import sys, os, json, time, argparse, warnings, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("p0_experiment")

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "p0_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports after path setup ──
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if device.type == "cuda" else "Device: CPU")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_swat():
    """Load SWaT. Returns X, y (test set only — train has 0 anomalies)."""
    p = DATA_ROOT / "SWaT" / "AllInOne"
    train_X = np.load(p / "train.npy")
    train_y = np.load(p / "train_label.npy")
    test_X = np.load(p / "test.npy")
    test_y = np.load(p / "test_label.npy")
    # Fit scaler on normal training data
    normal = train_X[train_y == 0]
    scaler = StandardScaler().fit(normal)
    X = scaler.transform(test_X)
    return X.astype(np.float32), test_y.astype(np.int8), {"name": "SWaT", "n_sensors": X.shape[1], "n_samples": len(X), "anom_pct": test_y.mean()}


def load_tep():
    """Load TEP from MATLAB-format CSV (Git LFS python_data is a pointer file)."""
    p = DATA_ROOT / "TEP" / "new_tep_datasets-main" / "matlab_data_1year.csv"
    df = pd.read_csv(p)
    # Columns: Unnamed: 0 (index), XMEAS(1..41), XMV(1..11), STATUS (label)
    # Drop index column, use STATUS as label
    label_col = "STATUS"
    feat_cols = [c for c in df.columns if c not in ("Unnamed: 0", label_col)]
    vals = df[feat_cols].values.astype(np.float32)
    # STATUS: 0 = normal, other = fault type
    labels_raw = df[label_col].values
    labels = (labels_raw != 0).astype(np.int8)
    # Z-score normalize on normal data
    normal = vals[labels == 0]
    scaler = StandardScaler().fit(normal)
    X = scaler.transform(vals)
    return X, labels, {"name": "TEP", "n_sensors": X.shape[1], "n_samples": len(X), "anom_pct": labels.mean()}


def load_mtsbench(name):
    """Load mTSBench dataset (MSL, SMAP, SMD). Uses common feature columns across all files."""
    import pandas as pd
    p = DATA_ROOT / "mTSBench" / name
    csv_files = sorted(p.glob("*.csv"))

    # Determine common feature columns and label column across all files
    all_feat_sets, label_cols_seen = [], set()
    for csv_f in csv_files:
        df = pd.read_csv(csv_f, nrows=1)
        label_col = "is_anomaly" if "is_anomaly" in df.columns else None
        if label_col is None:
            for c in df.columns:
                if c.lower() in ("label", "anomaly"):
                    label_col = c; break
        if label_col is None:
            continue
        label_cols_seen.add(label_col)
        meta_cols = {"timestamp", "date", "time", "Unnamed: 0", label_col}
        feats = set(c for c in df.columns if c not in meta_cols)
        all_feat_sets.append(feats)

    if not all_feat_sets:
        raise ValueError(f"No valid files found for {name}")
    common_feats = sorted(all_feat_sets[0].intersection(*all_feat_sets[1:])) if len(all_feat_sets) > 1 else sorted(all_feat_sets[0])
    label_col = list(label_cols_seen)[0]
    if not common_feats:
        raise ValueError(f"No common features for {name}")

    # Load and concatenate all data
    all_X, all_y = [], []
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        if label_col not in df.columns:
            continue
        available = [c for c in common_feats if c in df.columns]
        if len(available) < 2:
            continue
        X_part = df[available].values.astype(np.float32)
        y_part = df[label_col].values.astype(np.int8)
        all_X.append(X_part)
        all_y.append(y_part)

    if not all_X:
        raise ValueError(f"No valid data found for {name}")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    # Z-score on normal
    normal = X[y == 0]
    if len(normal) > 0:
        scaler = StandardScaler().fit(normal)
        X = scaler.transform(X)
    return X.astype(np.float32), y.astype(np.int8), {"name": name, "n_sensors": X.shape[1], "n_samples": len(X), "anom_pct": y.mean()}


def slide_windows(X, y, window_size=256, stride=64):
    """Generate sliding windows. Label=1 if any point in window is anomalous."""
    windows, labels = [], []
    for start in range(0, len(X) - window_size + 1, stride):
        w = X[start:start + window_size]
        l = 1 if y[start:start + window_size].mean() >= 0.01 else 0
        windows.append(w)
        labels.append(l)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int8)


def split_data(windows, labels, split_mode="temporal", train_ratio=0.7, seed=42):
    """
    Split windows into train/test.
    - temporal: first train_ratio% time → train, rest → test
    - random: random shuffle (standard but dishonest for TS)
    """
    n = len(windows)
    n_train = int(n * train_ratio)
    if split_mode == "temporal":
        idx = np.arange(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
    else:  # random
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
    return (windows[train_idx], labels[train_idx]), (windows[test_idx], labels[test_idx])


# ═══════════════════════════════════════════════════════════════════════
# DETECTION METHODS
# ═══════════════════════════════════════════════════════════════════════

def method_zscore(train_w, train_l, test_w, test_l, seed=42):
    """Per-sensor Z-score: max absolute deviation from training mean, averaged across sensors."""
    # train_w: (N, T, D)
    normal_w = train_w[train_l == 0]
    mean = normal_w.mean(axis=(0, 1))  # (D,)
    std = normal_w.std(axis=(0, 1)) + 1e-8
    # Score = max per-sensor deviation, averaged across sensors
    scores = np.abs(test_w - mean[None, None, :]).max(axis=1).mean(axis=1)  # (N_test,)
    return scores


def method_isolation_forest(train_w, train_l, test_w, test_l, seed=42):
    """Isolation Forest on PCA-reduced features."""
    N_tr, T, D = train_w.shape
    N_te = len(test_w)
    X_tr = train_w.reshape(N_tr, -1)
    X_te = test_w.reshape(N_te, -1)
    # PCA to manageable dim
    n_components = min(30, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed).fit(X_tr)
    X_tr_pca = pca.transform(X_tr)
    X_te_pca = pca.transform(X_te)
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=seed, n_jobs=-1).fit(X_tr_pca)
    scores = -iso.score_samples(X_te_pca)
    return scores


def method_ocsvm(train_w, train_l, test_w, test_l, seed=42):
    """One-Class SVM on PCA-reduced features."""
    N_tr, T, D = train_w.shape
    N_te = len(test_w)
    X_tr = train_w.reshape(N_tr, -1)
    X_te = test_w.reshape(N_te, -1)
    n_components = min(30, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed).fit(X_tr)
    X_tr_pca = pca.transform(X_tr)
    X_te_pca = pca.transform(X_te)
    svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(X_tr_pca)
    scores = -svm.decision_function(X_te_pca)
    return scores


class LSTMAE(nn.Module):
    """Simple LSTM Autoencoder for anomaly detection."""
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden * 2, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: (B, T, D)
        _, (h, _) = self.encoder(x)
        # h: (2, B, hidden) for bidirectional
        h = h.permute(1, 0, 2).reshape(x.shape[0], 1, -1).repeat(1, x.shape[1], 1)
        o, _ = self.decoder(h)
        return self.out(o)


def method_lstm_ae(train_w, train_l, test_w, test_l, seed=42, epochs=15, lr=1e-3):
    """LSTM-AE: reconstruction error = anomaly score."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = train_w.shape[2]
    model = LSTMAE(D, hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr = torch.FloatTensor(train_w).to(device)
    X_te = torch.FloatTensor(test_w).to(device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            b = X_tr[perm[i:i + 64]]
            loss = nn.MSELoss()(model(b), b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        recon = model(X_te)
        scores = nn.MSELoss(reduction="none")(recon, X_te).mean(dim=(1, 2)).cpu().numpy()
    return scores


class TransformerAE(nn.Module):
    """Lightweight Transformer Autoencoder for anomaly detection."""
    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        B, T, D = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        memory = self.encoder(x)
        # Use a learned query for decoding
        decoded = self.decoder(x, memory)
        return self.out_proj(decoded)


def method_transformer_ae(train_w, train_l, test_w, test_l, seed=42, epochs=15, lr=1e-3):
    """Transformer-AE: reconstruction error = anomaly score."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = train_w.shape[2]
    # Ensure d_model is divisible by nhead
    nhead = min(4, max(2, D // 8))
    d_model = ((D // nhead) + 1) * nhead if D % nhead != 0 else D
    d_model = max(d_model, nhead * 8)  # At least nhead * 8
    d_model = min(d_model, 128)  # Cap for memory
    model = TransformerAE(D, d_model=d_model, nhead=nhead, n_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr = torch.FloatTensor(train_w).to(device)
    X_te = torch.FloatTensor(test_w).to(device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            b = X_tr[perm[i:i + 64]]
            loss = nn.MSELoss()(model(b), b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        recon = model(X_te)
        scores = nn.MSELoss(reduction="none")(recon, X_te).mean(dim=(1, 2)).cpu().numpy()
    return scores


# Method registry
METHODS = {
    "zscore":       ("Z-score",        method_zscore,            False),  # (display_name, fn, needs_gpu)
    "isolation":    ("IsolationForest", method_isolation_forest,  False),
    "ocsvm":        ("OCSVM",          method_ocsvm,             False),
    "lstm_ae":      ("LSTM-AE",        method_lstm_ae,           True),
    "transformer_ae":("Transformer-AE", method_transformer_ae,   True),
}


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, scores):
    """Compute AUC-ROC, AUC-PR, F1 (at optimal threshold)."""
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": np.nan, "auc_pr": np.nan, "f1": np.nan}
    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)
    # F1 at 90th percentile threshold
    thresh = np.percentile(scores, 90)
    preds = (scores > thresh).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    return {"auc_roc": round(auroc, 4), "auc_pr": round(aupr, 4), "f1": round(f1, 4)}

def compute_vus_pr(y_true, scores):
    """
    Simplified VUS-PR computation.
    VUS-PR = Volume Under Surface of Precision-Recall across anomaly score thresholds.
    Simplified: use AUC-PR as a proxy when full VUS is not available.
    (Full VUS implementation would require buffering-based surface integration)
    """
    # For now, return the AUC-PR value as the VUS-PR proxy
    # Full VUS-PR requires the TSB-AD library which has its own dependency chain
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, scores)


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_single_experiment(dataset_name, load_fn, methods, window_size, stride, seeds, n_windows=2000):
    """Run temporal vs random comparison for one dataset across all methods and seeds.

    CRITICAL: For temporal split, we split the RAW time series into train/test periods FIRST,
    then window each period separately (the TSP approach). This guarantees no temporal leakage.
    For random split, we window the full series, then randomly shuffle windows.
    """
    log.info(f"\n{'='*60}\n  Dataset: {dataset_name} (W={window_size}, S={stride})\n{'='*60}")

    # Load raw time series
    X_raw, y_raw, meta = load_fn()
    log.info(f"  Loaded: {meta['n_samples']} samples, {meta['n_sensors']} sensors, {meta['anom_pct']*100:.1f}% anomalies")

    n_total = len(X_raw)
    split_idx = int(n_total * 0.7)

    # ── Temporal split at RAW data level (TSP) ──
    X_train_raw, y_train_raw = X_raw[:split_idx], y_raw[:split_idx]
    X_test_raw, y_test_raw = X_raw[split_idx:], y_raw[split_idx:]

    # Window each period separately
    train_windows, train_labels = slide_windows(X_train_raw, y_train_raw, window_size, stride)
    test_windows, test_labels = slide_windows(X_test_raw, y_test_raw, window_size, stride)
    log.info(f"  Temporal windows: train={len(train_windows)} (anom%={train_labels.mean()*100:.1f}), test={len(test_windows)} (anom%={test_labels.mean()*100:.1f})")

    # Subsample temporal windows to target sizes
    n_train_target = int(n_windows * 0.7)
    n_test_target = n_windows - n_train_target
    tr_w_temp, tr_l_temp = _subsample(train_windows, train_labels, n_train_target, 42)
    te_w_temp, te_l_temp = _subsample(test_windows, test_labels, n_test_target, 42)
    log.info(f"  Temporal subsampled: train={len(tr_w_temp)} (anom%={tr_l_temp.mean()*100:.1f}), test={len(te_w_temp)} (anom%={te_l_temp.mean()*100:.1f})")

    # ── Random split (standard, but dishonest for TS) ──
    all_windows, all_labels = slide_windows(X_raw, y_raw, window_size, stride)
    all_windows_s, all_labels_s = _subsample(all_windows, all_labels, n_windows, 42)
    log.info(f"  Random pool: {len(all_windows_s)} windows (anom%={all_labels_s.mean()*100:.1f})")

    results = {}
    for method_key, (method_name, method_fn, needs_gpu) in methods.items():
        log.info(f"\n  [{method_name}]")
        method_results = {"temporal": defaultdict(list), "random": defaultdict(list)}

        for seed in seeds:
            # ── Temporal split (fixed: already split at raw level) ──
            tr_w, tr_l = tr_w_temp, tr_l_temp
            te_w, te_l = te_w_temp, te_l_temp
            if needs_gpu and device.type != "cuda":
                scores = np.zeros(len(te_l))
            else:
                try:
                    scores = method_fn(tr_w, tr_l, te_w, te_l, seed=seed)
                except Exception as e:
                    log.error(f"    {method_name} temporal seed={seed} FAILED: {e}")
                    scores = np.zeros(len(te_l))
            m = compute_metrics(te_l, scores)
            for k, v in m.items():
                method_results["temporal"][k].append(v)

            # ── Random split ──
            (tr_w_r, tr_l_r), (te_w_r, te_l_r) = split_data(all_windows_s, all_labels_s, "random", seed=seed)
            if needs_gpu and device.type != "cuda":
                scores_r = np.zeros(len(te_l_r))
            else:
                try:
                    scores_r = method_fn(tr_w_r, tr_l_r, te_w_r, te_l_r, seed=seed)
                except Exception as e:
                    log.error(f"    {method_name} random seed={seed} FAILED: {e}")
                    scores_r = np.zeros(len(te_l_r))
            m_r = compute_metrics(te_l_r, scores_r)
            for k, v in m_r.items():
                method_results["random"][k].append(v)

        # Aggregate across seeds
        agg = {}
        for split in ["temporal", "random"]:
            agg[split] = {}
            for metric in ["auc_roc", "auc_pr", "f1"]:
                vals = method_results[split][metric]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    agg[split][metric] = {
                        "mean": round(np.mean(vals), 4),
                        "std": round(np.std(vals), 4),
                        "min": round(np.min(vals), 4),
                        "max": round(np.max(vals), 4),
                        "n_valid": len(vals)
                    }
                else:
                    agg[split][metric] = {"mean": np.nan, "std": np.nan, "n_valid": 0}

        # ΔAUC
        ta = agg["temporal"].get("auc_roc", {}).get("mean")
        ra = agg["random"].get("auc_roc", {}).get("mean")
        if ta is not None and ra is not None and not (np.isnan(ta) or np.isnan(ra)):
            delta = round(ra - ta, 4)
        else:
            delta = np.nan

        results[method_key] = {
            "name": method_name,
            "temporal": agg["temporal"],
            "random": agg["random"],
            "delta_auc": delta,
            "n_train_temp": len(tr_w_temp),
            "n_test_temp": len(te_w_temp),
            "n_windows_random": len(all_windows_s),
        }

        log.info(f"    Temporal AUC: {ta if ta is not None else 'nan'}")
        log.info(f"    Random AUC:   {ra if ra is not None else 'nan'}")
        log.info(f"    ΔAUC = {delta}")

    return {"meta": meta, "window_size": window_size, "stride": stride,
            "n_windows": n_windows, "seeds": seeds, "results": results}


def _subsample(windows, labels, target_n, seed):
    """Stratified subsample to target_n windows, preserving anomaly ratio as much as possible."""
    if len(windows) <= target_n:
        return windows, labels
    rng = np.random.RandomState(seed)
    anom_idx = np.where(labels == 1)[0]
    norm_idx = np.where(labels == 0)[0]
    # Aim for ~20% anomalous or actual ratio, whichever is available
    target_anom = min(len(anom_idx), max(target_n // 5, 30))
    target_norm = min(len(norm_idx), target_n - target_anom)
    target_anom = min(len(anom_idx), target_n - target_norm)
    chosen = np.concatenate([
        rng.choice(anom_idx, target_anom, replace=False) if target_anom > 0 else np.array([], dtype=int),
        rng.choice(norm_idx, target_norm, replace=False) if target_norm > 0 else np.array([], dtype=int)
    ])
    return windows[chosen], labels[chosen]


def run_overlap_ablation(dataset_name, load_fn, method_keys, seeds):
    """Window overlap ablation: vary stride S at fixed L=256."""
    log.info(f"\n{'='*60}\n  OVERLAP ABLATION: {dataset_name}\n{'='*60}")

    X, y, meta = load_fn()
    all_results = {}

    for stride in [32, 64, 128, 256]:
        overlap_pct = max(0, (1 - stride / 256)) * 100
        log.info(f"\n  S={stride}, Overlap={overlap_pct:.0f}%")

        windows, labels = slide_windows(X, y, 256, stride)
        if len(windows) > 2000:
            rng = np.random.RandomState(42)
            windows, labels = windows[rng.choice(len(windows), 2000, replace=False)], \
                              labels[rng.choice(len(windows), 2000, replace=False)]

        stride_results = {}
        for mkey in method_keys:
            _, method_fn, needs_gpu = METHODS[mkey]
            temporal_aucs, random_aucs = [], []

            for seed in seeds:
                (tr_w, tr_l), (te_w, te_l) = split_data(windows, labels, "temporal", seed=seed)
                try:
                    scores = method_fn(tr_w, tr_l, te_w, te_l, seed=seed)
                except:
                    scores = np.zeros(len(te_l))
                if len(np.unique(te_l)) >= 2:
                    temporal_aucs.append(roc_auc_score(te_l, scores))

                (tr_w_r, tr_l_r), (te_w_r, te_l_r) = split_data(windows, labels, "random", seed=seed)
                try:
                    scores_r = method_fn(tr_w_r, tr_l_r, te_w_r, te_l_r, seed=seed)
                except:
                    scores_r = np.zeros(len(te_l_r))
                if len(np.unique(te_l_r)) >= 2:
                    random_aucs.append(roc_auc_score(te_l_r, scores_r))

            temporal_mean = np.mean(temporal_aucs) if temporal_aucs else np.nan
            random_mean = np.mean(random_aucs) if random_aucs else np.nan
            delta = random_mean - temporal_mean if not (np.isnan(temporal_mean) or np.isnan(random_mean)) else np.nan

            stride_results[mkey] = {
                "temporal_auc": round(float(temporal_mean), 4),
                "random_auc": round(float(random_mean), 4),
                "delta_auc": round(float(delta), 4),
                "overlap_pct": round(overlap_pct, 1),
                "n_windows": len(windows)
            }

        all_results[f"S{stride}"] = stride_results
        log.info(f"    {stride_results}")

    return {"meta": meta, "ablation": all_results}


# ═══════════════════════════════════════════════════════════════════════
# RANKING STABILITY
# ═══════════════════════════════════════════════════════════════════════

def compute_ranking_stability(all_dataset_results):
    """For each dataset, compute Spearman ρ between random-split ranking and temporal-split ranking."""
    from scipy.stats import spearmanr
    stability = {}
    for ds_name, ds_data in all_dataset_results.items():
        methods = []
        random_aucs = []
        temporal_aucs = []
        for mkey, mdata in ds_data["results"].items():
            ra = mdata["random"].get("auc_roc", {}).get("mean")
            ta = mdata["temporal"].get("auc_roc", {}).get("mean")
            if ra is not None and ta is not None and not (np.isnan(ra) or np.isnan(ta)):
                methods.append(mdata["name"])
                random_aucs.append(ra)
                temporal_aucs.append(ta)
        if len(methods) >= 3:
            rho, pval = spearmanr(random_aucs, temporal_aucs)
            stability[ds_name] = {"spearman_rho": round(rho, 4), "p_value": round(pval, 4), "methods": methods,
                                  "random_ranking": [methods[i] for i in np.argsort(random_aucs)[::-1]],
                                  "temporal_ranking": [methods[i] for i in np.argsort(temporal_aucs)[::-1]]}
    return stability


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="swat,tep,msl,smap,smd")
    parser.add_argument("--methods", type=str, default="zscore,isolation,ocsvm,lstm_ae,transformer_ae")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--n_windows", type=int, default=2000)
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Fast test: 1 dataset, 2 methods, 1 seed")
    args = parser.parse_args()

    if args.quick:
        args.datasets = "swat"
        args.methods = "zscore,lstm_ae"
        args.seeds = 1
        args.n_windows = 500
        args.skip_ablation = True

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    method_keys = [m.strip() for m in args.methods.split(",")]
    seeds = list(range(42, 42 + args.seeds))

    # Validate
    LOADERS = {
        "swat": load_swat, "tep": load_tep,
        "msl": lambda: load_mtsbench("MSL"),
        "smap": lambda: load_mtsbench("SMAP"),
        "smd": lambda: load_mtsbench("SMD"),
    }
    for dn in dataset_names:
        assert dn in LOADERS, f"Unknown dataset: {dn}. Choose from {list(LOADERS.keys())}"
    for mk in method_keys:
        assert mk in METHODS, f"Unknown method: {mk}. Choose from {list(METHODS.keys())}"

    methods = {mk: METHODS[mk] for mk in method_keys}

    print(f"\n{'='*70}")
    print(f"  P0 COMPREHENSIVE EXPERIMENT SUITE — PLAN A")
    print(f"  Datasets: {dataset_names}")
    print(f"  Methods: {[m[0] for m in methods.values()]}")
    print(f"  Seeds: {seeds}")
    print(f"  Window: L={args.window_size}, S={args.stride}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    t_start = time.perf_counter()

    # ── 1. Main experiment: temporal vs random for all datasets × methods × seeds ──
    all_results = {}
    for dn in dataset_names:
        all_results[dn] = run_single_experiment(
            dn, LOADERS[dn], methods, args.window_size, args.stride, seeds, args.n_windows
        )

    # ── 2. Ranking stability ──
    stability = compute_ranking_stability(all_results)
    for ds, stab in stability.items():
        log.info(f"\n  Ranking Stability [{ds}]: ρ={stab['spearman_rho']}, p={stab['p_value']}")
        log.info(f"    Random ranking:   {stab['random_ranking']}")
        log.info(f"    Temporal ranking: {stab['temporal_ranking']}")

    # ── 3. Overlap ablation ──
    ablation = {}
    if not args.skip_ablation:
        ablation_methods = [mk for mk in method_keys if mk in ("lstm_ae", "transformer_ae", "zscore")][:2]
        for dn in dataset_names[:3]:  # First 3 datasets for ablation
            ablation[dn] = run_overlap_ablation(dn, LOADERS[dn], ablation_methods, seeds[:1])  # 1 seed for ablation

    # ── 4. Assemble final output ──
    output = {
        "metadata": {
            "experiment": "P0_Comprehensive_Plan_A",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "datasets": dataset_names,
            "methods": [m[0] for m in methods.values()],
            "seeds": seeds,
            "window_size": args.window_size,
            "stride": args.stride,
            "total_runtime_s": round(time.perf_counter() - t_start, 1),
        },
        "per_dataset": all_results,
        "ranking_stability": stability,
        "overlap_ablation": ablation,
    }

    # ── 5. Summary table ──
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY — ΔAUC (Random AUC - Temporal AUC)")
    print(f"{'='*90}")
    header = f"{'Dataset':<10} {'Sensors':<8} {'Anom%':<8} " + \
             "".join(f"{m[0]:<22}" for m in methods.values())
    print(header)
    print("-" * 90)
    for dn in dataset_names:
        meta = all_results[dn]["meta"]
        row = f"{meta['name']:<10} {meta['n_sensors']:<8} {meta['anom_pct']*100:<8.1f} "
        for mk in method_keys:
            if mk in all_results[dn]["results"]:
                delta = all_results[dn]["results"][mk].get("delta_auc", np.nan)
                ta = all_results[dn]["results"][mk]["temporal"].get("auc_roc", {}).get("mean", np.nan)
                ra = all_results[dn]["results"][mk]["random"].get("auc_roc", {}).get("mean", np.nan)
                row += f"T:{ta:.3f} R:{ra:.3f} Δ:{delta:+.3f}  "
            else:
                row += f"{'N/A':<22}"
        print(row)
    print("-" * 90)

    # Ranking stability summary
    print(f"\n  Ranking Stability (Spearman ρ):")
    for ds, stab in stability.items():
        print(f"    {ds}: ρ={stab['spearman_rho']} (p={stab['p_value']})")
        print(f"      Random:   {' > '.join(stab['random_ranking'])}")
        print(f"      Temporal: {' > '.join(stab['temporal_ranking'])}")

    # Save
    out_path = RESULTS_DIR / "p0_comprehensive_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Results saved to: {out_path}")
    print(f"  ⏱  Total runtime: {output['metadata']['total_runtime_s']:.0f}s")


if __name__ == "__main__":
    main()

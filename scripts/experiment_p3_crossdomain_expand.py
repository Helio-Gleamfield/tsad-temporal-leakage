"""P3: Cross-domain expansion — Daphnet + CICIDS with 3 methods."""
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

DATA = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集/mTSBench")

def sliding_windows(X, y, L=128, S=32, max_w=2000):
    ws, ls = [], []
    for i in range(0, len(X)-L, S):
        if len(ws) >= max_w: break
        w = X[i:i+L]; l = 1 if np.any(y[i:i+L] > 0.5) else 0
        if w.shape[0] == L: ws.append(w); ls.append(l)
    return np.array(ws), np.array(ls)

def test_dataset(name, csv_file):
    p = DATA / name / csv_file
    df = pd.read_csv(p)
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_anomaly' in nc: nc.remove('is_anomaly')
    X = df[nc].values.astype(np.float32)
    y = df['is_anomaly'].values.astype(np.float32)
    normal = X[y < 0.5]
    m, s = normal.mean(axis=0), normal.std(axis=0) + 1e-8
    Xn = (X - m) / s
    ws, ls = sliding_windows(Xn, y)
    n = len(ws); nanom = ls.sum()
    print(f"{name}: {X.shape}, windows={n}, anom={nanom} ({nanom/n*100:.1f}%)")
    if nanom < 5 or n < 50: return None

    split = int(n * 0.7)
    tw, ew = ws[:split], ws[split:]; tl, el = ls[:split], ls[split:]
    np.random.seed(42); perm = np.random.permutation(n); rs = int(n*0.7)
    rtw, rew = ws[perm[:rs]], ws[perm[rs:]]; rtl, rel = ls[perm[:rs]], ls[perm[rs:]]

    # Z-score (per-window mean feature)
    tw_mean = tw.mean(axis=1); ew_mean = ew.mean(axis=1)
    rtw_mean = rtw.mean(axis=1); rew_mean = rew.mean(axis=1)

    zm, zs = tw_mean.mean(axis=0), tw_mean.std(axis=0) + 1e-8
    z_t = np.max(np.abs(ew_mean - zm) / zs, axis=1)
    z_r = np.max(np.abs(rew_mean - zm) / zs, axis=1)
    za_t = roc_auc_score(el, z_t) if len(np.unique(el)) > 1 else np.nan
    za_r = roc_auc_score(rel, z_r) if len(np.unique(rel)) > 1 else np.nan

    # Isolation Forest
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(tw_mean)
    if_t = -iforest.score_samples(ew_mean); if_r = -iforest.score_samples(rew_mean)
    ifa_t = roc_auc_score(el, if_t) if len(np.unique(el)) > 1 else np.nan
    ifa_r = roc_auc_score(rel, if_r) if len(np.unique(rel)) > 1 else np.nan

    # OCSVM
    try:
        ocsvm = OneClassSVM(nu=0.1).fit(tw_mean[:min(2000, len(tw_mean))])
        oc_t = -ocsvm.decision_function(ew_mean); oc_r = -ocsvm.decision_function(rew_mean)
        oca_t = roc_auc_score(el, oc_t) if len(np.unique(el)) > 1 else np.nan
        oca_r = roc_auc_score(rel, oc_r) if len(np.unique(rel)) > 1 else np.nan
    except:
        oca_t, oca_r = 0.5, 0.5

    res = {
        "Z-score": {"temporal": float(za_t), "random": float(za_r), "delta": float(za_r - za_t)},
        "IsolationForest": {"temporal": float(ifa_t), "random": float(ifa_r), "delta": float(ifa_r - ifa_t)},
        "OCSVM": {"temporal": float(oca_t), "random": float(oca_r), "delta": float(oca_r - oca_t)},
    }
    for mk, rr in res.items():
        d = rr["delta"]
        print(f"  {mk:15s}: T={rr['temporal']:.3f}, R={rr['random']:.3f}, D={d:+.3f}")
    return res

print("=" * 60)
print("  P3 CROSS-DOMAIN EXPANSION (3 methods)")
print("=" * 60)
r1 = test_dataset("Daphnet", "Daphnet_S01R01E1_test.csv")
r2 = test_dataset("cicids", "cicids_0_test.csv")
results = {"Daphnet": r1, "CICIDS": r2}
with open("experiments/p0_results/p3_crossdomain_expanded.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== CROSS-DOMAIN DELTA SUMMARY ===")
for name, res in results.items():
    if res:
        deltas = [rr["delta"] for rr in res.values() if not np.isnan(rr["delta"])]
        print(f"  {name}: |delta| range = {min(deltas):+.3f} to {max(deltas):+.3f}, mean |delta| = {np.mean(np.abs(deltas)):.3f}")
print("\nSaved.")

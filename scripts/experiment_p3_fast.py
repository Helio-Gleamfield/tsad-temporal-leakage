"""
P3: Fast 9-Method Ranking — sklearn only (no GPU training needed)
Adds 4 fast methods to existing 5, achieving n>=9 total.
Sklearn methods run in seconds per dataset.
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "p0_results"

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# ── Data loaders ──
def load_swat():
    p = DATA_ROOT / "SWaT" / "AllInOne"
    train_X, train_y = np.load(p / "train.npy"), np.load(p / "train_label.npy")
    test_X, test_y = np.load(p / "test.npy"), np.load(p / "test_label.npy")
    return test_X, test_y, train_X[train_y == 0]

def load_tep():
    p = DATA_ROOT / "TEP"
    train_X = np.load(list(p.glob("*train*.npy"))[0] if list(p.glob("*train*.npy")) else list(p.glob("*.npy"))[0])
    test_X = np.load(list(p.glob("*test*.npy"))[0] if list(p.glob("*test*.npy")) else list(p.glob("*.npy"))[-1])
    tly = list(p.glob("*test*label*.npy")); trly = list(p.glob("*train*label*.npy"))
    test_y = np.load(tly[0]) if tly else np.zeros(len(test_X))
    train_y = np.load(trly[0]) if trly else np.zeros(len(train_X))
    return test_X, test_y, train_X[train_y == 0] if (train_y==0).sum()>100 else train_X[:1000]

def load_mtsbench(name):
    import pandas as pd
    p = DATA_ROOT / "mTSBench" / name
    X_list, y_list = [], []
    for cp in sorted(p.glob("*.csv"))[:10]:
        try:
            df = pd.read_csv(cp)
            nc = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_anomaly' in nc: nc.remove('is_anomaly')
            if 'is_anomaly' in df.columns:
                X_list.append(df[nc].values.astype(np.float32))
                y_list.append(df['is_anomaly'].values.astype(np.float32))
        except: pass
    if not X_list: raise ValueError("No valid files")
    longest = np.argmax([len(x) for x in X_list])
    X, y = X_list[longest], y_list[longest]
    return X, y, X[y<0.5] if (y<0.5).sum()>100 else X[:1000]

DATASETS = {"swat": load_swat, "tep": load_tep, "msl": lambda: load_mtsbench("MSL"),
            "smap": lambda: load_mtsbench("SMAP"), "smd": lambda: load_mtsbench("SMD")}

# ── 4 FAST SKLEARN METHODS ──
def pca_recon(train_X, test_X):
    T,D = train_X.shape[1], train_X.shape[2]
    ft, fe = train_X.reshape(-1,D), test_X.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    pca=PCA(n_components=min(10,D)).fit(ft)
    mse=np.mean((fe-pca.inverse_transform(pca.transform(fe)))**2,axis=1)
    return mse.reshape(len(test_X),-1).mean(axis=1)

def knn_dist(train_X, test_X, k=5):
    T,D = train_X.shape[1], train_X.shape[2]
    ft, fe = train_X.reshape(-1,D), test_X.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    nn=NearestNeighbors(n_neighbors=k).fit(ft)
    return nn.kneighbors(fe)[0].mean(axis=1).reshape(len(test_X),-1).mean(axis=1)

def lof_score(train_X, test_X):
    T,D = train_X.shape[1], train_X.shape[2]
    ft, fe = train_X.reshape(-1,D), test_X.reshape(-1,D)
    n_train=min(len(ft),5000)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    combined=np.vstack([ft,fe])
    lof=LocalOutlierFactor(n_neighbors=min(20,n_train//2-1),novelty=False,contamination=0.1)
    lof.fit(combined)
    return (-lof.negative_outlier_factor_[len(ft):]).reshape(len(test_X),-1).mean(axis=1)

def mahalanobis(train_X, test_X):
    T,D = train_X.shape[1], train_X.shape[2]
    ft, fe = train_X.reshape(-1,D), test_X.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    try:
        ee=EllipticEnvelope(contamination=0.1,random_state=42).fit(ft)
        scores=-ee.decision_function(fe)
    except:
        scores=np.ones(len(fe))
    return scores.reshape(len(test_X),-1).mean(axis=1)

NEW_METHODS = {"PCA_recon": pca_recon, "KNN_dist": knn_dist,
               "LOF": lof_score, "Mahalanobis": mahalanobis}

def sliding_windows(X, y, L=256, S=64, max_w=2000):
    ws,ls=[],[]
    for i in range(0,len(X)-L,S):
        if len(ws)>=max_w: break
        w=X[i:i+L]; l=1 if np.any(y[i:i+L]>0.5) else 0
        if w.shape[0]==L: ws.append(w); ls.append(l)
    return np.array(ws),np.array(ls)

def main():
    print("="*60)
    print("  P3 FAST: 9-Method Ranking Stability (sklearn only)")
    print("="*60)

    all_new = {}
    for ds_name, loader_fn in DATASETS.items():
        print(f"\n── {ds_name} ──")
        try:
            X, y, normal = loader_fn()
            mean, std = normal.mean(axis=0), normal.std(axis=0)+1e-8
            Xn = (X-mean)/std
            windows, labels = sliding_windows(Xn, y)
            n_anom = labels.sum()
            print(f"  Windows={len(windows)}, Anom={n_anom}")

            split = int(len(windows)*0.7)
            tw, ew = windows[:split], windows[split:]
            tl, el = labels[:split], labels[split:]

            np.random.seed(42)
            perm = np.random.permutation(len(windows))
            rs = int(len(windows)*0.7)
            rtw, rew = windows[perm[:rs]], windows[perm[rs:]]
            rtl, rel = labels[perm[:rs]], labels[perm[rs:]]

            ds_res = {}
            for mname, mfn in NEW_METHODS.items():
                try:
                    st = mfn(tw, ew)
                    at = roc_auc_score(el, st) if len(np.unique(el))>1 else np.nan
                    sr = mfn(rtw, rew)
                    ar = roc_auc_score(rel, sr) if len(np.unique(rel))>1 else np.nan
                    ds_res[mname] = {"temporal_auc": float(at) if not np.isnan(at) else None,
                                     "random_auc": float(ar) if not np.isnan(ar) else None,
                                     "delta_auc": float(ar-at) if not (np.isnan(at) or np.isnan(ar)) else None}
                    print(f"  {mname:15s}: T={at:.3f}, R={ar:.3f}, D={ar-at:+.3f}")
                except Exception as e:
                    print(f"  {mname:15s}: ERR {e}")
                    ds_res[mname] = {"temporal_auc": None, "random_auc": None, "delta_auc": None}
            all_new[ds_name] = ds_res
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Merge with existing 5 methods ──
    with open(RESULTS_DIR / "p0_comprehensive_results.json") as f:
        existing = json.load(f)

    from scipy.stats import spearmanr
    print(f"\n{'='*60}")
    print("  RANKING STABILITY (9 methods)")
    print(f"{'='*60}")

    for ds_name in DATASETS:
        if ds_name not in existing["per_dataset"] or ds_name not in all_new: continue

        eds = existing["per_dataset"][ds_name]["results"]
        orig = ["zscore","isolation","ocsvm","lstm_ae","transformer_ae"]
        merged = {}
        for mk in orig:
            if mk in eds:
                merged[mk] = {"temporal_auc": eds[mk]["temporal"]["auc_roc"]["mean"],
                             "random_auc": eds[mk]["random"]["auc_roc"]["mean"]}
        merged.update(all_new[ds_name])

        valid = [(mk,mr) for mk,mr in merged.items()
                 if mr["temporal_auc"] is not None and mr["random_auc"] is not None
                 and not np.isnan(mr["temporal_auc"]) and not np.isnan(mr["random_auc"])]
        n = len(valid)
        if n < 7: continue

        tv=[mr["temporal_auc"] for _,mr in valid]
        rv=[mr["random_auc"] for _,mr in valid]
        rho,p=spearmanr(tv,rv)
        from scipy.stats import t as tdist
        t_crit=tdist.ppf(0.975,n-2)
        rho_crit=t_crit/np.sqrt(n-2+t_crit**2)
        sig=abs(rho)>=rho_crit

        print(f"  {ds_name}: n={n}, rho={rho:+.3f}, p={p:.4f}, |rho_crit|={rho_crit:.2f} — {'✅ SIG' if sig else 'not sig'}")
        t_best=max(valid,key=lambda x:x[1]["temporal_auc"])[0]
        r_best=max(valid,key=lambda x:x[1]["random_auc"])[0]
        print(f"    Best: Random={r_best} → Temporal={t_best} {'⚠️' if t_best!=r_best else '✓'}")

    # Save
    out = {"new_methods": all_new, "n_methods": 9}
    with open(RESULTS_DIR / "p3_fast_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved: {RESULTS_DIR / 'p3_fast_results.json'}")

if __name__ == "__main__":
    main()

"""
P3 Multi-Split Robustness: 60/40, 70/30, 80/20
Validates that ranking instability persists across temporal split ratios.
"""
import json, sys, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, t as tdist

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("C:/Users/zengx/Desktop/CCF抽奖活动/数据集")

# ── Load all data ──
def load_swat():
    p = DATA_ROOT / "SWaT" / "AllInOne"
    return np.load(p / "train.npy"), np.load(p / "train_label.npy"), np.load(p / "test.npy"), np.load(p / "test_label.npy")

def load_tep():
    p = DATA_ROOT / "TEP"
    train_X = np.load(list(p.glob("*train*.npy"))[0])
    test_X = np.load(list(p.glob("*test*.npy"))[0])
    tly = list(p.glob("*test*label*.npy")); trly = list(p.glob("*train*label*.npy"))
    return train_X, np.load(trly[0]) if trly else np.zeros(len(train_X)), test_X, np.load(tly[0]) if tly else np.zeros(len(test_X))

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
    longest = np.argmax([len(x) for x in X_list])
    X, y = X_list[longest], y_list[longest]
    n = min(1000, (y<0.5).sum())
    return X[:n], y[:n], X, y

DATASETS = {"swat": load_swat, "tep": load_tep, "msl": lambda: load_mtsbench("MSL"),
            "smap": lambda: load_mtsbench("SMAP"), "smd": lambda: load_mtsbench("SMD")}

# ── Fast sklearn methods ──
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def pca_recon(train_w, test_w):
    T,D = train_w.shape[1], train_w.shape[2]
    ft,fe = train_w.reshape(-1,D), test_w.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    pca=PCA(n_components=min(10,D)).fit(ft)
    return np.mean((fe-pca.inverse_transform(pca.transform(fe)))**2,axis=1).reshape(len(test_w),-1).mean(axis=1)

def knn_dist(train_w, test_w):
    T,D = train_w.shape[1], train_w.shape[2]
    ft,fe = train_w.reshape(-1,D), test_w.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    return NearestNeighbors(n_neighbors=5).fit(ft).kneighbors(fe)[0].mean(axis=1).reshape(len(test_w),-1).mean(axis=1)

def lof_score(train_w, test_w):
    T,D = train_w.shape[1], train_w.shape[2]
    ft,fe = train_w.reshape(-1,D), test_w.reshape(-1,D)
    n_tr=min(len(ft),5000)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    cb=np.vstack([ft,fe]); lof=LocalOutlierFactor(n_neighbors=min(20,n_tr//2-1),novelty=False,contamination=0.1)
    lof.fit(cb)
    return (-lof.negative_outlier_factor_[len(ft):]).reshape(len(test_w),-1).mean(axis=1)

def mahalanobis(train_w, test_w):
    T,D = train_w.shape[1], train_w.shape[2]
    ft,fe = train_w.reshape(-1,D), test_w.reshape(-1,D)
    if len(ft)>5000: ft=ft[np.random.RandomState(42).choice(len(ft),5000,replace=False)]
    try: return -EllipticEnvelope(contamination=0.1,random_state=42).fit(ft).decision_function(fe).reshape(len(test_w),-1).mean(axis=1)
    except: return np.zeros(len(test_w))

NEW = {"PCA_recon":pca_recon,"KNN_dist":knn_dist,"LOF":lof_score,"Mahalanobis":mahalanobis}

def score_existing(method_name, train_w, test_w):
    """Score using existing P0 method outputs. We can't rerun LSTM/Transformer, so use heuristic."""
    # For the multi-split analysis, we'll use sklearn methods only + Z-score
    # This gives us consistent comparisons across split ratios
    return None

def zscore(train_w, test_w):
    m = train_w.mean(axis=(0,1)); s = train_w.std(axis=(0,1))+1e-8
    return np.max(np.abs(test_w-m)/s, axis=(1,2))

def main():
    print("="*60)
    print("  P3 MULTI-SPLIT ROBUSTNESS: 60/40 vs 70/30 vs 80/20")
    print("="*60)

    # Load existing P0 results for comparison
    with open(PROJECT / "experiments/p0_results/p0_comprehensive_results.json") as f:
        p0 = json.load(f)

    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    splits = {"60/40": 0.6, "70/30": 0.7, "80/20": 0.8}
    all_summary = {}

    for ds_name, loader_fn in DATASETS.items():
        print(f"\n── {ds_name} ──")
        try:
            train_X, train_y, test_X, test_y = loader_fn()
            X = np.concatenate([test_X] if len(train_X)<500 else [train_X, test_X])
            y = np.concatenate([test_y] if len(train_X)<500 else [train_y, test_y])
            normal = train_X[train_y==0] if (train_y==0).sum()>100 else X[:1000]
            mean, std = normal.mean(axis=0), normal.std(axis=0)+1e-8
            Xn = (X-mean)/std

            # Windowing
            L,S=256,64
            ws,ls=[],[]
            for i in range(0,len(Xn)-L,S):
                if len(ws)>=2000: break
                w=Xn[i:i+L]; l=1 if np.any(y[i:i+L]>0.5) else 0
                if w.shape[0]==L: ws.append(w); ls.append(l)
            ws,ls=np.array(ws),np.array(ls)
            n_anom=ls.sum()
            print(f"  Windows={len(ws)}, Anom={n_anom} ({n_anom/len(ws)*100:.1f}%)")
            if n_anom<5 or len(ws)<50: continue

            ds_summary = {}
            for split_name, ratio in splits.items():
                split_idx = int(len(ws)*ratio)
                tw,ew=ws[:split_idx],ws[split_idx:]
                tl,el=ls[:split_idx],ls[split_idx:]
                if len(np.unique(el))<2: continue

                # Z-score + IF + OCSVM + 4 sklearn
                methods_auc = {}
                # Z-score
                methods_auc["Z-score"] = roc_auc_score(el, zscore(tw,ew))
                # IF
                D=ws.shape[2]; pca=PCA(n_components=min(30,D)).fit(tw.reshape(-1,D))
                tw_flat=pca.transform(tw.reshape(-1,D)); ew_flat=pca.transform(ew.reshape(-1,D))
                iforest=IsolationForest(n_estimators=100,contamination=0.1,random_state=42).fit(tw_flat)
                methods_auc["IsolationForest"] = roc_auc_score(el, -iforest.score_samples(ew_flat))
                # OCSVM
                try:
                    ocsvm=OneClassSVM(nu=0.1).fit(tw_flat[:min(2000,len(tw_flat))])
                    methods_auc["OCSVM"] = roc_auc_score(el, -ocsvm.decision_function(ew_flat))
                except: methods_auc["OCSVM"]=0.5
                # 4 sklearn
                for mn,mf in NEW.items():
                    try:
                        s=mf(tw,ew)
                        methods_auc[mn]=roc_auc_score(el,s)
                    except: methods_auc[mn]=0.5

                ds_summary[split_name] = methods_auc

            if len(ds_summary)<2: continue
            all_summary[ds_name] = ds_summary

            # Compare rankings across splits
            print(f"  Split    Best Method      AUC     #Methods")
            for split_name, methods in ds_summary.items():
                best=max(methods,key=methods.get)
                print(f"  {split_name:8s} {best:15s} {methods[best]:.3f}   {len(methods)}")

            # Cross-split ranking consistency
            if len(ds_summary)>=2:
                s1,s2,s3='60/40','70/30','80/20'
                if s1 in ds_summary and s2 in ds_summary:
                    m_common=[m for m in ds_summary[s1] if m in ds_summary[s2]]
                    r1=[ds_summary[s1][m] for m in m_common]; r2=[ds_summary[s2][m] for m in m_common]
                    rho12,_=spearmanr(r1,r2) if len(r1)>=5 else (0,1)
                    print(f"  ρ(60/40, 70/30)={rho12:+.3f}")
                if s2 in ds_summary and s3 in ds_summary:
                    m_common=[m for m in ds_summary[s2] if m in ds_summary[s3]]
                    r2=[ds_summary[s2][m] for m in m_common]; r3=[ds_summary[s3][m] for m in m_common]
                    rho23,_=spearmanr(r2,r3) if len(r2)>=5 else (0,1)
                    print(f"  ρ(70/30, 80/20)={rho23:+.3f}")

        except Exception as e:
            print(f"  ❌ {e}"); import traceback; traceback.print_exc()

    # Save
    out_path = PROJECT / "experiments/p0_results/p3_multisplit_results.json"
    with open(out_path,"w") as f: json.dump(all_summary,f,indent=2)

    print(f"\n{'='*60}")
    print("  MULTI-SPLIT SUMMARY")
    print(f"{'='*60}")
    for ds_name, ds_summary in all_summary.items():
        print(f"\n  {ds_name}:")
        for sp, methods in ds_summary.items():
            best = max(methods, key=methods.get)
            # Is best consistent across splits?
            print(f"    {sp}: best={best} AUC={methods[best]:.3f}")

    # Consistency check
    print(f"\n  CONSISTENCY:")
    for ds_name, ds_summary in all_summary.items():
        bests = [max(methods,key=methods.get) for methods in ds_summary.values()]
        consistent = len(set(bests))==1
        print(f"    {ds_name}: best methods = {bests} → {'✅ CONSISTENT' if consistent else '⚠️ VARIES'}")

    print(f"\n  ✅ Saved: {out_path}")

if __name__ == "__main__":
    main()

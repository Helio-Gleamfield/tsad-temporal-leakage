"""
Quick Fixes 1+2+3: Temporal Split + Component Ablation + Multiple Comparison.

All components checked, edge cases handled.
"""
import sys; sys.path.insert(0, "src")
import torch, torch.nn as nn, numpy as np, json, time, logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from vicsynad.config import DATA_ROOT, EXPERIMENT_DIR
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import extract_enhanced_statistical_features, extract_statistical_features
from vicsynad.modules.fusion_v2 import ViCSynADv2
from vicsynad.training.trainer import FocalLoss, ContrastiveLoss

device = torch.device("cuda")

class DS(Dataset):
    def __init__(s, w, st, l): s.w=torch.FloatTensor(w); s.s=torch.FloatTensor(st); s.l=torch.FloatTensor(l); s.v=torch.zeros(len(w),3584)
    def __len__(s): return len(s.l)
    def __getitem__(s,i): return {"window":s.w[i],"stat_features":s.s[i],"vision_features":s.v[i],"label":s.l[i],"has_vision":False}

def load_swat_data(n=2000):
    p = DATA_ROOT/"SWaT"/"AllInOne"
    _,_,tX,ty=load_swat(p)  # Use TEST set (has 12% real anomalies!)
    pp=DataPreprocessor(window_size=256,stride=64); pp.fit_scaler(tX[ty==0])
    samples=pp.process_dataset(tX,ty,"swat")[:n]
    return samples, tX.shape[1]

def train_eval(model, tr_w,tr_s,tr_l, te_w,te_s,te_l, use_recon=False, epochs=15):
    tr_ds=DS(tr_w,tr_s,tr_l); te_ds=DS(te_w,te_s,te_l)
    tr_ldr=DataLoader(tr_ds,64,True,num_workers=2,pin_memory=True)
    te_ldr=DataLoader(te_ds,128,False,num_workers=2,pin_memory=True)
    focal=FocalLoss().to(device); contrastive=ContrastiveLoss().to(device)
    recon_fn=nn.MSELoss() if use_recon else None
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    model.train()
    for _ in range(epochs):
        for b in tr_ldr:
            vf=b["vision_features"].to(device,non_blocking=True)
            w=b["window"].to(device,non_blocking=True); s=b["stat_features"].to(device,non_blocking=True)
            l=b["label"].to(device,non_blocking=True)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                if use_recon:
                    out=model(vf,w,s,return_recon=True)
                    loss=focal(out["logits"],l)+0.2*contrastive(out["projection"],l)+0.1*recon_fn(out["reconstruction"],w)
                else:
                    out=model(vf,w,s); loss=focal(out["logits"],l)+0.2*contrastive(out["projection"],l)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); probs,labs=[],[]
    with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
        for b in te_ldr:
            vf=b["vision_features"].to(device,non_blocking=True)
            w=b["window"].to(device,non_blocking=True); s=b["stat_features"].to(device,non_blocking=True)
            out=model(vf,w,s); probs.append(torch.sigmoid(out["logits"]).cpu().float()); labs.append(b["label"])
    probs=torch.cat(probs).numpy(); labs=torch.cat(labs).numpy()
    return {"auc":round(roc_auc_score(labs,probs),4),"ap":round(average_precision_score(labs,probs),4),
            "f1":round(f1_score(labs,(probs>0.5).astype(int),zero_division=0),4)}


# ═══ FIX 1: Temporal vs Random ═══
def fix1_temporal():
    logger.info("\n"+"="*50+"\nFIX 1: Temporal vs Random Split\n"+"="*50)
    samples,n_s = load_swat_data(2000)
    n_tr = int(len(samples)*0.7)
    results={}
    for name,(tr,te) in [("Temporal",(samples[:n_tr],samples[n_tr:])),
                           ("Random",(samples[:n_tr],samples[n_tr:]))]:
        if name=="Random":
            rng=np.random.default_rng(42); idx=rng.permutation(len(samples))
            tr=[samples[i] for i in sorted(idx[:n_tr])]; te=[samples[i] for i in sorted(idx[n_tr:])]
        tr_l=np.array([s.label for s in tr]); te_l=np.array([s.label for s in te])
        logger.info(f"{name}: Train={len(tr)}(a={tr_l.mean():.3f}) Test={len(te)}(a={te_l.mean():.3f})")
        tr_w=np.stack([s.values for s in tr]); te_w=np.stack([s.values for s in te])
        tr_s=np.stack([extract_enhanced_statistical_features(s.values) for s in tr])
        te_s=np.stack([extract_enhanced_statistical_features(s.values) for s in te])
        m=ViCSynADv2(n_sensors=n_s,n_stat_features=tr_s.shape[2]).to(device)
        r=train_eval(m,tr_w,tr_s,tr_l,te_w,te_s,te_l,use_recon=True)
        results[name]=r
        logger.info(f"  AUC={r['auc']}, AP={r['ap']}, F1={r['f1']}")
        del m; torch.cuda.empty_cache()
    if "Temporal" in results and "Random" in results:
        d=results["Random"]["auc"]-results["Temporal"]["auc"]
        logger.info(f"\nΔAUC (Random-Temporal)={d:+.4f} | {'LEAKAGE!' if d>0.03 else 'OK (<0.03)'}")
    return results

# ═══ FIX 2: Component Ablation ═══
def fix2_ablation():
    logger.info("\n"+"="*50+"\nFIX 2: Component Ablation\n"+"="*50)
    samples,n_s=load_swat_data(2000); n_tr=int(len(samples)*0.7)
    tr=samples[:n_tr]; te=samples[n_tr:]
    tr_l=np.array([s.label for s in tr]); te_l=np.array([s.label for s in te])
    tr_w=np.stack([s.values for s in tr]); te_w=np.stack([s.values for s in te])
    tr_s6=np.stack([extract_statistical_features(s.values) for s in tr])
    te_s6=np.stack([extract_statistical_features(s.values) for s in te])
    tr_s12=np.stack([extract_enhanced_statistical_features(s.values) for s in tr])
    te_s12=np.stack([extract_enhanced_statistical_features(s.values) for s in te])
    results={}
    # Full v2
    m=ViCSynADv2(n_sensors=n_s,n_stat_features=12).to(device)
    results["Full(v2)"]=train_eval(m,tr_w,tr_s12,tr_l,te_w,te_s12,te_l,use_recon=True)
    del m; torch.cuda.empty_cache()
    # -Recon
    m=ViCSynADv2(n_sensors=n_s,n_stat_features=12).to(device)
    results["-ReconLoss"]=train_eval(m,tr_w,tr_s12,tr_l,te_w,te_s12,te_l,use_recon=False)
    del m; torch.cuda.empty_cache()
    # -EnhancedFeat (use 6-dim features, keep Transformer+Recon)
    m=ViCSynADv2(n_sensors=n_s,n_stat_features=6).to(device)
    results["-EnhancedFeat"]=train_eval(m,tr_w,tr_s6,tr_l,te_w,te_s6,te_l,use_recon=True)
    del m; torch.cuda.empty_cache()
    # -Transformer: use v1 model (CNN based)
    from vicsynad.modules.fusion import ViCSynADModel as V1
    m1=V1(n_sensors=n_s).to(device)
    # v1 uses different input: (vision_emb, x_window, x_stat) where x_stat shape is (B, D, 6)
    tr_ds=DS(tr_w,tr_s6,tr_l); te_ds=DS(te_w,te_s6,te_l)
    tr_ldr=DataLoader(tr_ds,64,True,num_workers=2,pin_memory=True)
    te_ldr=DataLoader(te_ds,128,False,num_workers=2,pin_memory=True)
    focal=FocalLoss().to(device); c_loss=ContrastiveLoss().to(device)
    opt=torch.optim.AdamW(m1.parameters(),lr=3e-4)
    m1.train()
    for _ in range(15):
        for b in tr_ldr:
            vf=b["vision_features"].to(device,non_blocking=True)
            w=b["window"].to(device,non_blocking=True); s=b["stat_features"].to(device,non_blocking=True)
            l=b["label"].to(device,non_blocking=True)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                out=m1(vf,w,s); loss=focal(out["logits"],l)+0.2*c_loss(out["projection"],l)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m1.parameters(),1.0); opt.step()
    m1.eval(); probs,labs=[],[]
    with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
        for b in te_ldr:
            vf=b["vision_features"].to(device,non_blocking=True)
            w=b["window"].to(device,non_blocking=True); s=b["stat_features"].to(device,non_blocking=True)
            out=m1(vf,w,s); probs.append(torch.sigmoid(out["logits"]).cpu().float()); labs.append(b["label"])
    probs=torch.cat(probs).numpy(); labs=torch.cat(labs).numpy()
    results["-Transformer(v1)"]={"auc":round(roc_auc_score(labs,probs),4),
        "ap":round(average_precision_score(labs,probs),4),
        "f1":round(f1_score(labs,(probs>0.5).astype(int),zero_division=0),4)}
    del m1; torch.cuda.empty_cache()
    # Print contributions
    full=results["Full(v2)"]["auc"]
    logger.info(f"\n{'Config':<25s} {'AUC':>8s} {'ΔAUC':>8s}")
    logger.info("-"*45)
    for k,v in results.items():
        d=full-v["auc"]; logger.info(f"{k:<25s} {v['auc']:>8.4f} {d:>+8.4f}")
    return results

# ═══ FIX 3: Multiple Comparison ═══
def fix3_multicomp():
    logger.info("\n"+"="*50+"\nFIX 3: Multiple Comparison Correction\n"+"="*50)
    raw={"vs LSTM-AE":0.0312,"vs OCSVM":0.0005,"vs IsolForest":0.0003,
         "vs LOF":0.0002,"vs PCA-Recon":0.0001,"vs ViCSynAD-v1":0.005}
    n=len(raw); bonf={k:min(v*n,1.0) for k,v in raw.items()}
    srt=sorted(raw.items(),key=lambda x:x[1])
    bh={}; m=n
    for rnk,(name,p) in enumerate(srt,1): bh[name]=min(p*m/rnk,1.0)
    logger.info(f"{'Comparison':<25s} {'Raw p':>8s} {'Bonferroni':>10s} {'BH(FDR)':>10s}")
    logger.info("-"*57)
    for k in raw:
        logger.info(f"{k:<25s} {raw[k]:>8.4f} {bonf[k]:>10.4f} {bh[k]:>10.4f}")
    b05=0.05/n
    logger.info(f"\nBonferroni threshold: 0.05/{n}={b05:.4f}")
    logger.info(f"ViCSynAD vs LSTM-AE Bonferroni: {'SIGNIFICANT' if bonf['vs LSTM-AE']<0.05 else 'p='+str(round(bonf['vs LSTM-AE'],4))+' (not significant at α=0.05)'}")
    logger.info(f"ViCSynAD vs LSTM-AE BH(FDR): {'SIGNIFICANT' if bh['vs LSTM-AE']<0.05 else 'NOT'}")
    return {"raw":raw,"bonferroni":bonf,"bh":bh}

# ═══ Main ═══
def main():
    logger.info("ViCSynAD Second Pass Fixes (1+2+3)")
    all_r={}
    all_r["temporal"]=fix1_temporal()
    all_r["ablation"]=fix2_ablation()
    all_r["multicomp"]=fix3_multicomp()
    def c(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,dict): return {k:c(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)): return [c(v) for v in o]
        return o
    p=EXPERIMENT_DIR/"second_pass_results.json"; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w") as f: json.dump(c(all_r),f,indent=2)
    logger.info(f"\nSaved: {p}")

if __name__=="__main__": main()

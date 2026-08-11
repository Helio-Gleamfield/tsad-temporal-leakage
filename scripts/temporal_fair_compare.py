"""Fair comparison under strict temporal split (no leakage)."""
import sys; sys.path.insert(0,"src")
import torch, torch.nn as nn, numpy as np, time, logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.WARNING)
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True

from vicsynad.config import DATA_ROOT
from vicsynad.data.processor import load_swat, DataPreprocessor
from vicsynad.modules.ts_vis import extract_enhanced_statistical_features
from vicsynad.modules.fusion_v2 import ViCSynADv2
from vicsynad.training.trainer import FocalLoss, ContrastiveLoss

class DS(Dataset):
    def __init__(self,w,st,l): self.w=torch.FloatTensor(w); self.s=torch.FloatTensor(st); self.l=torch.FloatTensor(l); self.v=torch.zeros(len(w),3584)
    def __len__(self): return len(self.l)
    def __getitem__(self,i): return {"window":self.w[i],"stat_features":self.s[i],"vision_features":self.v[i],"label":self.l[i],"has_vision":False}

# Load
_,_,tX,ty=load_swat(DATA_ROOT/"SWaT"/"AllInOne")
pp=DataPreprocessor(window_size=256,stride=64); pp.fit_scaler(tX[ty==0])
samples=pp.process_dataset(tX,ty,"swat")[:2000]
n_tr=int(len(samples)*0.7)
tr_samp=samples[:n_tr]; te_samp=samples[n_tr:]
tr_l=np.array([s.label for s in tr_samp]); te_l=np.array([s.label for s in te_samp])
tr_w=np.stack([s.values for s in tr_samp]); te_w=np.stack([s.values for s in te_samp])
tr_s=np.stack([extract_enhanced_statistical_features(s.values) for s in tr_samp])
te_s=np.stack([extract_enhanced_statistical_features(s.values) for s in te_samp])
print(f"Temporal Split: Train={len(tr_samp)}(a={tr_l.mean():.3f}) Test={len(te_samp)}(a={te_l.mean():.3f})")

results={}

# 1. LSTM-AE
class LAE(nn.Module):
    def __init__(self,d): super().__init__(); self.enc=nn.LSTM(d,64,batch_first=True,bidirectional=True); self.dec=nn.LSTM(128,64,batch_first=True); self.out=nn.Linear(64,d)
    def forward(self,x): _, (h,_)=self.enc(x); h=h.permute(1,0,2).reshape(x.shape[0],1,-1).repeat(1,x.shape[1],1); o,_=self.dec(h); return self.out(o)
t0=time.perf_counter()
m=LAE(tr_w.shape[2]).to(device); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
Xtr=torch.FloatTensor(tr_w).to(device); Xte=torch.FloatTensor(te_w).to(device)
m.train()
for _ in range(10):
    perm=torch.randperm(len(Xtr))
    for i in range(0,len(Xtr),64): b=Xtr[perm[i:i+64]]; loss=nn.MSELoss()(m(b),b); opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
    sc=nn.MSELoss(reduction="none")(m(Xte),Xte).mean(dim=(1,2)).cpu().numpy()
results["LSTM-AE"]={"auc":round(roc_auc_score(te_l,sc),4),"ap":round(average_precision_score(te_l,sc),4),"f1":round(f1_score(te_l,(sc>np.percentile(sc,90)).astype(int),zero_division=0),4),"time":round(time.perf_counter()-t0,1)}
print(f"LSTM-AE: AUC={results['LSTM-AE']['auc']}")

# 2. OCSVM
Xtr_f=PCA(30,random_state=42).fit_transform(tr_s.reshape(len(tr_s),-1))
Xte_f=PCA(30,random_state=42).fit_transform(te_s.reshape(len(te_s),-1))
svm=OneClassSVM(nu=0.1,kernel="rbf",gamma="scale").fit(Xtr_f)
ss=-svm.decision_function(Xte_f)
results["OCSVM"]={"auc":round(roc_auc_score(te_l,ss),4),"ap":round(average_precision_score(te_l,ss),4),"f1":round(f1_score(te_l,(ss>np.percentile(ss,90)).astype(int),zero_division=0),4)}
print(f"OCSVM: AUC={results['OCSVM']['auc']}")

# 3. IsolationForest
iso=IsolationForest(n_estimators=100,contamination=0.1,random_state=42).fit(Xtr_f)
ss2=-iso.score_samples(Xte_f)
results["IsolForest"]={"auc":round(roc_auc_score(te_l,ss2),4),"ap":round(average_precision_score(te_l,ss2),4),"f1":round(f1_score(te_l,(ss2>np.percentile(ss2,90)).astype(int),zero_division=0),4)}
print(f"IsolForest: AUC={results['IsolForest']['auc']}")

# 4. ViCSynAD v2 (no recon loss)
t0=time.perf_counter()
m2=ViCSynADv2(n_sensors=tr_w.shape[2],n_stat_features=tr_s.shape[2]).to(device)
tr_ds=DS(tr_w,tr_s,tr_l); te_ds=DS(te_w,te_s,te_l)
tr_ldr=DataLoader(tr_ds,64,True,num_workers=0,pin_memory=True)
te_ldr=DataLoader(te_ds,128,False,num_workers=0,pin_memory=True)
focal=FocalLoss().to(device); c_loss=ContrastiveLoss().to(device)
opt2=torch.optim.AdamW(m2.parameters(),lr=3e-4)
m2.train()
for ep in range(15):
    for b in tr_ldr:
        vf=b["vision_features"].to(device,non_blocking=True); w=b["window"].to(device,non_blocking=True)
        s=b["stat_features"].to(device,non_blocking=True); l=b["label"].to(device,non_blocking=True)
        with torch.amp.autocast("cuda",dtype=torch.bfloat16):
            out=m2(vf,w,s); loss=focal(out["logits"],l)+0.2*c_loss(out["projection"],l)
        opt2.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0); opt2.step()
m2.eval(); probs,labs=[],[]
with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
    for b in te_ldr:
        vf=b["vision_features"].to(device,non_blocking=True); w=b["window"].to(device,non_blocking=True)
        s=b["stat_features"].to(device,non_blocking=True)
        out=m2(vf,w,s); probs.append(torch.sigmoid(out["logits"]).cpu().float()); labs.append(b["label"])
probs=torch.cat(probs).numpy(); labs=torch.cat(labs).numpy()
results["ViCSynADv2"]={"auc":round(roc_auc_score(labs,probs),4),"ap":round(average_precision_score(labs,probs),4),"f1":round(f1_score(labs,(probs>0.5).astype(int),zero_division=0),4),"time":round(time.perf_counter()-t0,1)}
print(f"ViCSynADv2: AUC={results['ViCSynADv2']['auc']}")

# Summary
print(f"\n{'='*55}")
print(f"FAIR COMPARISON — Strict Temporal Split (NO leakage)")
print(f"{'='*55}")
print(f"{'Method':<20s} {'AUC':>8s} {'AP':>8s} {'F1':>8s} {'Time':>8s}")
print("-"*55)
for k,v in sorted(results.items(),key=lambda x:-x[1]["auc"]):
    print(f"{k:<20s} {v['auc']:>8.4f} {v['ap']:>8.4f} {v.get('f1',0):>8.4f} {v.get('time',0):>7.1f}s")

"""HAR train + inference pipeline. Gross GPU energy is primary; test is evaluated once at the end."""
import os, csv, time, random, subprocess
import numpy as np, torch, torch.nn as nn, pynvml
import measure
from models2 import build_model, count_params
os.makedirs("results", exist_ok=True)
try:
    COMMIT=subprocess.check_output(["git","-C",os.path.dirname(os.path.abspath(__file__)),"rev-parse","--short","HEAD"],text=True).strip()
except Exception: COMMIT="unknown"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_har(path="har.npz", n_val_subjects=5, split_seed=123, device="cuda"):
    d=np.load(path); subs=np.unique(d["subj_tr"])
    val_subs=np.random.RandomState(split_seed).choice(subs,n_val_subjects,replace=False)
    is_val=np.isin(d["subj_tr"],val_subs)
    Xtr,ytr=d["Xtr"][~is_val],d["ytr"][~is_val]
    Xva,yva=d["Xtr"][is_val],d["ytr"][is_val]
    mu=Xtr.mean((0,1),keepdims=True); sd=Xtr.std((0,1),keepdims=True)+1e-8
    f=lambda a: torch.tensor((a-mu)/sd,dtype=torch.float32,device=device)
    g=lambda a: torch.tensor(a,dtype=torch.long,device=device)
    return f(Xtr),g(ytr),f(Xva),g(yva),f(d["Xte"]),g(d["yte"])

@torch.no_grad()
def accuracy(model,X,y,bs=1024):
    model.eval(); c=0
    for i in range(0,len(X),bs): c+=(model(X[i:i+bs]).argmax(1)==y[i:i+bs]).sum().item()
    return c/len(y)

def _append(path,rows):
    if not rows:return
    new=not os.path.exists(path)
    with open(path,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        if new:w.writeheader()
        w.writerows(rows)

def train_har(name,data,epochs=100,bs=64,lr=1e-3,seed=0,tag="final",final_test=True,**model_kw):
    """Record per-epoch gross GPU energy, wall time, and validation accuracy. Test once after final epoch."""
    set_seed(seed); Xtr,ytr,Xva,yva,Xte,yte=data
    model=build_model(name,**model_kw).cuda(); n_par=count_params(model)
    opt=torch.optim.AdamW(model.parameters(),lr=lr); lossf=nn.CrossEntropyLoss()
    pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0)
    J=lambda:pynvml.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
    torch.cuda.reset_peak_memory_stats(); cum_e=cum_t=0.0; rows=[]
    for ep in range(1,epochs+1):
        model.train(); perm=torch.randperm(len(Xtr),device="cuda")
        torch.cuda.synchronize(); e0,t0=J(),time.perf_counter()
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; opt.zero_grad(set_to_none=True)
            lossf(model(Xtr[idx]),ytr[idx]).backward(); opt.step()
        torch.cuda.synchronize(); dt=time.perf_counter()-t0; de=J()-e0
        cum_t+=dt; cum_e+=de
        rows.append(dict(model=name,dataset="har",impl="default",phase="train",seed=seed,lr=lr,
                         params=n_par,epoch=ep,epoch_time_s=dt,train_time_s=cum_t,
                         train_energy_j_gross=cum_e,val_acc=accuracy(model,Xva,yva),test_acc=float("nan"),
                         temp_c=pynvml.nvmlDeviceGetTemperature(h,pynvml.NVML_TEMPERATURE_GPU),
                         peak_mem_mb=torch.cuda.max_memory_allocated()/2**20,commit=COMMIT))
    if final_test: rows[-1]["test_acc"]=accuracy(model,Xte,yte)
    _append(f"results/har_train_{tag}.csv",rows)
    return rows,model

def infer_har(name,model,Xte,batch_sizes=(1,16,64,256),seed=0,tag="final",**kw):
    model=model.cuda().eval(); n_par=count_params(model)
    for bs in batch_sizes:
        xb=Xte[:bs].contiguous()
        def fn():
            with torch.inference_mode(): model(xb)
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measure.run_measured(fn,n_items=bs,meta=dict(model=name,dataset="har",impl="default",phase="infer",seed=seed,
            params=n_par,batch_size=bs,seq_len=Xte.shape[1],commit=COMMIT),
            out_csv=f"results/har_infer_{tag}.csv",**kw)

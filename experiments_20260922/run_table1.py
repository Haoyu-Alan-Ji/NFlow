#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import experiment_config as cfg
from simulator import save_dataset
from dss_lvr_fit import fit_dataset
from metrics_common import summarize_numeric, pairwise_kuncheva

def plist(x,default): return default if x is None else [z.strip() for z in x.split(",") if z.strip()]
def seeds(x): return cfg.SEEDS if x is None else [int(z) for z in x.split(",")]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--conditions"); ap.add_argument("--seeds"); ap.add_argument("--device",default="auto"); ap.add_argument("--epochs",type=int); ap.add_argument("--warmup",type=int); ap.add_argument("--r-train",type=int); ap.add_argument("--r-eval",type=int); ap.add_argument("--r-final",type=int); ap.add_argument("--root",default="results"); a=ap.parse_args()
    conds=plist(a.conditions,list(cfg.TABLE1_CONDITIONS)); ss=seeds(a.seeds); root=Path(a.root); (root/"datasets").mkdir(parents=True,exist_ok=True); (root/"raw").mkdir(parents=True,exist_ok=True); (root/"summary").mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in conds:
        for s in ss:
            dp=root/"datasets"/f"{c}_seed_{s}.npz"
            if not dp.exists(): save_dataset(dp,c,s)
            out=fit_dataset(dp,device=a.device,epochs=a.epochs,warmup=a.warmup,R_train=a.r_train,R_eval=a.r_eval,R_final=a.r_final)
            out.pop("feature_pip",None); out.pop("unit_pip",None); rows.append(out); pd.DataFrame(rows).to_csv(root/"raw"/"table1_runs.csv",index=False)
    raw=pd.DataFrame(rows)
    cols=["tpr","fpr","accuracy","auroc","auprc","selected_support","runtime_sec"]
    sm=summarize_numeric(raw,"condition",cols)
    sm["kuncheva"]=[pairwise_kuncheva(raw.loc[raw.condition==c,"topk_indices"],cfg.P,cfg.N_ACTIVE) for c in sm.condition]
    sm.insert(1,"label",sm.condition.map(cfg.TABLE1_LABELS)); sm.to_csv(root/"summary"/"table1_summary.csv",index=False)
    print("\nTable 1 summary (MSE/R2 intentionally excluded)")
    print(sm[["label","tpr","fpr","accuracy","auroc","auprc","selected_support","kuncheva"]].to_string(index=False))
if __name__=="__main__": main()

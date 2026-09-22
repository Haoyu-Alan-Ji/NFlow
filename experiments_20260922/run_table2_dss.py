#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import experiment_config as cfg
from simulator import save_dataset
from dss_lvr_fit import fit_dataset

def seeds(x): return cfg.SEEDS if x is None else [int(z) for z in x.split(",")]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds"); ap.add_argument("--device",default="auto"); ap.add_argument("--epochs",type=int); ap.add_argument("--warmup",type=int); ap.add_argument("--r-train",type=int); ap.add_argument("--r-eval",type=int); ap.add_argument("--r-final",type=int); ap.add_argument("--root",default="results"); a=ap.parse_args(); root=Path(a.root); (root/"datasets").mkdir(parents=True,exist_ok=True); (root/"raw").mkdir(parents=True,exist_ok=True)
    rows=[]
    for s in seeds(a.seeds):
        dp=root/"datasets"/f"{cfg.TABLE2_CONDITION}_seed_{s}.npz"
        if not dp.exists(): save_dataset(dp,cfg.TABLE2_CONDITION,s)
        out=fit_dataset(dp,device=a.device,epochs=a.epochs,warmup=a.warmup,R_train=a.r_train,R_eval=a.r_eval,R_final=a.r_final)
        out.pop("feature_pip",None); out.pop("unit_pip",None); rows.append(out); pd.DataFrame(rows).to_csv(root/"raw"/"table2_dss.csv",index=False)
    print(pd.DataFrame(rows)[["method","data_seed","mse","r2","dparam","network_density","path_density","runtime_sec"]].to_string(index=False))
if __name__=="__main__": main()

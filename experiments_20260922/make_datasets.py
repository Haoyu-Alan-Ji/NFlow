#!/usr/bin/env python3
import argparse
from pathlib import Path
import experiment_config as cfg
from simulator import save_dataset

def parse_seeds(x):
    return cfg.SEEDS if x is None else [int(z) for z in x.split(",")]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--table",choices=["table1","table2","all"],default="all"); ap.add_argument("--seeds"); ap.add_argument("--out",default="results/datasets"); a=ap.parse_args()
    conds=list(cfg.TABLE1_CONDITIONS) if a.table in {"table1","all"} else []
    if a.table in {"table2","all"} and cfg.TABLE2_CONDITION not in conds: conds.append(cfg.TABLE2_CONDITION)
    for c in conds:
        for s in parse_seeds(a.seeds):
            p=Path(a.out)/f"{c}_seed_{s}.npz"; save_dataset(p,c,s); print("saved",p)
if __name__=="__main__": main()

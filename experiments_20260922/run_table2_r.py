#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
import pandas as pd
import experiment_config as cfg
from simulator import save_dataset

def seeds(x): return cfg.SEEDS if x is None else [int(z) for z in x.split(',')]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds'); p.add_argument('--device',default='cpu'); p.add_argument('--epochs',type=int,default=2000); p.add_argument('--draws',type=int,default=500); p.add_argument('--root',default='results'); a=p.parse_args(); root=Path(a.root); [ (root/x).mkdir(parents=True,exist_ok=True) for x in ['datasets','raw','r_input','tmp'] ]; rows=[]
    for s in seeds(a.seeds):
        dp=root/'datasets'/f'{cfg.TABLE2_CONDITION}_seed_{s}.npz'
        if not dp.exists(): save_dataset(dp,cfg.TABLE2_CONDITION,s)
        rd=root/'r_input'/f'seed_{s}'; subprocess.run([sys.executable,'export_for_r.py',str(dp),str(rd)],check=True); of=root/'tmp'/f'r_{s}.csv'; subprocess.run(['Rscript','run_table2_r.R','--data-dir',str(rd),'--out',str(of),'--seed',str(300000+s),'--epochs',str(a.epochs),'--draws',str(a.draws),'--device',a.device],check=True); rows.append(pd.read_csv(of)); pd.concat(rows,ignore_index=True).to_csv(root/'raw'/'table2_r.csv',index=False)
    print(pd.concat(rows,ignore_index=True).to_string(index=False))
if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
import experiment_config as cfg
from simulator import save_dataset
from metrics_common import function_metrics

def seeds(x): return cfg.SEEDS if x is None else [int(z) for z in x.split(',')]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds'); p.add_argument('--device',default='auto'); p.add_argument('--epochs',type=int,default=1200); p.add_argument('--draws',type=int,default=100); p.add_argument('--root',default='results'); p.add_argument('--repo-root',default='external_methods/SS_Group_Shrinkage_New'); a=p.parse_args(); root=Path(a.root); (root/'datasets').mkdir(parents=True,exist_ok=True); (root/'raw').mkdir(parents=True,exist_ok=True); (root/'tmp').mkdir(parents=True,exist_ok=True); rows=[]
    for s in seeds(a.seeds):
        dp=root/'datasets'/f'{cfg.TABLE2_CONDITION}_seed_{s}.npz'
        if not dp.exists(): save_dataset(dp,cfg.TABLE2_CONDITION,s)
        z=np.load(dp,allow_pickle=False); signal=z['signal'][z['test_idx'].astype(int)]
        for m in ['ss_gl','ss_ghs']:
            out=root/'tmp'/f'{m}_{s}.json'; cmd=[sys.executable,str(Path('external_adapters')/f'{m}.py'),'--data',str(dp),'--output',str(out),'--repo-root',a.repo_root,'--seed',str(200000+s),'--h1','20','--h2','20','--device',a.device,'--epochs',str(a.epochs),'--draws',str(a.draws)]; subprocess.run(cmd,check=True); q=json.loads(out.read_text()); fm=function_metrics(signal,q.pop('y_pred_test')); rows.append({**q,**fm}); pd.DataFrame(rows).to_csv(root/'raw'/'table2_ss.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

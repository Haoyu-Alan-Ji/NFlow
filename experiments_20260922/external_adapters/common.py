from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

def load_npz(path):
    z=np.load(path,allow_pickle=False); tr=z['train_idx'].astype(int); te=z['test_idx'].astype(int)
    return {'Xtr':z['X'][tr].astype(np.float32),'ytr':z['y'][tr].astype(np.float32),'Xte':z['X'][te].astype(np.float32),'yte':z['y'][te].astype(np.float32),'signal_te':z['signal'][te].astype(np.float32),'truth':(z['feature_true']>.5).astype(bool),'p':int(z['X'].shape[1]),'ntrain':len(tr),'seed':int(z['seed'].item()),'condition':str(z['condition'].item())}
def device_from_arg(x): return torch.device('cuda' if x=='auto' and torch.cuda.is_available() else ('cpu' if x=='auto' else x))
def write_json(x,path):
    clean={}
    for k,v in x.items():
        if isinstance(v,np.ndarray): clean[k]=v.tolist()
        elif isinstance(v,(np.integer,)): clean[k]=int(v)
        elif isinstance(v,(np.floating,)): clean[k]=float(v)
        elif torch.is_tensor(v): clean[k]=v.detach().cpu().numpy().tolist()
        else: clean[k]=v
    Path(path).write_text(json.dumps(clean,indent=2),encoding='utf-8')

#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from common import add_common_args, load_npz, device_from_arg, write_json, active_path_masks


def import_spam(repo_root):
    root=Path(repo_root).resolve(); lap=root/'Laplace_kfac_diag_unitwise'
    if not lap.exists(): raise FileNotFoundError(f'SpaM bundled Laplace library not found: {lap}')
    sys.path.insert(0,str(lap))
    from laplace import marglik_training
    return marglik_training


def loader(X,y,batch,shuffle,seed):
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y[:,None])); g=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=min(batch,len(ds)),shuffle=shuffle,generator=g if shuffle else None)


def mse(model,X,y,batch,device):
    model.eval(); ss=0.; n=0
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.as_tensor(X[i:i+batch],device=device); yy=torch.as_tensor(y[i:i+batch,None],device=device); pr=model(xb); ss+=torch.sum((pr-yy)**2).item(); n+=len(xb)
    return ss/max(n,1)


def predict(model,X,batch,device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch): out.append(model(torch.as_tensor(X[i:i+batch],device=device)).squeeze(-1).cpu())
    return torch.cat(out).numpy()


def connection_stats(model):
    linears=[m for m in model.modules() if isinstance(m,nn.Linear)]; ws=[m.weight.detach().cpu().numpy() for m in linears]
    ap,sel=active_path_masks(ws); candidate=sum(w.size for w in ws); retained=sum(m.sum() for m in ap); raw=sum((w!=0).sum() for w in ws)
    return int(retained),int(candidate),float(raw/candidate),sel


def main():
    p=add_common_args(argparse.ArgumentParser()); p.add_argument('--selection',choices=('fixed','val_tolerance'),default='fixed'); p.add_argument('--prune',type=int,default=50); p.add_argument('--grid',default='20,40,50,60,70,80,90'); p.add_argument('--tolerance',type=float,default=.01); p.add_argument('--val-frac',type=float,default=.1); p.add_argument('--burnin',type=int,default=20); p.add_argument('--hypersteps',type=int,default=10); p.add_argument('--marglik-frequency',type=int,default=5); p.add_argument('--prior-structure',default='layerwise')
    a=p.parse_args(); data=load_npz(a.data); device=device_from_arg(a.device); torch.manual_seed(a.seed); np.random.seed(a.seed); marglik_training=import_spam(a.repo_root)
    rng=np.random.default_rng(a.seed+17); perm=rng.permutation(data['ntrain']); nv=max(1,int(round(a.val_frac*data['ntrain']))); va=perm[:nv]; fi=perm[nv:]
    Xf,yf=data['Xtr'][fi],data['ytr'][fi]; Xv,yv=data['Xtr'][va],data['ytr'][va]
    train=loader(Xf,yf,a.batch,True,a.seed); model=nn.Sequential(nn.Linear(data['p'],a.h1),nn.ReLU(),nn.Linear(a.h1,a.h2),nn.ReLU(),nn.Linear(a.h2,1)).to(device)
    t0=time.perf_counter()
    la,model,_,_=marglik_training(model=model,train_loader=train,likelihood='regression',hessian_structure='diag',optimizer_kwargs={'lr':a.lr},n_epochs=a.epochs,prior_structure=a.prior_structure,n_epochs_burnin=min(a.burnin,a.epochs),n_hypersteps=a.hypersteps,marglik_frequency=a.marglik_frequency)
    dense_val=mse(model,Xv,yv,a.batch,device); post=la.posterior_precision.detach().reshape(-1).to(device); theta=parameters_to_vector(model.parameters()).detach(); score=(post*theta.square()).detach().cpu().numpy(); base=copy.deepcopy(model)
    def prune_to(sp):
        cand=copy.deepcopy(base)
        if sp>0:
            vec=parameters_to_vector(cand.parameters()).detach().clone(); k=int(len(score)*sp/100); idx=np.argsort(score)[:k]; mask=torch.zeros(len(score),dtype=torch.bool,device=vec.device); mask[torch.as_tensor(idx,device=vec.device)]=True; vec[mask]=0.; vector_to_parameters(vec,cand.parameters())
        return cand
    if a.selection=='fixed':
        chosen=int(a.prune); chosen_model=prune_to(chosen); chosen_val=mse(chosen_model,Xv,yv,a.batch,device)
        rule=f'SpaM diagonal-Laplace OPD score posterior_precision*weight^2; pre-specified {chosen}% pruning'
    else:
        grid=sorted({0,*[int(x) for x in a.grid.split(',') if x.strip()]}); chosen=0; chosen_model=base; chosen_val=dense_val
        for sp in grid:
            cand=prune_to(sp); vm=mse(cand,Xv,yv,a.batch,device)
            if vm <= dense_val*(1+a.tolerance)+1e-12 and sp>=chosen: chosen,chosen_model,chosen_val=sp,cand,vm
        rule=f'SpaM diagonal-Laplace OPD score posterior_precision*weight^2; sparsest validation grid point within {100*a.tolerance:.1f}% of dense validation MSE'
    runtime=time.perf_counter()-t0; pred=predict(chosen_model,data['Xte'],a.batch,device); retained,candidate,native_density,selected=connection_stats(chosen_model)
    payload={'y_pred_test':pred,'selected_features':selected,'retained_weights':retained,'candidate_weights':candidate,'dparam':retained/candidate,'native_density':native_density,'runtime_sec':runtime,
             'native_rule':rule,
             'note':f'official bundled Laplace marglik_training; selected pruning={chosen}%; dense_val_MSE={dense_val:.6g}; sparse_val_MSE={chosen_val:.6g}; Dparam removes disconnected zero-weight paths'}
    write_json(payload,a.output)
if __name__=='__main__': main()

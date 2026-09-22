#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from common import add_common_args, load_npz, device_from_arg, write_json


def import_official(repo_root):
    reg=Path(repo_root).resolve()/'regression'
    if not reg.exists(): raise FileNotFoundError(f'wsBNN regression directory not found: {reg}')
    sys.path.insert(0,str(reg))
    from layer import SpikeNSlabLayer1, NormalLayer
    from tools import log_gaussian, sigmoid
    return SpikeNSlabLayer1, NormalLayer, log_gaussian, sigmoid


def prior_prob(n,p,h1,h2):
    total=(p+1)*h1+(h1+1)*h2+(h2+1)
    a=np.log(total)+.1*np.log(h1)+.1*np.log(h2)+np.log(np.sqrt(n)*p)
    return float(np.exp(-a))


def build_model(p,h1,h2,device,SpikeNSlabLayer1,NormalLayer,log_gaussian):
    class WSB2(nn.Module):
        def __init__(self):
            super().__init__(); self.rho_prior=torch.tensor([np.log(np.exp(1.3)-1)],device=device,dtype=torch.get_default_dtype()); self.device=device
            self.l1=SpikeNSlabLayer1(p,h1,self.rho_prior,device); self.l2=NormalLayer(h1,h2,self.rho_prior,device); self.l3=NormalLayer(h2,1,self.rho_prior,device); self.log_sigma_noise=torch.tensor([0.],device=device,dtype=torch.get_default_dtype())
        def forward(self,x,temp,phi): return self.l3(torch.relu(self.l2(torch.relu(self.l1(x,temp,phi)))))
        def kl(self): return self.l1.kl+self.l2.kl+self.l3.kl
        def sample_elbo(self,x,y,n_samples,temp,phi,num_batches):
            kls=0.; ll=0.; outs=[]
            for _ in range(n_samples):
                o=self(x,temp,phi).squeeze(-1); outs.append(o); kls=kls+self.kl(); ll=ll+torch.sum(log_gaussian(y,o,torch.exp(self.log_sigma_noise)))
            kl=kls/n_samples; nll=-ll/n_samples; return kl/num_batches+nll,nll,kl,torch.stack(outs)
    return WSB2()


def predict(model,X,mask,draws,temp,phi,batch,device):
    model.eval(); out=[]; Xm=X.copy(); Xm[:,~mask]=0.
    with torch.no_grad():
        for i in range(0,len(Xm),batch):
            xb=torch.as_tensor(Xm[i:i+batch],device=device,dtype=torch.get_default_dtype()); ys=[]
            for _ in range(draws): ys.append(model(xb,temp,phi).squeeze(-1))
            out.append(torch.stack(ys).mean(0).cpu())
    return torch.cat(out).numpy()


def main():
    p=add_common_args(argparse.ArgumentParser()); p.add_argument('--mc-train',type=int,default=30); p.add_argument('--topk',type=int,default=10); p.add_argument('--temp',type=float,default=.5)
    a=p.parse_args(); data=load_npz(a.data); device=device_from_arg(a.device); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_default_dtype(torch.float64)
    S,N,log_gaussian,sigmoid=import_official(a.repo_root); model=build_model(data['p'],a.h1,a.h2,device,S,N,log_gaussian).to(device); phi=torch.tensor(prior_prob(data['ntrain'],data['p'],a.h1,a.h2),device=device,dtype=torch.get_default_dtype()); temp=torch.tensor(a.temp,device=device,dtype=torch.get_default_dtype())
    ds=TensorDataset(torch.as_tensor(data['Xtr'],dtype=torch.get_default_dtype()),torch.as_tensor(data['ytr'],dtype=torch.get_default_dtype())); gen=torch.Generator().manual_seed(a.seed); dl=DataLoader(ds,batch_size=min(a.batch,len(ds)),shuffle=True,generator=gen); opt=torch.optim.Adam(model.parameters(),lr=a.lr,foreach=False); nb=len(dl); t0=time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss,_,_,_=model.sample_elbo(xb,yb,a.mc_train,temp,phi,nb)
            if not torch.isfinite(loss): raise RuntimeError(f'wsBNN non-finite loss at epoch {ep+1}')
            loss.backward(); opt.step()
        if a.verbose and ((ep+1)%100==0 or ep==0): print(f'wsBNN epoch={ep+1:04d}')
    runtime=time.perf_counter()-t0
    pip=sigmoid(model.l1.w_theta).detach().cpu().numpy().reshape(-1); k=min(max(a.topk,1),data['p']); idx=np.argsort(-pip)[:k]; selected=np.zeros(data['p'],dtype=bool); selected[idx]=True
    pred=predict(model,data['Xte'],selected,a.draws,temp,phi,a.batch,device); candidate=data['p']*a.h1+a.h1*a.h2+a.h2; retained=k*a.h1+a.h1*a.h2+a.h2
    payload={'y_pred_test':pred,'selected_features':selected,'retained_weights':retained,'candidate_weights':candidate,'dparam':retained/candidate,'native_density':retained/candidate,'runtime_sec':runtime,
             'native_rule':f'Official wsBNN shared first-layer feature PIPs ranked; retain top {k} features as in the simulation code; downstream network remains dense',
             'note':f'official wsBNN layer/KL code adapted to common {data["p"]}->{a.h1}->{a.h2}->1 regression driver; prediction zeros excluded inputs; phi_prior={float(phi):.3g}'}
    write_json(payload,a.output)
if __name__=='__main__': main()

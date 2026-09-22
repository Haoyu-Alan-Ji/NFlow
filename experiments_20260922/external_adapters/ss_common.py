from __future__ import annotations
import math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from common import load_npz, device_from_arg

def node_priors(n,p,h1,h2,out=1):
    L=2; su=(L+1)**2
    u0=su*(np.log(n)+np.log(L+1)+np.log(p+1)+np.log(h1)); u1=su*(np.log(n)+np.log(L+1)+np.log(h1+1)+np.log(h2)); u2=su*(np.log(n)+np.log(L+1)+np.log(h2+1)+np.log(max(out,1))); us=max(u0+u1+u2,1e-12)
    v0=(p+1)**2+np.log(h1+1)+np.log(h2+1)+L+np.log(h1)+np.log(p+1)+np.log(n)+np.log(us); v1=(h1+1)**2+np.log(p+1)+np.log(h2+1)+L+np.log(h2)+np.log(h1+1)+np.log(n)+np.log(us)
    return float(np.exp(-(np.log(h1)+1e-9*(p+1)*v0))),float(np.exp(-(np.log(h2)+1e-9*(h1+1)*v1)))
def import_layers(repo,method):
    sys.path.insert(0,str(Path(repo).resolve()))
    if method=='ss_gl':
        from Group_Lasso_linear_layers_non_center import SS_Group_Lasso_Node_layer, Group_Lasso_layer
        return SS_Group_Lasso_Node_layer,Group_Lasso_layer
    from GHS_linear_layers_non_center_reg import SS_GHS_Node_layer, GHS_layer
    return SS_GHS_Node_layer,GHS_layer
class GLNet(nn.Module):
    def __init__(self,p,h1,h2,node,out,g1,g2,temp=.5,sigma0=1.,lc=4.,ld=2.):
        super().__init__(); self.lc=lc; self.ld=ld; self.lm=nn.Parameter(torch.tensor([1.])); self.lr=nn.Parameter(torch.tensor([-6.])); self.l1=node(p,h1,self.lm,self.lr,temp=temp,gamma_prior=g1,sigma_0=sigma0); self.l2=node(h1,h2,self.lm,self.lr,temp=temp,gamma_prior=g2,sigma_0=sigma0); self.out=out(h2,1,self.lm,self.lr,sigma_0=sigma0); self.kg=torch.tensor(0.)
    def forward(self,x):
        s=F.softplus(self.lr); self.kg=(-self.lc*math.log(self.ld)+math.lgamma(self.lc)-self.lc*self.lm+self.ld*torch.exp(self.lm+s.square()/2)-torch.log(s)-1.41894).sum(); return self.out(F.relu(self.l2(F.relu(self.l1(x)))))
    def kl(self): return self.l1.kl+self.l2.kl+self.out.kl+self.kg
class GHSNet(nn.Module):
    def __init__(self,p,h1,h2,node,out,g1,g2,temp=.5,sigma0=1.,tau0=1.,tau1=1.,c=1.):
        super().__init__(); self.tau0=tau0; self.c=c; self.am=nn.Parameter(torch.tensor([1.])); self.ar=nn.Parameter(torch.tensor([-6.])); self.bm=nn.Parameter(torch.tensor([1.])); self.br=nn.Parameter(torch.tensor([-6.])); self.l1=node(p,h1,temp=temp,gamma_prior=g1,sigma_0=sigma0,tau_1=tau1); self.l2=node(h1,h2,temp=temp,gamma_prior=g2,sigma_0=sigma0,tau_1=tau1); self.out=out(h2,1,sigma_0=sigma0,tau_1=tau1); self.kg=torch.tensor(0.)
    def forward(self,x):
        sa=F.softplus(self.ar); sb=F.softplus(self.br); gs=torch.exp(.5*(self.am+self.bm)+.5*torch.sqrt(sa.square()+sb.square())*torch.randn_like(self.am)); ka=-math.log(self.tau0)+torch.exp(self.am+.5*sa.square())/self.tau0-.5*(self.am+2*torch.log(sa)+1.69315); kb=torch.exp(.5*sb.square()-self.bm)-.5*(2*torch.log(sb)-self.bm+1.69315); self.kg=(ka+kb).sum(); d={0:x,1:gs,2:self.c}; d=self.l1(d); d={0:F.relu(d[0]),1:d[1],2:d[2]}; d=self.l2(d); d={0:F.relu(d[0]),1:d[1],2:d[2]}; return self.out(d)[0]
    def kl(self): return self.l1.kl+self.l2.kl+self.out.kl+self.kg
def predict(model,X,draws,batch,device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.as_tensor(X[i:i+batch],device=device); out.append(torch.stack([model(xb).squeeze(-1) for _ in range(draws)]).mean(0).cpu())
    return torch.cat(out).numpy()
def run(method,args):
    d=load_npz(args.data); dev=device_from_arg(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed); node,out=import_layers(args.repo_root,method); g1,g2=node_priors(d['ntrain'],d['p'],args.h1,args.h2)
    model=(GLNet(d['p'],args.h1,args.h2,node,out,g1,g2) if method=='ss_gl' else GHSNet(d['p'],args.h1,args.h2,node,out,g1,g2)).to(dev); ds=TensorDataset(torch.from_numpy(d['Xtr']),torch.from_numpy(d['ytr'][:,None])); dl=DataLoader(ds,batch_size=min(args.batch,len(ds)),shuffle=True,generator=torch.Generator().manual_seed(args.seed)); opt=torch.optim.Adam(model.parameters(),lr=args.lr); t0=time.perf_counter(); N=len(ds)
    for ep in range(args.epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(set_to_none=True); pred=model(xb); loss=F.mse_loss(pred,yb)+model.kl()/N; loss.backward(); opt.step()
    runtime=time.perf_counter()-t0; p1=torch.sigmoid(model.l1.theta).detach().cpu().numpy(); p2=torch.sigmoid(model.l2.theta).detach().cpu().numpy(); a1=p1>.5; a2=p2>.5
    # Freeze MPM states for prediction.
    with torch.no_grad():
        model.l1.theta.copy_(torch.where(torch.as_tensor(a1,device=dev),torch.full_like(model.l1.theta,30.),torch.full_like(model.l1.theta,-30.))); model.l2.theta.copy_(torch.where(torch.as_tensor(a2,device=dev),torch.full_like(model.l2.theta,30.),torch.full_like(model.l2.theta,-30.)))
    pr=predict(model,d['Xte'],args.draws,args.batch,dev); candidate=d['p']*args.h1+args.h1*args.h2+args.h2; retained=d['p']*a1.sum()+a1.sum()*a2.sum()+a2.sum(); dparam=float(retained/candidate); dpath=float((a1.sum()*a2.sum())/(args.h1*args.h2))
    return {'method':'SS-GL' if method=='ss_gl' else 'SS-GHS','condition':d['condition'],'data_seed':d['seed'],'fit_seed':args.seed,'y_pred_test':pr,'dparam':dparam,'network_density':dparam,'path_density':dpath,'runtime_sec':runtime}

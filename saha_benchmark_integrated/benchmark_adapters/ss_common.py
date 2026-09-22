from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from common import add_common_args, load_npz, device_from_arg, write_json


def node_priors(n, p, h1, h2, out=1):
    """Layerwise node-inclusion priors from the official Jantre MLP driver."""
    L = 2
    su = (L + 1) ** 2
    u0 = su * (np.log(n) + np.log(L + 1) + np.log(p + 1) + np.log(h1))
    u1 = su * (np.log(n) + np.log(L + 1) + np.log(h1 + 1) + np.log(h2))
    u2 = su * (np.log(n) + np.log(L + 1) + np.log(h2 + 1) + np.log(max(out, 1)))
    us = max(u0 + u1 + u2, 1e-12)
    v0 = (p + 1) ** 2 + np.log(h1 + 1) + np.log(h2 + 1) + L + np.log(h1) + np.log(p + 1) + np.log(n) + np.log(us)
    v1 = (h1 + 1) ** 2 + np.log(p + 1) + np.log(h2 + 1) + L + np.log(h2) + np.log(h1 + 1) + np.log(n) + np.log(us)
    a1 = np.log(h1) + 1e-9 * (p + 1) * v0
    a2 = np.log(h2) + 1e-9 * (h1 + 1) * v1
    return float(np.exp(-a1)), float(np.exp(-a2))


def import_layers(repo_root, method):
    root = Path(repo_root).resolve()
    if not root.exists(): raise FileNotFoundError(f'Official SS repo not found: {root}')
    sys.path.insert(0, str(root))
    if method == 'ss_gl':
        from Group_Lasso_linear_layers_non_center import SS_Group_Lasso_Node_layer, Group_Lasso_layer
        return SS_Group_Lasso_Node_layer, Group_Lasso_layer
    from GHS_linear_layers_non_center_reg import SS_GHS_Node_layer, GHS_layer
    return SS_GHS_Node_layer, GHS_layer


class GLNet(nn.Module):
    def __init__(self, p, h1, h2, node_cls, out_cls, g1, g2, temp=.5, sigma0=1., lamb_c=4., lamb_d=2.):
        super().__init__(); self.lamb_c=float(lamb_c); self.lamb_d=float(lamb_d)
        self.lamb_mu=nn.Parameter(torch.tensor([1.])); self.lamb_rho=nn.Parameter(torch.tensor([-6.]))
        self.l1=node_cls(p,h1,self.lamb_mu,self.lamb_rho,temp=temp,gamma_prior=g1,sigma_0=sigma0)
        self.l2=node_cls(h1,h2,self.lamb_mu,self.lamb_rho,temp=temp,gamma_prior=g2,sigma_0=sigma0)
        self.out=out_cls(h2,1,self.lamb_mu,self.lamb_rho,sigma_0=sigma0); self.kl_global=torch.tensor(0.)
    def forward(self,x):
        s=F.softplus(self.lamb_rho)
        self.kl_global=(-self.lamb_c*math.log(self.lamb_d)+math.lgamma(self.lamb_c)-self.lamb_c*self.lamb_mu+self.lamb_d*torch.exp(self.lamb_mu+s.square()/2)-torch.log(s)-1.41894).sum()
        return self.out(F.relu(self.l2(F.relu(self.l1(x)))))
    def kl(self): return self.l1.kl+self.l2.kl+self.out.kl+self.kl_global


class GHSNet(nn.Module):
    def __init__(self,p,h1,h2,node_cls,out_cls,g1,g2,temp=.5,sigma0=1.,tau0=1.,tau1=1.,c_reg=1.):
        super().__init__(); self.tau0=float(tau0); self.c_reg=float(c_reg)
        self.sig_a_mu=nn.Parameter(torch.tensor([1.])); self.sig_a_rho=nn.Parameter(torch.tensor([-6.]))
        self.sig_b_mu=nn.Parameter(torch.tensor([1.])); self.sig_b_rho=nn.Parameter(torch.tensor([-6.]))
        self.l1=node_cls(p,h1,temp=temp,gamma_prior=g1,sigma_0=sigma0,tau_1=tau1)
        self.l2=node_cls(h1,h2,temp=temp,gamma_prior=g2,sigma_0=sigma0,tau_1=tau1)
        self.out=out_cls(h2,1,sigma_0=sigma0,tau_1=tau1); self.kl_global=torch.tensor(0.)
    def forward(self,x):
        sa=F.softplus(self.sig_a_rho); sb=F.softplus(self.sig_b_rho)
        eps=torch.randn_like(self.sig_a_mu); ss=.5*torch.sqrt(sa.square()+sb.square())
        global_scale=torch.exp(.5*(self.sig_a_mu+self.sig_b_mu)+ss*eps)
        kl_a=-math.log(self.tau0)+torch.exp(self.sig_a_mu+.5*sa.square())/self.tau0-.5*(self.sig_a_mu+2*torch.log(sa)+1.69315)
        kl_b=torch.exp(.5*sb.square()-self.sig_b_mu)-.5*(2*torch.log(sb)-self.sig_b_mu+1.69315)
        self.kl_global=(kl_a+kl_b).sum(); d={0:x,1:global_scale,2:self.c_reg}
        d=self.l1(d); d={0:F.relu(d[0]),1:d[1],2:d[2]}; d=self.l2(d); d={0:F.relu(d[0]),1:d[1],2:d[2]}; return self.out(d)[0]
    def kl(self): return self.l1.kl+self.l2.kl+self.out.kl+self.kl_global


def force_node_mpm(model):
    p1=torch.sigmoid(model.l1.theta).detach(); p2=torch.sigmoid(model.l2.theta).detach()
    a1=p1>0.5; a2=p2>0.5
    with torch.no_grad():
        model.l1.theta.copy_(torch.where(a1, torch.full_like(model.l1.theta,30.), torch.full_like(model.l1.theta,-30.)))
        model.l2.theta.copy_(torch.where(a2, torch.full_like(model.l2.theta,30.), torch.full_like(model.l2.theta,-30.)))
    return a1.cpu().numpy(), a2.cpu().numpy(), p1.cpu().numpy(), p2.cpu().numpy()


def predict_mc(model, X, draws, batch, device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.as_tensor(X[i:i+batch],device=device)
            ys=[]
            for _ in range(draws): ys.append(model(xb).squeeze(-1))
            out.append(torch.stack(ys).mean(0).cpu())
    return torch.cat(out).numpy()


def run(method,args):
    data=load_npz(args.data); device=device_from_arg(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed)
    node_cls,out_cls=import_layers(args.repo_root,method); g1,g2=node_priors(data['ntrain'],data['p'],args.h1,args.h2)
    if method=='ss_gl': model=GLNet(data['p'],args.h1,args.h2,node_cls,out_cls,g1,g2,args.temp,args.sigma0,args.lamb_c,args.lamb_d)
    else: model=GHSNet(data['p'],args.h1,args.h2,node_cls,out_cls,g1,g2,args.temp,args.sigma0,args.tau0,args.tau1,args.c_reg)
    model=model.to(device); ds=TensorDataset(torch.from_numpy(data['Xtr']),torch.from_numpy(data['ytr'][:,None])); gen=torch.Generator().manual_seed(args.seed)
    dl=DataLoader(ds,batch_size=min(args.batch,len(ds)),shuffle=True,generator=gen); opt=torch.optim.Adam(model.parameters(),lr=args.lr); N=len(ds); t0=time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); pred=model(xb); loss=F.mse_loss(pred,yb)+model.kl()/N
            if not torch.isfinite(loss): raise RuntimeError(f'{method} non-finite loss at epoch {ep+1}')
            loss.backward(); opt.step()
        if args.verbose and ((ep+1)%100==0 or ep==0): print(f'{method} epoch={ep+1:04d}')
    runtime=time.perf_counter()-t0; a1,a2,pip1,pip2=force_node_mpm(model); pred=predict_mc(model,data['Xte'],args.draws,args.batch,device)
    candidate=data['p']*args.h1+args.h1*args.h2+args.h2; retained=int(data['p']*a1.sum()+int(a1.sum())*int(a2.sum())+a2.sum())
    selected=np.ones(data['p'],dtype=bool) if a1.any() else np.zeros(data['p'],dtype=bool)
    return {
        'y_pred_test':pred,'selected_features':selected,'retained_weights':retained,'candidate_weights':candidate,'dparam':retained/candidate,
        'native_density':float((a1.sum()+a2.sum())/(args.h1+args.h2)),'runtime_sec':runtime,
        'native_rule':'Posterior node inclusion PIP > 0.5 (official spike-and-slab group prior); connection density induced by retained nodes',
        'note':f'official Jantre layers; node PIPs H1={int(a1.sum())}/{args.h1}, H2={int(a2.sum())}/{args.h2}; node-only method implies all inputs retained whenever H1 is nonempty'
    }


def main(method):
    p=add_common_args(argparse.ArgumentParser()); p.add_argument('--temp',type=float,default=.5); p.add_argument('--sigma0',type=float,default=1.); p.add_argument('--lamb-c',type=float,default=4.); p.add_argument('--lamb-d',type=float,default=2.); p.add_argument('--tau0',type=float,default=1.); p.add_argument('--tau1',type=float,default=1.); p.add_argument('--c-reg',type=float,default=1.)
    a=p.parse_args(); write_json(run(method,a),a.output)

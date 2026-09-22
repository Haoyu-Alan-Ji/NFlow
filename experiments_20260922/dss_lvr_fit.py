from __future__ import annotations
import inspect, random, time, sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from Python import model2 as md
from Python import bnn_metric as metric
from metrics_common import function_metrics, selection_metrics, topk_indices, encode_indices
import experiment_config as cfg


def _load(path,device):
    z=np.load(path,allow_pickle=False)
    tr=z["train_idx"].astype(int); te=z["test_idx"].astype(int)
    X=torch.as_tensor(z["X"],dtype=torch.float32,device=device)
    y=torch.as_tensor(z["y"],dtype=torch.float32,device=device)
    signal=torch.as_tensor(z["signal"],dtype=torch.float32,device=device)
    truth=z["feature_true"].astype(float)
    return X,y,signal,truth,tr,te,str(z["condition"].item()),int(z["seed"].item())


def fit_dataset(data_path, fit_seed=None, device="auto", epochs=None, warmup=None,
                R_train=None, R_eval=None, R_final=None, verbose=True):
    device=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device))
    X,y,signal,truth,tr,te,condition,data_seed=_load(data_path,device)
    fit_seed=int(data_seed+100000+cfg.P+17*len(cfg.HIDDEN_DIMS)) if fit_seed is None else int(fit_seed)
    epochs=cfg.EPOCHS if epochs is None else int(epochs); warmup=cfg.WARMUP_EPOCHS if warmup is None else int(warmup)
    R_train=cfg.R_TRAIN if R_train is None else int(R_train); R_eval=cfg.R_EVAL if R_eval is None else int(R_eval); R_final=cfg.R_FINAL if R_final is None else int(R_final)
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(fit_seed)
    ti=torch.as_tensor(tr,device=device,dtype=torch.long); tei=torch.as_tensor(te,device=device,dtype=torch.long)
    n_diag=max(1,int(round(.10*len(tr)))); di=ti[:n_diag]
    kwargs=dict(
        X=X[ti],y=y[ti],input_dim=cfg.P,hidden_dims=cfg.HIDDEN_DIMS,out_dim=1,
        selection_mode=cfg.SELECTION_MODE,family="gaussian",sigma2=cfg.SIGMA2,init_sd=cfg.INIT_SD,
        K_flow=cfg.K_FLOW,flow_type="iaf",flow_hidden_units=cfg.FLOW_HIDDEN_UNITS,
        flow_hidden_layers=cfg.FLOW_HIDDEN_LAYERS,scale_clip=cfg.SCALE_CLIP,flow_seed=fit_seed+17,
        iaf_ordering_scheme=cfg.IAF_ORDERING,iaf_shuffle_within_role=True,
        gate_type=cfg.GATE_TYPE,gate_scale=cfg.GATE_SCALE,
    )
    sig=inspect.signature(md.GroupedBNNVI.__init__).parameters
    if "slab_init" in sig:
        kwargs.update(slab_init=cfg.SLAB_INIT,slab_sd_ratio=cfg.SLAB_SD_RATIO,slab_bias_sd=cfg.SLAB_BIAS_SD)
    model=md.GroupedBNNVI(**kwargs).to(device)
    g=torch.Generator(device=model.q0.loc.device); g.manual_seed(fit_seed+99173)
    with torch.no_grad():
        model.q0.loc.add_(cfg.INIT_LOC_JITTER*torch.randn(model.q0.loc.shape,generator=g,device=model.q0.loc.device,dtype=model.q0.loc.dtype))
    opt=torch.optim.Adam(model.parameters(),lr=cfg.LR)
    if device.type=="cuda": torch.cuda.synchronize()
    t0=time.perf_counter()
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); is_warm=ep<=warmup
        if is_warm:
            xi,logq=model.sample_posterior(R_train)
            elbo=model.log_likelihood(xi,force_all_on=True)+model.log_prior(xi)-logq
        else:
            elbo=model.elbo_draws(R_train)["elbo"]
        (-elbo.mean()).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.GRAD_CLIP); opt.step()
        if verbose and (ep==1 or ep==warmup or ep==epochs or ep%500==0):
            model.eval()
            with torch.no_grad():
                xe,_=model.sample_posterior(R_eval); pred=model.decoder(X[di],xe,force_all_on=is_warm)
                fm=metric.function_metrics(signal[di],pred)
            print(f"{condition} seed={data_seed} epoch={ep:04d} {'repr' if is_warm else 'select'} R2={float(fm['r2']):.4f}")
    if device.type=="cuda": torch.cuda.synchronize()
    runtime=time.perf_counter()-t0
    model.eval()
    with torch.no_grad():
        xi,_=model.sample_posterior(R_final)
        pred_draw=model.decoder(X[tei],xi)
        pred=pred_draw.mean(0).detach().cpu().numpy()
        f_pip=model.decoder.feature_semantics(xi)["active"].float().mean(0).cpu().numpy()
        u_pip=model.decoder.unit_semantics(xi)["active"].float().mean(0).cpu().numpy()
        d_edge=float(metric.network_density(model.decoder,xi))
        d_path=float(metric.active_path_density(model.decoder,xi))
    fm=function_metrics(signal[tei].cpu().numpy(),pred); sm=selection_metrics(f_pip,truth>0.5,cfg.SUPPORT_THRESHOLD)
    selected_f=f_pip>cfg.SUPPORT_THRESHOLD
    unit_counts=[]; start=0
    for h in cfg.HIDDEN_DIMS:
        unit_counts.append(int(np.sum(u_pip[start:start+h]>cfg.SUPPORT_THRESHOLD))); start+=h
    sf=int(selected_f.sum()); dims=[sf]+unit_counts+[1]; dense=[cfg.P]+list(cfg.HIDDEN_DIMS)+[1]
    retained=sum(dims[i]*dims[i+1] for i in range(len(dims)-1)); candidate=sum(dense[i]*dense[i+1] for i in range(len(dense)-1))
    topk=topk_indices(f_pip,cfg.N_ACTIVE)
    selected_idx=np.flatnonzero(selected_f)
    return {
        "method":"DSS-LVR","condition":condition,"data_seed":data_seed,"fit_seed":fit_seed,
        **fm,**sm,"dparam":float(retained/candidate),"retained_weights":int(retained),"candidate_weights":int(candidate),
        "network_density":d_edge,"path_density":d_path,"runtime_sec":float(runtime),
        "selected_active_units":int(sum(unit_counts)),"selected_indices":encode_indices(selected_idx),
        "topk_indices":encode_indices(topk),
        "feature_pip":f_pip,"unit_pip":u_pip,
    }

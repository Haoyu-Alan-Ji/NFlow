"""Seven one-seed rebuttal experiments; all submitted source modules remain untouched.

RAT uses original framework.simflow_stagewise and original model.py.
Gaussian MCMC executes the user's original run(1).R verbatim.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, gammaln, logsumexp
from sklearn.metrics import roc_auc_score, average_precision_score

from Python import framework as fw, config as cfg
from Python.utils import make_split
from Python.metric import recovery_metrics
from .data import ROOT, Dataset, synthetic, boston, gse32448, splits_for
from .mfvb import fit_mfvb
from .r_bridge import reference

# Manuscript schedule unchanged; pilot is restored exactly from the earlier runner.
# Note: pilot also changes temperature (.5 -> .1), so it is NOT a pure
# shorter-epoch ablation against the fixed-tau=.8 manuscript schedule.
PROFILES = {
    'paper_full': dict(tau_start=0.8, tau_end=0.8,
                       warmup_epochs=200, n_anneal_stages=5,
                       min_stage_epochs=500, max_stage_epochs=1000,
                       eval_every=25, diag_R_train=64, diag_R_final=4000,
                       num_samples=128, base_lr=5e-5,
                       stage_lr_decay=0.7, min_lr_scale=0.2),
    'pilot': dict(tau_start=0.5, tau_end=0.1,
                  warmup_epochs=100, n_anneal_stages=6,
                  min_stage_epochs=40, max_stage_epochs=100,
                  eval_every=25, diag_R_train=64, diag_R_final=600,
                  num_samples=32, base_lr=2e-4,
                  stage_lr_decay=0.9, min_lr_scale=0.5),
}


def verify_untouched_sources():
    hashes = json.loads((ROOT/'ORIGINAL_SOURCE_SHA256.json').read_text(encoding='utf-8'))
    for name, digest in hashes.items():
        current = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if current != digest:
            raise RuntimeError(f'Protected original source changed: {name}; expected SHA256 {digest}')


def set_threads(n):
    if n is not None:
        torch.set_num_threads(int(n))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def canonical_splits(data, split_seed=12345):
    """Use original Python split implementation; group split only for paired GEO.

    At n non-multiple-of-five, fractions are the R original's floor counts,
    so both untouched samplers train on the same observations.
    """
    n = len(data.y)
    counts = (int(.6*n), int(.2*n))
    config = cfg.SplitConfig(counts[0]/n, counts[1]/n, (n-sum(counts))/n, seed=int(split_seed))
    internal = make_split(n, config)
    if data.groups is None:
        desired = internal
    else:
        desired = splits_for(data, seed=int(split_seed))
        if tuple(len(desired[x]) for x in ('train','val')) != counts:
            raise ValueError('Grouped split sizes differ from the original R train/val floor counts. '
                             'Change the data split externally; original sampler will not be patched.')
        group = np.asarray(data.groups)
        sets = [set(group[desired[k]]) for k in ('train','val','test')]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Group leakage in GEO train/val/test split')
    desired_order = np.concatenate([desired[x] for x in ('train','val','test')])
    perm = np.concatenate([internal[x] for x in ('train','val','test')])
    placement = np.empty(n, dtype=int)
    placement[perm] = desired_order
    for k in ('train','val','test'):
        if not np.array_equal(placement[internal[k]], desired[k]):
            raise RuntimeError('Cannot align original Python split without modifying its code.')
    return desired, config, placement


def selection_stats(pip, data, cutoff=.5):
    pip = np.asarray(pip, dtype=float)
    selected = pip > cutoff  # original R writer uses pip > pip_threshold
    out = {'support_size': int(selected.sum()), 'expected_support':float(pip.sum())}
    if data.beta_true is not None:
        active = np.abs(data.beta_true) > 1e-12
        out['TPR'] = float(np.mean(selected[active])) if np.any(active) else np.nan
        out['FPR'] = float(np.mean(selected[~active])) if np.any(~active) else np.nan
        out['AUROC_selection'] = float(roc_auc_score(active, pip)) if np.unique(active).size == 2 else np.nan
    else:
        out['TPR'] = np.nan
        out['FPR'] = np.nan
    if data.noise_start is not None:
        out['noise_FPR'] = float(np.mean(selected[data.noise_start:]))
        out['noise_PIP_mean'] = float(np.mean(pip[data.noise_start:]))
    return out


def predictive(data, desired, B):
    X = np.asarray(data.X[desired['test']],dtype=float)
    y = np.asarray(data.y[desired['test']],dtype=float)
    B = np.asarray(B,dtype=float)
    lls=[]
    pred=[]
    for start in range(0,len(y),256):
        xx = X[start:start+256]
        yy = y[start:start+256]
        eta = xx @ B.T
        if data.family == 'gaussian':
            mu=eta
            ll=-.5*((yy[:,None]-eta)**2/data.sigma2+np.log(2*np.pi*data.sigma2))
        elif data.family=='poisson':
            with np.errstate(over='ignore',invalid='ignore'):
                mu=np.exp(eta)
                ll=yy[:,None]*eta-mu-gammaln(yy[:,None]+1)
        elif data.family=='logistic':
            mu=expit(eta)
            ll=yy[:,None]*(-np.logaddexp(0,-eta)) + (1-yy[:,None])*(-np.logaddexp(0,eta))
        else:
            raise ValueError(data.family)
        lls.append(logsumexp(ll,axis=1)-math.log(B.shape[0]))
        pred.extend(np.mean(mu,axis=1).tolist())
    pred=np.asarray(pred)
    out={'test_nll':-float(np.mean(np.concatenate(lls))),
         'test_mse':float(np.mean((pred-y)**2))}
    if data.family == 'poisson':
        mu=pred
        with np.errstate(divide='ignore',invalid='ignore'):
            terms=np.where(y>0,y*np.log(y/np.maximum(mu,1e-300)),0)
        out['test_poisson_deviance']=float(np.mean(2*(terms-y+mu)))
    if data.family=='logistic' and np.unique(y).size==2:
        out['test_classification_auroc']=float(roc_auc_score(y,pred))
        out['test_classification_auprc']=float(average_precision_score(y,pred))
        out['test_accuracy_0.5']=float(np.mean((pred>=.5)==y))
    return out


def cross_target_diagnostics(B, ref, beta_eps):
    """Descriptive cross-prior diagnostics only; never call posterior recovery."""
    b=np.asarray(B, dtype=float)
    ref_b=np.asarray(ref['beta'],dtype=float)
    pp=(np.abs(b)>beta_eps).mean(axis=0)
    out={'cross_prior_effective_PIP_RMSE':float(np.sqrt(np.mean((pp-ref['pip'])**2))),
         'cross_prior_beta_mean_RMSE':float(np.sqrt(np.mean((b.mean(0)-ref_b.mean(0))**2)))}
    idx=np.argsort(-ref['pip'])[:min(10,b.shape[1])]
    pairs=[(idx[i],idx[j]) for i in range(len(idx)) for j in range(i+1,len(idx))]
    if pairs:
        out['cross_prior_joint_inclusion_MAE']=float(np.mean([
            abs(np.mean((np.abs(b[:,j])>beta_eps)&(np.abs(b[:,k])>beta_eps))-
                np.mean((np.abs(ref_b[:,j])>beta_eps)&(np.abs(ref_b[:,k])>beta_eps)))
            for j,k in pairs]))
    return out


def active_joint_sw2(B, ref_B, active_idx, seed, n_projections=256):
    """Monte Carlo sliced W2 of the FULL joint beta_A posterior (including spikes).

    Uses all VI and reference draws; integrates the empirical 1D W2 exactly even
    when the posterior sample counts differ. Projection directions are fixed by
    the dataset seed and are identical across VI methods for the same dataset.
    Returns SW2, not SW2 squared, on the original coefficient scale.
    """
    active_idx = np.asarray(active_idx, dtype=int)
    if active_idx.size == 0:
        return np.nan
    x = np.asarray(B, dtype=float)[:, active_idx]
    y = np.asarray(ref_B, dtype=float)[:, active_idx]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    rng = np.random.default_rng(int(seed) + 1927)
    directions = rng.standard_normal((active_idx.size, int(n_projections)))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    x = np.sort(x @ directions, axis=0)
    y = np.sort(y @ directions, axis=0)
    boundaries = np.union1d(np.arange(nx + 1) / nx, np.arange(ny + 1) / ny)
    mid = (boundaries[1:] + boundaries[:-1]) / 2
    ix = np.minimum((mid * nx).astype(int), nx - 1)
    iy = np.minimum((mid * ny).astype(int), ny - 1)
    w2_squared = np.sum((boundaries[1:] - boundaries[:-1])[:, None]
                        * (x[ix] - y[iy]) ** 2, axis=0)
    return float(np.sqrt(w2_squared.mean()))


def posterior_distribution_metrics(B, active_draws, data, ref):
    """Original SKL/JS from Python.metric; add only a whole-active-vector SW2."""
    if data.beta_true is None or ref is None:
        return {}
    rec = recovery_metrics(B, active_draws, data.beta_true,
                           {'beta': ref['beta'], 'pip': ref['pip']})
    active_idx = np.flatnonzero(np.abs(data.beta_true) > 1e-12)
    return {
        'cross_prior_active_marginal_SKL': rec['active_marg_skl_partial_median'],
        'cross_prior_zero_JS': rec['zero_js_median'],
        'cross_prior_active_pairwise_SKL': rec['joint_skl_median'],
        'cross_prior_active_joint_SW2': active_joint_sw2(B, ref['beta'], active_idx, data.seed),
    }


def fit_one(data, desired, split_cfg, placement, method, depth, profile, threads, mfvb_epochs, ref,
            beta_eps=0.5, mfvb_slab_var=1.0, mfvb_slab_mode="em", mfvb_restarts=3):
    set_threads(threads)
    started=time.perf_counter()
    if method=='mfvb':
        # New, separately identified conventional MFVB baseline.
        res=fit_mfvb(data.X[desired['train']],data.y[desired['train']],
                     family=data.family,sigma2=data.sigma2,seed=data.seed,epochs=mfvb_epochs,
                     draws=PROFILES[profile]['diag_R_final'],slab_var=mfvb_slab_var,
                     slab_mode=mfvb_slab_mode, restarts=mfvb_restarts)
        B=res['beta_draws']
        A=res['active_draws']
        pip=res['pip']
        elapsed=res['runtime_sec']
        mfvb_meta=dict(mfvb_optimizer=res['optimizer'], mfvb_slab_var=float(res['slab_prior_var']),
                       mfvb_initial_slab_var=float(res['slab_prior_initial_var']),
                       mfvb_slab_mode=res['slab_mode'], mfvb_pi_hat=float(res['pi_hat']),
                       mfvb_restarts=int(res['restarts']), mfvb_em_iterations=int(res['em_iterations']))
    else:
        coupling={'rat':'semantic','odd_even':'affine','thinner':'semantic_affine_control'}.get(method)
        if coupling is None: raise ValueError(method)
        c=PROFILES[profile]
        schedule=cfg.StagewiseAnnealConfig(
            tau_start=c['tau_start'],tau_end=c['tau_end'],warmup_epochs=c['warmup_epochs'],
            n_anneal_stages=c['n_anneal_stages'],min_stage_epochs=c['min_stage_epochs'],
            max_stage_epochs=c['max_stage_epochs'],base_lr=c['base_lr'],
            stage_lr_decay=c['stage_lr_decay'],min_lr_scale=c['min_lr_scale'],eval_every=c['eval_every'],
            print_every=max(50,c['eval_every']),diag_R_train=c['diag_R_train'],
            diag_R_final=c['diag_R_final'],num_samples=c['num_samples'])
        # Real-data beta_true is unknown; 0 only satisfies the ORIGINAL
        # framework's diagnostics interface. It is never used in optimization,
        # truth evaluation, MCMC reference for real data, or outer results.
        technical_beta=np.zeros(data.X.shape[1],dtype='float32') if data.beta_true is None else data.beta_true
        x=torch.as_tensor(np.ascontiguousarray(data.X[placement]),dtype=torch.float32)
        y=torch.as_tensor(np.ascontiguousarray(data.y[placement]),dtype=torch.float32)
        bt=torch.as_tensor(technical_beta,dtype=torch.float32)
        meta=dict(seed=int(data.seed),setting=data.setting,sim=data.setting,
                  n=int(len(data.y)),p=int(data.X.shape[1]),sigma2=float(data.sigma2),
                  n_active=int(np.sum(np.abs(technical_beta)>1e-12)))
        fit=fw.simflow_stagewise(
            X=x,y=y,beta_true=bt,sim_info=meta,seed=int(data.seed),device=torch.device('cpu'),
            family=data.family,beta_mode='sigmoid',coupling_type=coupling,
            conditioner_type='mlp',hidden_units=64,num_hidden_layers=2,
            K_q=64 if int(depth)==68 else int(depth)//2,
            K_g=4 if int(depth)==68 else int(depth)-int(depth)//2,
            schedule_cfg=schedule,split_cfg=split_cfg,
            save_cfg=cfg.SaveConfig(output_dir=None),compare_mcmc=False,
            show_start=False,show_final=False,mcmc_during_training=False)
        B=fit['final']['beta_samples'].detach().cpu().numpy()
        A=fit['final']['active_samples'].detach().cpu().numpy()
        pip=fit['final']['active_samples'].float().mean(dim=0).detach().cpu().numpy()
        elapsed=fit['total_runtime_sec']
        flow_meta=dict(training_epochs=int(fit['history_df']['epoch'].max()),
                       selected_epoch=int(fit['final']['selected_ckpt_meta']['epoch']),
                       selected_tau=float(fit['final']['tau_selected']))
        del fit
    row=dict(dataset=data.setting, family=data.family, seed=data.seed,
             method=method, depth=int(depth) if method!='mfvb' else 0,
             n_coupling_transforms=(5 if method=='thinner' else 3)*int(depth) if method!='mfvb' else 0,
             coefficient_map=('sigmoid_train_indicator_report' if method!='mfvb' else 'pointmass_gaussian'),
             reference_coefficient_map='relu' if ref is not None else '',
             same_posterior_target=False if ref is not None else None,
             n=len(data.y),p=data.X.shape[1],profile=profile,
             wall_seconds=float(elapsed),mcmc_beta_eps=float(beta_eps) if ref else np.nan,
             pip_definition='original_gate_active' if method!='mfvb' else 'mfvb_bernoulli_inclusion',
             **selection_stats(pip,data),**predictive(data,desired,B),
             **(mfvb_meta if method=='mfvb' else flow_meta))
    if ref is not None:
        row.update(cross_target_diagnostics(B,ref,beta_eps))
        row.update(posterior_distribution_metrics(B,A,data,ref))
        row['MCMC_seconds']=ref['wall_seconds']
    print(f'[result] {data.setting} {method} K={depth} time={elapsed:.2f}s '
          f'support={row["support_size"]} test_nll={row["test_nll"]:.3f}')
    return row


def run_dataset(data, methods, *, profile='pilot',mcmc=True,
                mcmc_n=10000,mcmc_burn=2000,mcmc_smax=100,mcmc_thin=1,beta_eps=.5,
                mfvb_epochs=600,mfvb_slab_var=1.0,mfvb_slab_mode='em',mfvb_restarts=3,
                threads=6,rscript='Rscript',output=None):
    verify_untouched_sources()
    if profile not in PROFILES: raise ValueError(profile)
    output = Path(output or ROOT/'results')
    output.mkdir(parents=True,exist_ok=True)
    desired, split_cfg, placement=canonical_splits(data)
    ref=reference(data,desired,output=output,rscript=rscript,n_iter=mcmc_n,burn=mcmc_burn,
                  thin=mcmc_thin,smax=mcmc_smax,beta_eps=beta_eps) if mcmc else None
    rows=[]
    if ref is not None:
        rows.append(dict(dataset=data.setting,family=data.family,seed=data.seed,
                         method='MCMC_original_R_ESS',depth=0,n_coupling_transforms=0,
                         coefficient_map='relu',reference_coefficient_map='relu',
                         same_posterior_target=True,profile='original_R',
                         pip_definition=f'original_abs_beta_gt_{beta_eps}',
                         n=len(data.y),p=data.X.shape[1],wall_seconds=ref['wall_seconds'],
                         mcmc_beta_eps=float(beta_eps),
                         **selection_stats(ref['pip'],data),**predictive(data,desired,ref['beta'])))
    for method,depth in methods:
        print(f'[run] {data.setting} {method} depth={depth} seed={data.seed} profile={profile}')
        rows.append(fit_one(data,desired,split_cfg,placement,method,depth,profile,
                            threads,mfvb_epochs,ref,beta_eps=beta_eps,
                            mfvb_slab_var=mfvb_slab_var,mfvb_slab_mode=mfvb_slab_mode,
                            mfvb_restarts=mfvb_restarts))
    return pd.DataFrame(rows)


def save(df,name,output=None):
    # Keep existing paper_full filenames; never overwrite them with pilot CSVs.
    if 'profile' in df and (df['profile'] == 'pilot').any():
        name = str(Path(name).with_name(Path(name).stem + '_pilot.csv'))
    path=Path(output or ROOT/'results')/name
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)
    print('[saved]',path)
    return df


def run_depth(seed=400, depths=(4,8,16,32,64,68), setting='baseline', **kw):
    data=synthetic(seed,setting)
    return save(run_dataset(data,[('rat',int(k)) for k in depths],**kw),f'depth_seed{seed}.csv',kw.get('output'))

def run_mfvb(seed=400,setting='baseline',family='gaussian',depth=68,**kw):
    data=synthetic(seed,setting,family=family)
    return save(run_dataset(data,[('mfvb',0),('rat',depth)],**kw),f'mfvb_{family}_{setting}_seed{seed}.csv',kw.get('output'))

def run_difficult(seed=400,settings=('weak_signal','low_snr','p_gt_n'),depth=68,**kw):
    methods=[('mfvb',0),('rat',depth),('odd_even',depth),('thinner',depth)]
    parts=[run_dataset(synthetic(seed,s),methods,**kw) for s in settings]
    return save(pd.concat(parts,ignore_index=True),f'difficult_4methods_seed{seed}.csv',kw.get('output'))

def run_correlation(seed=400,rhos=(.25,.5,.75),depth=68,**kw):
    parts=[run_dataset(synthetic(seed,'baseline',rho=r),[('mfvb',0),('rat',depth)],**kw) for r in rhos]
    return save(pd.concat(parts,ignore_index=True),f'correlation_seed{seed}.csv',kw.get('output'))

def run_poisson(seed=400,settings=('baseline','weak_signal','low_snr','n_gt_p','p_gt_n'),depth=68,**kw):
    parts=[run_dataset(synthetic(seed,s,family='poisson'),[('mfvb',0),('rat',depth)],**kw) for s in settings]
    return save(pd.concat(parts,ignore_index=True),f'poisson_seed{seed}.csv',kw.get('output'))

def run_boston(seed=400,data_dir=None,n_noise=12,rho_noise=.25,**kw):
    data=boston(seed,data_dir,n_noise,rho_noise,rscript=kw.get('rscript','Rscript'))
    return save(run_dataset(data,[('mfvb',0),('rat',68)],**kw),f'boston_seed{seed}.csv',kw.get('output'))

def run_gse(seed=400,data_dir=None,n_genes=40,n_noise=30,outcome='tumor',rho_noise=.25,**kw):
    data=gse32448(seed,data_dir,n_genes,n_noise,rho_noise,rscript=kw.get('rscript','Rscript'),outcome=outcome)
    return save(run_dataset(data,[('mfvb',0),('rat',68)],**kw),f'gse32448_{outcome}_seed{seed}.csv',kw.get('output'))

def run_task(task,**kwargs):
    handlers=dict(depth=run_depth,mfvb=run_mfvb,difficult=run_difficult,
                  correlation=run_correlation,poisson=run_poisson,boston=run_boston,gse=run_gse)
    return handlers[task](**kwargs)

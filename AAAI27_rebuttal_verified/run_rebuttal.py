"""Run one local task at a time; RAT / Gaussian ESS use untouched originals."""
import os
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('MKL_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
import argparse
from rebuttal27 import run_task

p=argparse.ArgumentParser()
p.add_argument('task',choices=['depth','mfvb','difficult','correlation','poisson','boston','gse'])
p.add_argument('--seed',type=int,default=400)
p.add_argument('--profile',choices=['pilot','paper_full'],default='pilot')
p.add_argument('--no-mcmc',action='store_true')
p.add_argument('--mcmc-n',type=int,default=10000)
p.add_argument('--mcmc-burn',type=int,default=2000)
p.add_argument('--mcmc-smax',type=int,default=100)
p.add_argument('--mcmc-thin',type=int,default=1)
p.add_argument('--beta-eps',type=float,default=.5)
p.add_argument('--threads',type=int,default=6)
p.add_argument('--mfvb-epochs',type=int,default=600)
p.add_argument('--mfvb-slab-var',type=float,default=1.0)
p.add_argument('--mfvb-slab-mode',choices=['em','fixed'],default='em')
p.add_argument('--mfvb-restarts',type=int,default=3)
p.add_argument('--rscript',default='Rscript')
p.add_argument('--output',default=None)
p.add_argument('--setting',default='baseline')
p.add_argument('--depth',type=int,default=68)
p.add_argument('--depths',type=int,nargs='+',default=[4,8,16,32,64,68])
p.add_argument('--settings',nargs='+',default=None)
p.add_argument('--rhos',type=float,nargs='+',default=[.25,.5,.75])
p.add_argument('--data-dir',default=None)
p.add_argument('--n-genes',type=int,default=40)
p.add_argument('--n-noise',type=int,default=None)
p.add_argument('--outcome',choices=['tumor','grade'],default='tumor')
a=p.parse_args()
kw=dict(seed=a.seed,profile=a.profile,mcmc=not a.no_mcmc,mcmc_n=a.mcmc_n,
        mcmc_burn=a.mcmc_burn,mcmc_smax=a.mcmc_smax,mcmc_thin=a.mcmc_thin,
        beta_eps=a.beta_eps,threads=a.threads,mfvb_epochs=a.mfvb_epochs,
        mfvb_slab_var=a.mfvb_slab_var,mfvb_slab_mode=a.mfvb_slab_mode,
        mfvb_restarts=a.mfvb_restarts,
        rscript=a.rscript,output=a.output)
if a.task=='depth': kw.update(depths=a.depths,setting=a.setting)
if a.task=='mfvb': kw.update(setting=a.setting,depth=a.depth)
if a.task in ('difficult','poisson') and a.settings is not None: kw['settings']=a.settings
if a.task in ('difficult','poisson'): kw.update(depth=a.depth)
if a.task=='correlation': kw.update(rhos=a.rhos,depth=a.depth)
if a.task in ('boston','gse'):
    kw.update(data_dir=a.data_dir)
    if a.n_noise is not None: kw.update(n_noise=a.n_noise)
if a.task=='gse': kw.update(n_genes=a.n_genes,outcome=a.outcome)
run_task(a.task,**kw)

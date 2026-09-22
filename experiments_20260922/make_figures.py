#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import experiment_config as cfg
from metrics_common import decode_indices


def table1_frequency_figure(raw: pd.DataFrame, out: Path):
    conds=list(cfg.TABLE1_CONDITIONS)
    freq=np.zeros((len(conds),cfg.P),dtype=float)
    for i,c in enumerate(conds):
        rows=raw.loc[raw.condition==c]
        if len(rows)==0:
            continue
        for val in rows.selected_indices.fillna(''):
            idx=decode_indices(val)
            if idx is not None and len(idx):
                freq[i,np.asarray(idx,dtype=int)] += 1
        freq[i] /= len(rows)
    fig,ax=plt.subplots(figsize=(11,3.2))
    im=ax.imshow(freq,aspect='auto',vmin=0,vmax=1,interpolation='nearest')
    ax.axvline(cfg.N_ACTIVE-0.5,linestyle='--',linewidth=1)
    ax.set_yticks(np.arange(len(conds)),[cfg.TABLE1_LABELS[c] for c in conds])
    ax.set_xlabel('Predictor index (first 10 are active)')
    ax.set_ylabel('Simulation condition')
    cb=fig.colorbar(im,ax=ax)
    cb.set_label('MPM selection frequency')
    fig.tight_layout()
    fig.savefig(out,dpi=300,bbox_inches='tight')
    plt.close(fig)


def table2_tradeoff_figure(summary: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for _,r in summary.iterrows():
        if pd.isna(r.get('mse')) or pd.isna(r.get('dparam')):
            continue
        ax.scatter(r.dparam,r.mse,s=45)
        ax.annotate(str(r.method),(r.dparam,r.mse),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel(r'$D_{\mathrm{param}}$')
    ax.set_ylabel(r'Function MSE')
    ax.set_title('Function recovery vs. hard parameter density')
    fig.tight_layout()
    fig.savefig(out,dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='results'); a=ap.parse_args()
    root=Path(a.root); figdir=root/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    f1=root/'raw'/'table1_runs.csv'
    if f1.exists():
        raw=pd.read_csv(f1)
        if 'selected_indices' in raw:
            table1_frequency_figure(raw,figdir/'table1_selection_frequency.png')
    f2=root/'summary'/'table2_summary.csv'
    if f2.exists():
        table2_tradeoff_figure(pd.read_csv(f2),figdir/'table2_mse_dparam.png')

if __name__=='__main__':
    main()

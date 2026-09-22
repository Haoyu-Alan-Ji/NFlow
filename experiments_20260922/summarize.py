#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import experiment_config as cfg
from metrics_common import summarize_numeric,pairwise_kuncheva

def fmt(m,lo,hi,d=3):
    if pd.isna(m): return '--'
    if pd.isna(lo): return f'{m:.{d}f}'
    return f'{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='results'); a=p.parse_args(); root=Path(a.root); (root/'summary').mkdir(parents=True,exist_ok=True)
    f1=root/'raw'/'table1_runs.csv'
    if f1.exists():
        raw=pd.read_csv(f1); cols=['tpr','fpr','accuracy','auroc','auprc','selected_support']; sm=summarize_numeric(raw,'condition',cols); sm['kuncheva']=[pairwise_kuncheva(raw.loc[raw.condition==c,'topk_indices'],cfg.P,cfg.N_ACTIVE) for c in sm.condition]; sm.insert(1,'label',sm.condition.map(cfg.TABLE1_LABELS)); sm.to_csv(root/'summary'/'table1_summary.csv',index=False)
        lines=['\\begin{table}[t]','\\centering','\\small','\\caption{Predictor selection across nonlinear simulation conditions. MSE and $R_f^2$ are intentionally omitted; all settings use $n=5000$, $p=200$, and $s=10$.}','\\label{tab:selection_conditions}','\\begin{tabular}{lccccccc}','\\toprule','Condition & TPR & FPR & Acc. & AUROC & AUPRC & $|\\widehat S|$ & Kuncheva \\\\','\\midrule']
        for _,r in sm.iterrows(): lines.append(f"{r['label']} & {r.tpr:.3f} & {r.fpr:.3f} & {r.accuracy:.3f} & {r.auroc:.3f} & {r.auprc:.3f} & {r.selected_support:.2f} & {r.kuncheva:.3f} \\\\")
        lines += ['\\bottomrule','\\end{tabular}','\\end{table}']; (root/'summary'/'table1.tex').write_text('\n'.join(lines),encoding='utf-8')
    parts=[]
    for nm in ['table2_dss.csv','table2_ss.csv','table2_r.csv']:
        q=root/'raw'/nm
        if q.exists(): parts.append(pd.read_csv(q))
    if parts:
        raw=pd.concat(parts,ignore_index=True); raw.to_csv(root/'raw'/'table2_all.csv',index=False); cols=['mse','r2','dparam','network_density','path_density','runtime_sec']; sm=summarize_numeric(raw,'method',cols); sm['kuncheva']=np.nan
        if 'topk_indices' in raw:
            for i,m in enumerate(sm.method):
                vals=raw.loc[(raw.method==m)&raw.topk_indices.notna(),'topk_indices']; sm.loc[i,'kuncheva']=pairwise_kuncheva(vals,cfg.P,cfg.N_ACTIVE) if len(vals)>1 else np.nan
        order={m:i for i,m in enumerate(cfg.TABLE2_METHODS)}; sm['_o']=sm.method.map(order).fillna(999); sm=sm.sort_values('_o').drop(columns='_o'); sm.to_csv(root/'summary'/'table2_summary.csv',index=False)
        lines=['\\begin{table*}[t]','\\centering','\\scriptsize','\\setlength{\\tabcolsep}{4pt}','\\caption{Function recovery and structural sparsity on the fixed trigonometric-interaction benchmark ($n=5000$, $p=200$, $s=10$). Lower MSE, $D_{\\mathrm{param}}$, $D_E$, and $D_\\pi$ are better; higher $R_f^2$ is better.}','\\label{tab:function_sparsity}','\\begin{tabular}{lccccccc}','\\toprule','Method & MSE$_f$ & $R_f^2$ & Kuncheva & $D_{\\mathrm{param}}$ & $D_E$ & $D_\\pi$ & Time (s) \\\\','\\midrule']
        for _,r in sm.iterrows():
            ku='--' if pd.isna(r.kuncheva) else f'{r.kuncheva:.3f}'; lines.append(f"{r.method} & {r.mse:.3f} & {r.r2:.3f} & {ku} & {r.dparam:.4f} & {r.network_density:.4f} & {r.path_density:.4f} & {r.runtime_sec:.1f} \\\\")
        lines += ['\\bottomrule','\\end{tabular}','\\end{table*}']; (root/'summary'/'table2.tex').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()

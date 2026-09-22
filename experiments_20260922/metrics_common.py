from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import t as student_t


def function_metrics(signal, pred):
    signal=np.asarray(signal,float).reshape(-1); pred=np.asarray(pred,float).reshape(-1)
    mse=float(np.mean((pred-signal)**2))
    den=float(np.sum((signal-signal.mean())**2))
    r2=float(1.0-np.sum((pred-signal)**2)/max(den,1e-12))
    return {"mse":mse,"r2":r2}


def selection_metrics(pip, truth, threshold=.5):
    pip=np.asarray(pip,float); truth=np.asarray(truth,bool)
    sel=pip>float(threshold)
    tp=np.sum(sel & truth); fp=np.sum(sel & ~truth); fn=np.sum(~sel & truth); tn=np.sum(~sel & ~truth)
    out={
        "tpr": float(tp/max(tp+fn,1)),
        "fpr": float(fp/max(fp+tn,1)),
        "accuracy": float((tp+tn)/len(truth)),
        "selected_support": int(sel.sum()),
        "expected_support": float(pip.sum()),
    }
    if np.unique(truth.astype(int)).size==2:
        out["auroc"]=float(roc_auc_score(truth.astype(int),pip))
        out["auprc"]=float(average_precision_score(truth.astype(int),pip))
    else:
        out["auroc"]=out["auprc"]=np.nan
    return out


def topk_indices(scores,k):
    scores=np.asarray(scores,float)
    return np.argsort(-scores,kind="stable")[:int(k)].astype(int)


def encode_indices(idx):
    return ";".join(map(str,np.asarray(idx,int).tolist()))


def decode_indices(x):
    if x is None or (isinstance(x,float) and np.isnan(x)) or str(x).strip()=="": return None
    return np.asarray([int(z) for z in str(x).split(";") if z!=""],dtype=int)


def kuncheva_pair(a,b,p,k):
    a=set(map(int,a)); b=set(map(int,b)); k=int(k); p=int(p)
    if k<=0 or k>=p: return np.nan
    r=len(a.intersection(b))
    return float((r*p-k*k)/(k*(p-k)))


def pairwise_kuncheva(encoded_sets,p,k):
    sets=[decode_indices(x) for x in encoded_sets]
    sets=[x for x in sets if x is not None and len(x)==k]
    vals=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            vals.append(kuncheva_pair(sets[i],sets[j],p,k))
    return float(np.mean(vals)) if vals else np.nan


def mean_ci(x, level=.95):
    a=np.asarray(pd.Series(x).dropna(),float)
    if len(a)==0: return np.nan,np.nan,np.nan
    m=float(a.mean())
    if len(a)==1: return m,np.nan,np.nan
    se=float(a.std(ddof=1)/math.sqrt(len(a)))
    q=float(student_t.ppf((1+level)/2,df=len(a)-1))
    return m,m-q*se,m+q*se


def summarize_numeric(df, by, cols):
    rows=[]
    for key,g in df.groupby(by,sort=False):
        row={by:key,"n_rep":len(g)}
        for col in cols:
            m,lo,hi=mean_ci(g[col]); row[col]=m; row[col+"_lo"]=lo; row[col+"_hi"]=hi
        rows.append(row)
    return pd.DataFrame(rows)

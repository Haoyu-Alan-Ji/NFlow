from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from Python.simfun import simfun1
from Python.utils import make_split
from Python.config import SplitConfig

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    beta_true: np.ndarray | None
    family: str
    setting: str
    seed: int
    sigma2: float = 1.0
    feature_names: list[str] | None = None
    noise_start: int | None = None
    groups: np.ndarray | None = None
    info: dict | None = None


def ar1(n, p, rho, rng):
    # Generates iid rows from a multivariate normal with Sigma_jk=rho^|j-k|.
    z = rng.standard_normal((n, p))
    if rho == 0:
        return z
    x = z.copy()
    x[:, 0] = z[:, 0]
    a = np.sqrt(1.0 - rho * rho)
    for j in range(1, p):
        x[:, j] = rho * x[:, j - 1] + a * z[:, j]
    return x


def dimensions(setting):
    return {
        'baseline': (160, 100), 'weak_signal': (160, 100),
        'low_snr': (160, 100), 'n_gt_p': (600, 100),
        'p_gt_n': (100, 500),
    }[setting]


def synthetic(seed=400, setting='baseline', family='gaussian', rho=0.0, n=None, p=None):
    n0, p0 = dimensions(setting)
    n, p = int(n or n0), int(p or p0)
    weak = setting == 'weak_signal'
    low_snr = setting == 'low_snr'
    beta_low, beta_high = (0.1, 0.5) if weak else (0.3, 2.0)
    sigma2 = 10.0 if low_snr else 1.0
    X, y0, bt, _ = simfun1(sim='simple', n=n, p=p, seed=int(seed), n_active=10,
                             sigma2=sigma2, beta_low=beta_low, beta_high=beta_high,
                             center_y=True, device=torch.device('cpu'))
    X = X.cpu().numpy().astype(np.float64)
    bt = bt.cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(int(seed) + 991)
    if rho:
        X = ar1(n, p, float(rho), rng)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)

    if family == 'gaussian':
        y = y0.cpu().numpy().astype(np.float64) if not rho else X @ bt + rng.normal(0, np.sqrt(sigma2), n)
        if rho:
            y = y - y.mean()
    elif family == 'poisson':
        # Use moderate rate range; a large Poisson rate, not large n, yields Gaussian-looking counts.
        target = 0.35 if weak else 0.23 if low_snr else 0.70
        lin = X @ bt
        scale = min(target / (lin.std() + 1e-12), 1.7 / (np.max(np.abs(lin)) + 1e-12))
        bt *= scale
        eta = X @ bt
        y = rng.poisson(np.exp(eta)).astype(np.float64)
        sigma2 = 1.0  # placeholder: never used in a Poisson likelihood
    elif family in {'bernoulli', 'logistic'}:
        lin = X @ bt
        bt *= min(1.0, 1.0 / (lin.std() + 1e-12))
        eta = X @ bt
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(np.float64)
        sigma2 = 1.0
        family = 'logistic'
    else:
        raise ValueError(f'Unsupported family: {family}')

    info = {'rho': float(rho), 'setting': setting, 'seed': int(seed)}
    if family == 'poisson':
        info.update(rate_min=float(np.exp(eta).min()), rate_max=float(np.exp(eta).max()),
                    rate_median=float(np.median(np.exp(eta))), zero_fraction=float(np.mean(y == 0)))
    return Dataset(X.astype('float32'), y.astype('float32'), bt.astype('float32'), family,
                   setting if not rho else f'{setting}_rho{rho:g}', int(seed),
                   sigma2=sigma2, feature_names=[f'x{j + 1}' for j in range(p)], info=info)


def splits_for(data: Dataset, seed=12345):
    n = len(data.y)
    if data.groups is None:
        return make_split(n, SplitConfig(train_frac=.60, val_frac=.20, test_frac=.20, seed=int(seed)))
    outer = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=int(seed))
    pool, test = next(outer.split(np.arange(n), data.y, data.groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=int(seed) + 1)
    train0, val0 = next(inner.split(pool, data.y[pool], data.groups[pool]))
    return {'train': pool[train0], 'val': pool[val0], 'test': test}


def _run_r(script, folder, rscript='Rscript'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([rscript, str(ROOT / 'r' / script), str(folder)],
                          capture_output=True, text=True, errors='replace')
    if proc.returncode:
        raise RuntimeError(f'R data export failed: {script}\n{proc.stderr[-6000:]}\n{proc.stdout[-2000:]}')
    if proc.stdout:
        print(proc.stdout.strip())


def boston(seed=400, data_dir=None, n_noise=12, rho_noise=.25, rscript='Rscript', drop_sensitive=False):
    folder = Path(data_dir or ROOT / 'external' / 'Boston')
    filename = folder / 'boston.csv'
    if not filename.exists():
        _run_r('export_boston.R', folder, rscript=rscript)
    df = pd.read_csv(filename)
    if 'medv' not in df.columns:
        raise ValueError('Boston CSV requires a medv outcome column')
    if drop_sensitive and 'black' in df.columns:
        df = df.drop(columns=['black'])
    y = df.pop('medv').to_numpy(dtype='float64')
    return _real(data=df, y=y, family='gaussian', setting=f'boston_noise{n_noise}', seed=seed,
                 n_noise=n_noise, rho_noise=rho_noise)


def gse32448(seed=400, data_dir=None, n_genes=40, n_noise=30, rho_noise=.25,
              rscript='Rscript', outcome='tumor'):
    folder = Path(data_dir or ROOT / 'external' / 'GSE32448')
    matrix = folder / 'expression.csv'
    meta_file = folder / 'metadata.csv'
    if not matrix.exists() or not meta_file.exists():
        _run_r('export_gse32448.R', folder, rscript=rscript)
    df = pd.read_csv(matrix)
    pheno = pd.read_csv(meta_file)
    if 'sample_id' not in pheno or 'sample_id' not in df:
        raise ValueError('Export requires sample_id in expression.csv and metadata.csv')
    pheno = pheno.set_index('sample_id')
    df = df.set_index('sample_id')
    if set(df.index) != set(pheno.index):
        raise ValueError('Expression and phenotype sample IDs differ')
    pheno = pheno.loc[df.index]
    titles = pheno['title'].fillna('').astype(str)
    # Series includes 40 matched tumor-normal samples from 40 subjects; avoid sample-wise leakage.
    tissue = titles.str.extract(r'(?:^|[_\s-])(T|N|B)(?=[_\s-]|$)', expand=False).str.upper()
    grade = titles.str.extract(r'(?:^|[_\s-])(PD|WD)(?=[_\s-]|$)', expand=False).str.upper()
    if outcome == 'tumor':
        y = (tissue == 'T').astype('float64')
        valid = tissue.isin(['T', 'N', 'B'])
    elif outcome == 'grade':
        y = (grade == 'PD').astype('float64')
        valid = grade.isin(['PD', 'WD']) & (tissue == 'T')
    else:
        raise ValueError('outcome must be tumor or grade')
    if not valid.any() or len(y[valid].unique()) != 2:
        raise ValueError('Could not infer the two classes from GEO metadata; inspect metadata.csv title')
    titles = titles[valid]
    df = df.loc[valid.values]
    y = y[valid].to_numpy(dtype='float64')
    groups = titles.str.extract(r'([0-9]+)\s*$', expand=False)
    if groups.isna().any():
        raise ValueError('Patient pairing cannot be recovered from sample titles; add patient_id to metadata.csv')
    # PD/WD patient number suffixes are used as within-cohort patient IDs.
    groups = (grade[valid].fillna('unknown') + '_' + groups).to_numpy()
    raw = df.apply(pd.to_numeric, errors='coerce')
    if raw.isna().any().any():
        raw = raw.fillna(raw.median(axis=0))
    X = raw.to_numpy(dtype='float64')
    if np.nanmedian(X) > 20:
        X = np.log2(np.maximum(X, 0) + 1.0)
    cols = list(raw.columns)
    if len(cols) < n_genes:
        raise ValueError('Requested n_genes exceeds available columns')
    # Choose features using only TRAINING variance (no label-based feature selection).
    preliminary = Dataset(X.astype('float32'), y.astype('float32'), None, 'logistic',
                          'gse32448', int(seed), groups=groups)
    index = splits_for(preliminary)
    variances = X[index['train']].var(axis=0)
    top = np.argsort(-variances, kind='stable')[:int(n_genes)]
    selected = pd.DataFrame(X[:, top], columns=[cols[j] for j in top])
    obj = _real(selected, y, 'logistic', 'gse32448', seed, n_noise, rho_noise, groups=groups)
    obj.setting = f'gse32448_{outcome}_genes{n_genes}_noise{n_noise}'
    obj.info['outcome'] = outcome
    obj.info['grouped_by_patient'] = True
    return obj


def _real(data, y, family, setting, seed, n_noise, rho_noise, groups=None):
    X = data.to_numpy(dtype='float64')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite real-data values')
    base = Dataset(X.astype('float32'), np.asarray(y, dtype='float32'), None, family,
                   setting, int(seed), groups=groups)
    idx = splits_for(base)
    mu, sd = X[idx['train']].mean(0), X[idx['train']].std(0)
    X = (X - mu) / np.where(sd < 1e-8, 1.0, sd)
    y = np.asarray(y, dtype='float64')
    if family == 'gaussian':
        y_mu, y_sd = y[idx['train']].mean(), y[idx['train']].std()
        y = (y - y_mu) / max(y_sd, 1e-8)
    rng = np.random.default_rng(int(seed) + 8801)
    U = ar1(len(y), int(n_noise), float(rho_noise), rng)
    U = (U - U[idx['train']].mean(0)) / (U[idx['train']].std(0) + 1e-12)
    feature_names = list(data.columns) + [f'negative_control_{j + 1}' for j in range(n_noise)]
    return Dataset(np.column_stack([X, U]).astype('float32'), y.astype('float32'), None,
                   family, setting, int(seed), 1., feature_names=feature_names,
                   noise_start=X.shape[1], groups=groups,
                   info={'n_original': int(X.shape[1]), 'n_noise': int(n_noise),
                         'rho_noise': float(rho_noise), 'n_patients': None if groups is None else int(len(np.unique(groups)))})


def save_training_csv(data: Dataset, splits, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ix = np.asarray(splits['train'])
    frame = pd.DataFrame(data.X[ix], columns=[f'x{j}' for j in range(data.X.shape[1])])
    frame.insert(0, 'y', data.y[ix])
    frame.to_csv(folder / 'mcmc_training.csv', index=False)
    return folder / 'mcmc_training.csv'

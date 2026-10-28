"""External adapter for the user's untouched run(1).R.

Gaussian: executes the byte-identical original source.
Poisson/logistic: copies its text and changes ONLY the Gaussian likelihood
expression in L.exa; ESS.Gibbs, initialization, PIP, and CSV writer are identical.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'r' / 'run_fast_ESS_original.R'
GAUSS_LIK = '''    sum(dnorm(
      x = Y,
      mean = as.numeric(X %*% theta),
      sd = sigma,
      log = TRUE
    ))'''
NEW_LIK = {
    'poisson': '''    eta <- as.numeric(X %*% theta)
    sum(Y * eta - exp(eta) - lgamma(Y + 1))''',
    'logistic': '''    eta <- as.numeric(X %*% theta)
    sum(Y * eta - pmax(eta, 0) - log1p(exp(-abs(eta))))''',
}


def r_executable(rscript):
    p = shutil.which(str(rscript))
    if p is None and not Path(str(rscript)).is_file():
        raise FileNotFoundError(
            f'Cannot locate Rscript: {rscript}. Install R or set RSCRIPT to its full executable path. '
            'To run only RAT-Flow/MFVB without the reference, use mcmc=False. '
            'No replacement MCMC algorithm will be used.'
        )
    return p or str(rscript)


def r_permutation(n, seed, rscript):
    """Call R's sample.int under the exact seed expression used in run(1).R."""
    code = f'set.seed(as.integer({int(seed)})); cat(paste(sample.int({int(n)}), collapse=","))'
    proc = subprocess.run([rscript, '--vanilla', '-e', code], check=True, text=True, capture_output=True)
    perm = np.fromstring(proc.stdout.strip(), sep=',', dtype=int) - 1
    if perm.size != n or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f'Original-R split permutation extraction failed: {proc.stdout!r}')
    return perm


def reference(data, desired, *, output, rscript='Rscript', n_iter=10000,
              burn=2000, thin=1, smax=100, beta_eps=0.5, split_seed=12345):
    """Execute untouched ESS.Gibbs + untouched PIP writer with aligned train rows."""
    if data.family not in ('gaussian', 'poisson', 'logistic'):
        raise ValueError(data.family)
    rscript = r_executable(rscript)
    cache_setting = data.setting if data.family == 'gaussian' else f'{data.family}_{data.setting}'
    folder = Path(output).resolve() / '_mcmc_original_R' / cache_setting / f'seed_{data.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    n, p = data.X.shape
    n_train, n_val = int(.6*n), int(.2*n)
    if any(len(desired[x]) != c for x, c in [('train', n_train), ('val', n_val), ('test', n-n_train-n_val)]):
        raise ValueError('Canonical data partition does not match original R 60/20/20 floor split.')

    rperm = r_permutation(n, split_seed, rscript)
    desired_order = np.concatenate([desired['train'], desired['val'], desired['test']])
    arrangement = np.empty(n, dtype=int)
    arrangement[rperm] = desired_order
    assert np.array_equal(arrangement[rperm[:n_train]], desired['train'])
    X = data.X[arrangement]
    y = data.y[arrangement]
    b = np.zeros(p, dtype=float) if data.beta_true is None else data.beta_true
    dataset = folder / 'r_input.csv'
    beta_path = folder / 'r_beta.csv'
    pd.DataFrame(np.column_stack([y, X]), columns=['y']+[f'x{j+1}' for j in range(p)]).to_csv(dataset, index=False)
    pd.DataFrame({'beta_true': b}).to_csv(beta_path, index=False)
    manifest = folder / 'manifest.csv'
    pd.DataFrame([dict(data_path=dataset.as_posix(), beta_path=beta_path.as_posix(),
                       out_dir=folder.as_posix(), setting=data.setting,
                       seed=int(data.seed), sigma2=float(data.sigma2))]).to_csv(manifest,index=False)
    fingerprint = hashlib.sha256(dataset.read_bytes()+beta_path.read_bytes()+SOURCE.read_bytes()).hexdigest()
    meta = folder / 'reference_protocol.json'
    setup = dict(fingerprint=fingerprint, n_iter=int(n_iter), burn=int(burn), thin=int(thin),
                 smax=int(smax), beta_eps=float(beta_eps), split_seed=int(split_seed),
                 family=data.family, sampler='original run(1).R ESS.Gibbs',
                 initialization='original: b0=beta_true; W=1; W0=0',
                 selection='original: abs(beta)>beta_eps; pip>0.5')
    # Recompute ESS for every experiment invocation; never reuse a previous reference.
    # R overwrites this run's three output tables. A failed R call raises before reading.
    script = SOURCE
    if data.family != 'gaussian':
        source = SOURCE.read_bytes()
        # Preserve the complete original source and replace only the likelihood.
        # The original file is never written to.
        line_end = b'\r\n' if b'\r\n' in source else b'\n'
        pattern = GAUSS_LIK.encode().replace(b'\n', line_end)
        replacement = NEW_LIK[data.family].encode().replace(b'\n', line_end)
        if source.count(pattern) != 1:
            raise RuntimeError('Expected unique Gaussian likelihood block not found in untouched R file.')
        script = folder / f'run_original_ESS_{data.family}_likelihood_extension.R'
        script.write_bytes(source.replace(pattern, replacement, 1))
    cmd = [rscript, str(script), str(manifest), '1', str(int(n_iter)),
           str(int(smax)), str(int(burn)), str(int(thin)),
           str(float(beta_eps)), str(int(split_seed))]
    print('[MCMC] fresh original R ESS:', data.family, f'N={n_iter}', f'burn={burn}', f'beta_eps={beta_eps}')
    started = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT)
    elapsed = time.perf_counter() - started
    setup['wall_seconds'] = elapsed
    meta.write_text(json.dumps(setup, indent=2)+'\n', encoding='utf-8')
    B = pd.read_csv(folder/'mcmc_beta_draws.csv.gz').drop(columns=['draw_id'],errors='ignore').to_numpy(float)
    P = pd.read_csv(folder/'mcmc_pip.csv').sort_values('j0')['pip'].to_numpy(float)
    if B.shape[1] != p or P.size != p:
        raise ValueError('Original ESS posterior output dimensions do not match dataset.')
    # Verify exact original PIP definition, not gate inclusion.
    if not np.allclose((np.abs(B)>beta_eps).mean(0),P,atol=5e-6):
        raise ValueError('Original R PIP does not match abs(beta)>beta_eps.')
    return dict(beta=B, pip=P, wall_seconds=elapsed, map='relu', beta_eps=beta_eps,
                source='original_R_ESS' if data.family=='gaussian' else 'original_R_ESS_new_likelihood',
                path=str(folder))

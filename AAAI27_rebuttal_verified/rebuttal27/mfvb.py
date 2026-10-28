"""Traditional Bernoulli point-mass + Gaussian slab MFVB.

Gaussian regression uses paired mean-field coordinate updates and variational EM
(point-estimated pi and optionally slab variance). For other likelihoods the
previous Adam implementation is retained, with its original Beta(1,1) prior.
No latent threshold, RAT flow, or MCMC updates are used here.
"""
from __future__ import annotations
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _fit_mfvb_adam(X, y, family='gaussian', sigma2=1., epochs=600, lr=.035,
             seed=400, draws=1000, print_every=0, slab_var=1.0):
    if slab_var <= 0:
        raise ValueError('slab_var must be positive')
    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device('cpu')
    X = torch.as_tensor(X, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1)
    n, p = X.shape
    rlogits = torch.nn.Parameter(torch.full((p,), -2.4, dtype=torch.float64))
    mean = torch.nn.Parameter(torch.zeros(p, dtype=torch.float64))
    logsd = torch.nn.Parameter(torch.full((p,), -1.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([rlogits, mean, logsd], lr=lr)
    t0 = time.perf_counter()
    best, best_params = float('inf'), None
    x2 = X.square()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        probs = torch.sigmoid(rlogits)
        sd2 = torch.exp(2.0 * logsd.clamp(-6, 2))
        mu = X @ (probs * mean)
        var = x2 @ (probs * (sd2 + mean.square()) - (probs * mean).square())
        if family == 'gaussian':
            ell = -.5 * (((y - mu).square() + var).sum() / sigma2 + n * math.log(2 * math.pi * sigma2))
        elif family == 'poisson':
            # Independence yields an EXACT Poisson likelihood expectation under q.
            # E exp(sum X_j z_j beta_j)=product_j (1-r_j + r_j exp(X_j m_j+.5 X_j^2 v_j)).
            tmp = X * mean + .5 * x2 * sd2
            logmgf = torch.logaddexp(F.logsigmoid(-rlogits), F.logsigmoid(rlogits) + tmp).sum(1)
            ell = (y * mu - torch.exp(logmgf) - torch.lgamma(y + 1.)).sum()
        elif family in {'logistic', 'bernoulli'}:
            # Rigorous Jaakkola-Jordan variational lower bound, NOT an exact logistic expectation.
            xi = (mu.square() + var).clamp_min(1e-12).sqrt().detach()
            lam = (torch.sigmoid(xi) - .5) / (2. * xi)
            ell = (F.logsigmoid(xi) + ( (2 * y - 1) * mu - xi) / 2. - lam * (mu.square() + var - xi.square())).sum()
        else:
            raise ValueError(family)
        # Exact q(pi) optimal Beta(1+sum r,1+p-sum r) integrated analytically.
        nr = probs.sum()
        beta_term = torch.lgamma(1 + nr) + torch.lgamma(1 + p - nr) - torch.lgamma(torch.as_tensor(2 + p, dtype=torch.float64))
        entropy = -(probs * F.logsigmoid(rlogits) + (1 - probs) * F.logsigmoid(-rlogits)).sum()
        kl_slab = .5 * ((mean.square() + sd2) / slab_var - 1 + math.log(slab_var) - torch.log(sd2))
        objective = -(ell + beta_term + entropy - (probs * kl_slab).sum())
        if not torch.isfinite(objective):
            raise FloatingPointError(f'MFVB nonfinite objective at epoch {epoch}; lower lr or response scale')
        objective.backward()
        torch.nn.utils.clip_grad_norm_([rlogits, mean, logsd], 50.)
        optimizer.step()
        loss = float(objective.detach())
        if loss < best:
            best = loss
            best_params = (rlogits.detach().clone(), mean.detach().clone(), logsd.detach().clone())
        if print_every and (epoch + 1) % print_every == 0:
            print(f'[mfvb] epoch {epoch+1}/{epochs} neg_bound={loss:.4f} Esize={float(probs.sum()):.2f}')
    with torch.no_grad():
        rl, m, ls = best_params
        probs = torch.sigmoid(rl).cpu().numpy()
        means = m.cpu().numpy()
        std = torch.exp(ls).cpu().numpy()
    rng = np.random.default_rng(int(seed) + 10713)
    z = rng.random((draws, p)) < probs
    beta = z * rng.normal(means, std, size=(draws, p))
    return {'method': 'traditional_spike_slab_mfvb', 'family': family,
            'beta_draws': beta.astype(np.float32), 'active_draws': z.astype(np.float32),
            'pip': probs, 'mean_active': means, 'sd_active': std,
            'runtime_sec': time.perf_counter() - t0, 'neg_elbo_bound': best,
            'slab_prior_var': slab_var, 'slab_prior_match': 'submitted_indicator_slab' if slab_var == 1.0 else 'custom_gaussian_slab',
            'logistic_bound': 'Jaakkola-Jordan' if family in {'logistic', 'bernoulli'} else 'exact expectation'}


def _fit_mfvb_vem(X, y, sigma2=1., epochs=600, seed=400, draws=1000,
                  slab_var=1., slab_mode='em', restarts=3, print_every=0):
    """Gaussian paired mean-field CAVI + empirical-Bayes variational EM.

    E-step (one coordinate at a time): exact Bernoulli-Gaussian CAVI updates.
    M-step: empirical-Bayes pi and optionally active Gaussian slab variance.
    No Gaussian VI, RAT, or MCMC source files are changed.
    """
    from scipy.special import expit
    if sigma2 <= 0 or slab_var <= 0:
        raise ValueError('sigma2 and slab_var must be positive')
    if slab_mode not in {'em', 'fixed'}:
        raise ValueError('slab_mode must be em or fixed')
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = X.shape
    if y.size != n:
        raise ValueError('X/y dimension mismatch')
    col2 = np.einsum('ij,ij->j', X, X)
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    max_sweeps = max(1, int(epochs))
    max_em = max(1, min(30, (max_sweeps + 9) // 10))
    sweeps_per_em = max(1, (max_sweeps + max_em - 1) // max_em)
    best_run = None
    best_elbo = -np.inf

    def elbo(phi, mu, s2, pi, slab_v, residual):
        eb = phi * mu
        vb = phi * (s2 + mu * mu) - eb * eb
        ll = -.5 * (n * np.log(2 * np.pi * sigma2)
                    + (residual @ residual + col2 @ vb) / sigma2)
        pc = np.clip(phi, 1e-12, 1 - 1e-12)
        inclusion = np.sum(phi * np.log(pi) + (1 - phi) * np.log1p(-pi)
                           - pc * np.log(pc) - (1 - pc) * np.log1p(-pc))
        kl = .5 * ((mu * mu + s2) / slab_v - 1 + np.log(slab_v / s2))
        return float(ll + inclusion - phi @ kl)

    for restart in range(max(1, int(restarts))):
        pi = .1 if restart == 0 else float(rng.uniform(.03, .3))
        slab_v = float(slab_var)
        phi = np.full(p, pi) if restart == 0 else rng.uniform(.01, min(.8, 2*pi), p)
        mu = np.zeros(p) if restart == 0 else rng.normal(0, .01, p)
        s2 = 1 / (col2 / sigma2 + 1 / slab_v)
        residual = y - X @ (phi * mu)
        prev = -np.inf
        used_sweeps = 0
        for em in range(max_em):
            for _ in range(sweeps_per_em):
                if used_sweeps >= max_sweeps:
                    break
                used_sweeps += 1
                max_delta = 0.
                for j in range(p):
                    old = phi[j] * mu[j]
                    sj2 = 1 / (col2[j] / sigma2 + 1 / slab_v)
                    mj = sj2 * (X[:, j] @ residual + col2[j] * old) / sigma2
                    logit = np.log(pi / (1 - pi)) + .5 * (np.log(sj2 / slab_v) + mj * mj / sj2)
                    pj = float(expit(np.clip(logit, -700., 700.)))
                    new = pj * mj
                    residual += X[:, j] * (old - new)
                    max_delta = max(max_delta, abs(new - old))
                    phi[j], mu[j], s2[j] = pj, mj, sj2
                if max_delta < 1e-7:
                    break
            # Empirical Bayes M-step. Uniform Beta(1,1) prior gives the same MAP pi update.
            pi = float(np.clip(phi.mean(), 1e-9, 1 - 1e-9))
            if slab_mode == 'em':
                slab_v = float(max(np.dot(phi, mu * mu + s2) / max(phi.sum(), 1e-12), 1e-9))
            score = elbo(phi, mu, s2, pi, slab_v, residual)
            if print_every and ((em + 1) % print_every == 0):
                print(f'[MFVB CAVI/EM] restart={restart+1} iter={em+1} ELBO={score:.5f} pi={pi:.5f} slab_var={slab_v:.5f}')
            if np.isfinite(prev) and abs(score - prev) <= 1e-7 * (1 + abs(score)):
                break
            prev = score
        # Finish E-step at final hyperparameters: variances must reflect last M-step.
        for _ in range(10):
            max_delta = 0.
            for j in range(p):
                old = phi[j] * mu[j]
                sj2 = 1 / (col2[j] / sigma2 + 1 / slab_v)
                mj = sj2 * (X[:, j] @ residual + col2[j] * old) / sigma2
                logit = np.log(pi / (1 - pi)) + .5 * (np.log(sj2 / slab_v) + mj * mj / sj2)
                pj = float(expit(np.clip(logit, -700., 700.)))
                new = pj * mj
                residual += X[:, j] * (old - new)
                max_delta = max(max_delta, abs(new - old))
                phi[j], mu[j], s2[j] = pj, mj, sj2
            if max_delta < 1e-7:
                break
        score = elbo(phi, mu, s2, pi, slab_v, residual)
        if score > best_elbo:
            best_elbo = score
            best_run = (phi.copy(), mu.copy(), s2.copy(), pi, slab_v, em+1, used_sweeps)
    phi, mu, s2, pi, slab_v, em_iters, cavi_sweeps = best_run
    rng = np.random.default_rng(int(seed) + 10713)
    z = rng.random((draws, p)) < phi
    beta = z * rng.normal(mu, np.sqrt(s2), size=(draws, p))
    return {'method': 'traditional_spike_slab_mfvb', 'family': 'gaussian',
            'beta_draws': beta.astype(np.float32), 'active_draws': z.astype(np.float32),
            'pip': phi, 'mean_active': mu, 'sd_active': np.sqrt(s2),
            'runtime_sec': time.perf_counter() - t0, 'neg_elbo_bound': -best_elbo,
            'slab_prior_var': slab_v, 'slab_prior_initial_var': slab_var,
            'slab_prior_match': 'empirical_bayes_learned' if slab_mode == 'em' else 'fixed_specified',
            'pi_hat': pi, 'optimizer': 'gaussian_CAVI_variational_EM',
            'em_iterations': em_iters, 'cavi_sweeps': cavi_sweeps,
            'restarts': max(1, int(restarts)), 'slab_mode': slab_mode,
            'logistic_bound': 'not_applicable'}


def fit_mfvb(X, y, family='gaussian', sigma2=1., epochs=600, lr=.035,
             seed=400, draws=1000, print_every=0, slab_var=1.0,
             slab_mode='em', restarts=3):
    """Gaussian: CAVI/variational EM; non-Gaussian: unchanged Adam MFVB.

    slab_var initializes EM or is the pre-specified variance when slab_mode=fixed.
    Non-Gaussian models retain the original implementation; no claim of EM for them.
    """
    if family == 'gaussian':
        return _fit_mfvb_vem(X, y, sigma2=sigma2, epochs=epochs, seed=seed,
                             draws=draws, slab_var=slab_var, slab_mode=slab_mode,
                             restarts=restarts, print_every=print_every)
    if slab_mode == 'em':
        # No analytic non-Gaussian coordinate E-step in the original module.
        slab_mode = 'fixed'
    out = _fit_mfvb_adam(X, y, family=family, sigma2=sigma2, epochs=epochs,
                         lr=lr, seed=seed, draws=draws, print_every=print_every,
                         slab_var=slab_var)
    out.update(optimizer='non_gaussian_adam_original', pi_hat=np.nan,
               slab_prior_initial_var=slab_var, slab_mode=slab_mode,
               em_iterations=0, cavi_sweeps=0, restarts=1)
    return out

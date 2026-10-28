# AAAI-27 Rebuttal - paper_full and restored pilot

This project preserves the original submitted flow and MCMC algorithms. The 12 protected Python/R source files are byte-identical to the original uploads and verified by `ORIGINAL_SOURCE_SHA256.json`. Gaussian ESS executes the original `r/run_fast_ESS_original.R`; Poisson/logistic use copies with only the likelihood substituted, as before. The EM-MFVB implementation and four posterior metrics added in the prior revision are retained.

Unzip alongside your NFlow project so that this package is at `NFlow/AAAI27_rebuttal_verified/`. Use `rebuttal_local_tasks.ipynb` from this folder. If overwriting an older `AAAI27_rebuttal_verified` folder, existing `results/` and `external/` directories need not be deleted. Restart the Jupyter kernel after replacing modules to prevent stale imports.

## Two VI profiles

`profile="paper_full"` uses the manuscript Table 2 configuration; `profile="pilot"` restores the **exact** previous fast exploratory schedule. `smoke` and the older invented `full` remain absent. Both use the unmodified `Python.config.StagewiseAnnealConfig` and training framework. The table below describes `paper_full`:

| Option | Value |
|---|---|
| Flow default | 68 semantic cycles (`K_q=64`, `K_g=4`); 204 transforms |
| Conditioner | MLP 2/64 |
| Warm-up | 200 epochs |
| Post-warm-up | 5 stages; each 500-1000 epochs, with original plateau advancement |
| Base LR | `5e-5` |
| Stage LR multiplier / floor | `0.7` / `0.2` |
| Relaxation temperature (fixed) | `tau_start=0.8`, `tau_end=0.8` (same tau in warm-up and all five stages) |
| Checkpoint interval, diagnostic draws | 25 epochs, 64 draws |
| Final posterior draws | 4000 |
| ELBO draws per step | 128 (retained from the prior `full` runner; Table 2 did not list this option) |
| Inclusion threshold | 0.5 |
| Train seed / split seed | simulation seed / 12345 |

The restored `pilot` uses warm-up 100, six stages of 40-100 epochs, 32 ELBO samples, 600 final draws, learning-rate multiplier 0.9 and floor 0.5. **Importantly, its temperature changes from 0.5 to 0.1; it is not a fixed-temperature training-length-only ablation against `paper_full`**. Pilot and paper_full outputs are saved separately: `*_pilot.csv` versus the previous full filenames. VI result rows now also record actual trained epochs, selected checkpoint epoch, and selected tau.

Flow-depth ablation deliberately includes other values of K, and **every** depth uses the selected profile's schedule. At the default K=68, the runner explicitly passes `K_q=64`, `K_g=4`; at other requested depths it retains the original runner's half-and-half decomposition. Both are interpreted by the protected model as their sum K.

MCMC iteration/burn-in counts are **not** part of manuscript Table 2, which describes VI optimization. Existing MCMC settings are retained independently (Notebook uses N=10000, burn-in=2000, `beta_eps=0.05`, matching the user's recent runs); change these explicitly only if the intended reference protocol requires it. The original R ESS sampler, initialization (`b0=beta_true` in simulations), `abs(beta)>beta_eps` PIP calculation, and checkpoint/stage-training algorithms are unchanged.

The notebook sets EM-MFVB slab initialization to 0.5 (`mfvb_slab_mode="em"`, three restarts). That is an **initial variance**, not a fixed final slab variance. To fix at 0.5, explicitly use `mfvb_slab_mode="fixed"`, but do not change it based on relative performance.

## Four posterior diagnostics

`cross_prior_active_marginal_SKL`, `cross_prior_zero_JS`, `cross_prior_active_pairwise_SKL`, and `cross_prior_active_joint_SW2`. The first three use the *original* `Python.metric.recovery_metrics`. As requested, the marginal/pairwise SKL output is the median over **successfully estimated** components or pairs; `valid/total` columns are not saved. If none can be estimated, the result remains `NaN`. This conditional median can obscure numerical failures and should be interpreted with care. SW2 remains unchanged (256 common deterministic random projections across the full active vector). These are cross-prior discrepancy measures relative to the original ReLU-MCMC and cannot be described as strict same-target posterior recovery.

## Non-Gaussian error fixes

- **Poisson:** the ESS cache now uses `results/_mcmc_original_R/poisson_<setting>/seed_<seed>`; Gaussian caches keep their previous paths. No cache deletion or source-code changes are needed. `reference_protocol.json` still checks data and parameters before reuse.
- **GSE32448:** the R exporter calls `GEOquery::getGEO` with `returnType="ExpressionSet"` **only if the installed version supports it**. Older GEOquery releases therefore retain their native ExpressionSet behavior. The Python adapter now propagates R stderr/stdout if a GEO download or dependency fails.

To run GSE, the local R installation still needs `GEOquery` and `Biobase` from Bioconductor, a working network connection to GEO, and a valid `RSCRIPT` path. R execution could not be tested in the current container (Rscript not installed); it must be tested on Windows. The GEO expression data are microarray intensities; the tumor/benign task remains logistic, not Poisson.

## Running

Set `TASK` and `PROFILE` (`pilot` default, or `paper_full`) at the top of `rebuttal_local_tasks.ipynb` to one of `depth`, `mfvb`, `difficult`, `correlation`, `poisson`, `boston`, `gse`. The notebook calls `runner.run_*()` directly, so `%autoreload 2` refreshes the latest module. One task saves one CSV in `results/`; required R ESS outputs/caches are written separately to `results/_mcmc_original_R/`.

Command-line examples:

```powershell
python run_rebuttal.py mfvb --seed 400 --depth 68 --profile paper_full --mfvb-slab-var 0.5 --mfvb-slab-mode em --rscript "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" --beta-eps 0.05
python run_rebuttal.py difficult --seed 400 --depth 68 --profile paper_full --settings weak_signal p_gt_n --mfvb-slab-var 0.5 --rscript "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" --beta-eps 0.05
python run_rebuttal.py poisson --seed 400 --depth 68 --profile paper_full --settings baseline weak_signal p_gt_n --rscript "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
```

The Rscript path above is only an example; adjust to your local installation. Validation tested profile parameter forwarding and filename separation without launching expensive training. Full end-to-end quality and any efficiency-equivalence claim require completed independent runs.

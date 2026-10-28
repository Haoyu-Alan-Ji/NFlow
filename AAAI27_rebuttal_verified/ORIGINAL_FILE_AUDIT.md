# Original-code preservation audit

Every file below is byte-for-byte copied from the files uploaded by the user **before any of the assistant's rebuttal modifications**. `ORIGINAL_SOURCE_SHA256.json` gives the SHA256 digest of every protected file; `runner.verify_untouched_sources()` enforces these hashes at runtime.

| Bundled path | Original uploaded name | Algorithmic status |
|---|---|---|
| `Python/artifact.py` | `artifact(5).py` | byte-identical |
| `Python/checkpoint_rule.py` | `checkpoint_rule(1).py` | byte-identical |
| `Python/config.py` | `config(3).py` | byte-identical |
| `Python/framework.py` | `framework(2).py` | byte-identical |
| `Python/make_data.py` | `make_data(1).py` | byte-identical |
| `Python/metric.py` | `metric(10).py` | byte-identical |
| `Python/model.py` | `model(5).py` | byte-identical |
| `Python/run.py` | `run(2).py` | byte-identical |
| `Python/simfun.py` | `simfun(20261028-050339).py` | byte-identical |
| `Python/utils.py` | `utils(6).py` | byte-identical |
| `r/run_fast_ESS_original.R` | `run(1).R` | byte-identical; **called for Gaussian** |
| `r/run_slow_MH_original.R` | `run_mh_hard_dss(3).R` | byte-identical; not called |

**What is NEW, separately identified:**
- `rebuttal27/runner.py`: task/data/metric wrapper around original `simflow_stagewise`.
- `rebuttal27/r_bridge.py`: invokes exact original Gaussian `run(1).R`, aligns row ordering; for Poisson/logistic, writes a derived R script with only its likelihood block changed. All original ESS lines, original initialization and original PIP lines remain identical.
- `rebuttal27/data.py`: extra simulation environments and Boston/GEO preparation.
- `rebuttal27/mfvb.py`: independent new point-mass Gaussian slab baseline; **not** a port of the original RAT-Flow method.
- `r/export_boston.R`, `r/export_gse32448.R`: R-to-CSV data export, never change source datasets.
- `run_rebuttal.py` and `rebuttal_local_tasks.ipynb`: orchestration.

**Explicitly NOT done:** rewriting ESS sampler, switching to Python MCMC, changing MCMC PIP event, changing beta_true initialization, changing RAT sigmoid to Softplus or ReLU, altering RAT prior/ELBO/checkpoint function, or introducing source monkeypatches.

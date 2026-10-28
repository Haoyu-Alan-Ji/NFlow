# MCMC rerun policy

- Each `runner.run_*()` dataset invocation now runs a fresh original R ESS reference.
- Within one invocation (for example `run_depth(depths=(4,8,16,32))`), all RAT depths use that freshly computed reference, rather than rerunning ESS for each depth.
- Existing `_mcmc_original_R/` tables and metadata are overwritten by the new run. Python reads the new MCMC draws only after R exits successfully; if R fails, output files may be partially written, but a subsequent invocation will rerun rather than reuse them. Source code, simulation, partition alignment, priors, PIP definition and original `run_fast_ESS_original.R` are unchanged.
- `mcmc=False` still skips ESS entirely.
- Do not start concurrent calls for the same family/setting/seed/output path; they write to the same filenames.
- Identical seed, dataset and R settings yield the same ESS draws in a reproducible R environment. Recalculation removes cache ambiguity, but does not reconcile different posterior targets (ReLU vs RAT's hard-indicator map).

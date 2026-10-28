# Pilot restoration (minimal outer-runner patch)

Only three executable/notebook files changed relative to AAAI27_rebuttal_paper_full_fixed_tau08.zip (plus documentation updates):
- rebuttal27/runner.py: restored exactly the earlier pilot profile; runner default is pilot, with paper_full preserved; pilot CSVs receive `_pilot` before `.csv` so existing full results are not overwritten; new rows record `training_epochs`, `selected_epoch` and `selected_tau` from the original returned fit object.
- run_rebuttal.py: allows --profile pilot (default) or paper_full.
- rebuttal_local_tasks.ipynb: defaults to PROFILE="pilot" and clarifies profile difference.

Original pilot: tau 0.5->0.1, warmup 100, six stages min40/max100, 32 ELBO draws, 600 final posterior draws, base_lr 5e-5, stage_lr_decay .9, min_lr_scale .5, eval_every 25, diag_R_train 64.
Original paper_full: fixed tau .8, warmup 200, five stages min500/max1000, 128 ELBO draws, 4000 final posterior draws, base_lr 5e-5, stage_lr_decay .7, min_lr_scale .2.

**These profiles differ in temperature and sample budget, not just iterations**. Comparing the profiles is an end-to-end configuration comparison, not a causal experiment on training length. A matched-temperature duration ablation would require pilot-style budgets with tau kept at .8, predeclared independently of outcomes.

The protected original Python and R source files, MCMC procedure, MFVB implementation, posterior metrics, default K=68 paper_full decomposition and dataset code remain byte-identical to the immediately preceding package. Existing paper_full CSVs retain their names. Existing cached ESS draws are reused when the MCMC arguments are unchanged.

When reporting the study: if an actual run used K=16 but the manuscript says K=68, correct the implementation detail in the revised manuscript. Pilot exploratory results must not be presented as if they were a controlled training-length comparison.

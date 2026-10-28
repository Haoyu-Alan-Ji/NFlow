> Historical changelog for the paper_full-only revision. For the present package with pilot restored, see `PILOT_RESTORATION_NOTES.md`.

# Changes against the last EM-MFVB + four-metric runner

Changed files only (all protected original sources unchanged):

- `rebuttal27/runner.py`: retain the prior EM MFVB and posterior metrics; remove `smoke/pilot/full`; add the single `paper_full` schedule from the manuscript's Table 2; use Kq=64, Kg=4 at the default K=68; other K only for an explicit depth ablation; replace strict marginal SKL completeness with the original `active_marg_skl_partial_median`; remove all four `valid/total` output columns. Pairwise SKL already took the median over valid pairs.
- `rebuttal27/r_bridge.py`: disambiguate non-Gaussian reference-cache paths by family; do not alter original R ESS code, PIP, initialization or Gaussian cache paths.
- `r/export_gse32448.R`: conditionally pass `returnType` to GEOquery only when the installed version supports it.
- `rebuttal27/data.py`: show actual R stderr/stdout on data-export failures instead of only Python's `CalledProcessError`.
- `rebuttal_local_tasks.ipynb`: paper_full only, K=68 defaults, notebook path fix, direct reloadable runner imports.
- `run_rebuttal.py`: CLI only supports `--profile paper_full`, defaults to K=68, and exposes existing MFVB slab settings.
- `README.md`: updated usage and limitations.

No protected original `Python/*.py` file, original `r/run_fast_ESS_original.R`, original MH R file, or `rebuttal27/mfvb.py` was changed against the last supplied EM implementation.

**Fixed-temperature correction:** the manuscript used fixed `tau=0.8` at every training stage. The external runner now sets `tau_start=tau_end=0.8` without changing the original stagewise trainer. This is the only functional change relative to the previous ZIP. The ELBO batch size `num_samples=128` remains as before.

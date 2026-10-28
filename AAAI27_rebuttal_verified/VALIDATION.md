# Paper-full patch validation

- All 12 protected submitted original sources verified byte-identical against `ORIGINAL_SOURCE_SHA256.json`.
- Python source files compile and notebook parses; two flow profiles (`paper_full` and restored `pilot`).
- Paper default K=68 is passed as Kq=64/Kg=4 with original sigmoid mode; 5 post-warm-up stages with 500-1000 epochs and 200 warm-up epochs, unchanged original stagewise code.
- Marginal SKL uses the original partial median (successful KDE estimates); pairwise SKL uses the original successful-pair median; all success-count columns removed.
- Non-Gaussian reference-cache isolation and R-script likelihood-splicing paths verified by mock-interface tests; actual ESS code unchanged.
- GEOquery compatibility branch statically checked. R/GEO network execution has not been verified here because Rscript is unavailable.
- `mfvb.py` is byte-identical to the EM MFVB version supplied in the previous turn.
- Fixed `tau=0.8` in warm-up and all five training stages by setting `tau_start=tau_end=0.8` in the external runner; no protected trainer changes.

- Restored `pilot` and unchanged `paper_full` verified through a non-training mocked-fit interface test; validated `_pilot` output suffix and new checkpoint-epoch columns. The actual pilot runs were not launched in this container.

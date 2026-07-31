# RaT-BNN direct unit isolation

Place the four files in `Python/` and replace the existing `Python/model2.py`
with the included version. Open `benchmark_direct_bnn.ipynb` from the same
notebook directory as the existing BNN benchmark.

The direct model is

$$
f(x)=\beta_0+\ell x+\sum_{j=1}^H a_j\operatorname{ReLU}(w_jx-b_j).
$$

`beta0` and `ell` are continuous and never gated. The learned roles are
controlled by `gate_roles`:

| Experiment | `gate_roles` | `init_sd` |
|---|---|---:|
| A | `()` | 0.5 |
| B1 | `("input", "breakpoint", "output")` | 0.082 |
| B2 | `("input", "breakpoint", "output")` | 0.25 |
| B3 | `("input", "breakpoint", "output")` | 0.5 |
| B4 | `("input", "breakpoint", "output")` | 1.0 |

For the role interventions, omit exactly one role to fix that gate open:

- input fixed: `("breakpoint", "output")`
- output fixed: `("input", "breakpoint")`
- breakpoint fixed: `("input", "output")`

The updated `model2.py` adds `init_sd` to the original residual
`LaSTBNNVI`, so experiment C uses `init_sd=0.5` with the existing residual
configuration.

`train_direct_bnn()` returns:

- `history`: ELBO decomposition, validation function metrics, path count,
  continuous-path magnitude, and cancellation at every checkpoint;
- `role_history`: role-wise margin, PIP, gate, magnitude, and gradients;
- `unit_history`: breakpoint-ordered role PIPs and joint path dependence;
- `contribution_history`: breakpoint-ordered unit energy;
- `final`: `R_final` draws, test metrics, MCMC comparison, conditional slab
  metrics, and final diagnostic tables.

Checkpoint selection uses only validation-grid `signal_r2`. The implementation
contains no gate warm-up, KL annealing, or mode-seeking modification.
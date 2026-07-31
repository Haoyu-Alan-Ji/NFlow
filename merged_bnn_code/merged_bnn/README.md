# Unified direct-unit and edge-level RaT-BNN code

Copy the four files under `Python/` into the existing project package:

- `model2.py`: existing residual decoder and RaT VI, direct-unit decoder and
  VI, plus mixed sigmoid/RePU edge gates.
- `bnn_mcmc.py`: one MCMC implementation for both parameterizations. It calls
  `model.log_likelihood()` directly, so MCMC and VI share the decoder and
  likelihood code.
- `metric.py`: the existing metrics plus direct role/path/energy/cancellation,
  function recovery, spike/slab, breakpoint ordering, and residual effective
  path metrics.
- `bnn_train.py`: complete `train_direct_bnn()` and `train_edge_bnn()` entry
  points with checkpoint histories.

The old `direct_bnn.py`, `direct_train.py`, `direct_metric.py`, and
`edge_metric.py` are no longer imported.

## Tests

Run the notebooks from the same notebook directory used by the original
benchmark:

1. `benchmark_direct_bnn.ipynb` repeats the current all-gated direct-unit test
   with `R_train=100`, `R_eval=1000`, and `R_final=5000`; only the module
   imports/calls have changed.
2. `benchmark_edge_sigmoid_bnn.ipynb` returns to the one-block residual model.
   `E` and `Wout` use sigmoid gates; `W1_0`, `b1_0`, `W2_0`, and `b2_0` retain
   normalized-RePU gates; attention remains off.

For the mixed edge model,

```python
sigmoid_params=("E", "Wout")
sigmoid_tau=1.0
```

The sigmoid remains continuous throughout training and MCMC. The optional
`sigmoid_zero_threshold=0.05` is applied only when reporting whether a
sigmoid-controlled edge is effectively disconnected. `Pr(gate > 0.5)` remains
a margin-sign diagnostic, not exact spike mass.

To include the input/output biases in the sigmoid experiment, change the tuple
to `("E", "e", "Wout", "bout")`; the supplied benchmark intentionally changes
only the two matrices that control the complete nonlinear path.

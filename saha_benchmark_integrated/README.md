# Saha nonlinear BNN sparsity benchmark

This bundle runs the same Saha–Liu–Liang nonlinear regression DGP through DSS-LVR and the selected sparse-BNN comparators. It saves only the generated DGP and final result tables; no epoch histories or checkpoints are kept by the benchmark wrappers.

## Methods

- DSS-LVR
- IS-ANN-L1
- LBBNN-LRT
- LBBNN-FLOW
- ISLaB-FLOW
- SS-GL
- SS-GHS
- Laplace-SpaM
- wsBNN

The external Python adapters reuse the official GitHub implementations. Only their hard-coded dataset/architecture drivers are replaced so that every method sees the same Saha split and, where applicable, the common `p -> 20 -> 10 -> 1` regression architecture.

## 1. Clone the official repositories

From this bundle directory:

```bash
python benchmark_adapters/setup_external_repos.py --root external_methods
```

For Laplace-SpaM, install the repository's bundled modified `laplace-torch` library and dependencies:

```bash
python benchmark_adapters/setup_external_repos.py --root external_methods --install-spam
```

This creates:

```text
external_methods/
  SS_Group_Shrinkage_New/
  spam-pruning/
  wsBNN/
```

## 2. Install R methods

```r
install.packages(c("torch", "jsonlite", "LBBNN"))
torch::install_torch()
```

`Rscript saha_r_methods.R ...` provides a standalone R runner for LBBNN-LRT, LBBNN-FLOW, ISLaB-FLOW and IS-ANN-L1. The Python master runner can also invoke the LBBNN package directly when `Rscript` is available.

## 3. Smoke tests

Built-in methods:

```bash
python saha_sparsity_benchmark.py --methods dss_lvr,is_ann_l1 --dss-epochs 5 --dss-warmup 2 --is-ann-epochs 5
```

SS-GL / SS-GHS:

```bash
python saha_sparsity_benchmark.py --methods ss_gl,ss_ghs --ss-epochs 5 --ss-draws 5
```

wsBNN:

```bash
python saha_sparsity_benchmark.py --methods wsbnn --ws-epochs 5 --ws-mc-train 1 --ws-draws 5
```

Laplace-SpaM:

```bash
python saha_sparsity_benchmark.py --methods laplace_spam --spam-epochs 5 --spam-burnin 5 --spam-hypersteps 1 --spam-frequency 5
```

## 4. Full single-seed run

```bash
python saha_sparsity_benchmark.py --project-root . --external-root external_methods --seed 400
```

Default Saha DGP: `n=2000`, `p=100`, `pi=0.2`, `alpha=2`, `sigma2=1`, 80/20 split and two hidden layers 20/10.

Outputs:

```text
results_saha_sparsity/
  saha_dgp_seed_400.npz
  saha_sparsity_results_seed_400.csv
```

## Unified metrics

Every method is reduced to the same final columns where the quantity is meaningful:

- noise-free signal MSE and R2
- noisy-response MSE/RMSE
- feature TPR, FPR, selection accuracy and selected support
- `Dparam`: retained connection weights / candidate connection weights, biases excluded
- runtime
- native sparsification rule

DSS-LVR additionally reports its posterior induced-edge density `D_E` and active-path density `D_pi` because these are method-specific posterior structural summaries rather than universal pruning counts.

### Native rules

- DSS-LVR: MPM of hard `V > tau` states, then induced edges.
- LBBNN/ISLaB: native PIP > 0.5 MPM and active-path pruning.
- IS-ANN-L1: native `|w| >= 0.005`, followed by removal of disconnected paths.
- SS-GL/SS-GHS: posterior node inclusion PIP > 0.5 using the official group-prior layers. These methods do not perform predictor selection; if any first-layer node remains, all input predictors remain structurally connected.
- wsBNN: official shared feature-PIP ranking with the top-10 feature rule used by its simulation code. The downstream network remains dense, so its `Dparam` quantifies how much feature selection alone compresses the network.
- Laplace-SpaM: official OPD score `posterior_precision * weight^2`. Because SpaM defines a pruning path rather than one intrinsic cutoff, the adapter selects the sparsest candidate pruning level whose internal-validation MSE is within 1% of the dense model; test data are never used to select the pruning level.

## Standalone R runner

First export a generated NPZ file:

```bash
python export_saha_for_r.py results_saha_sparsity/saha_dgp_seed_400.npz
```

Then:

```bash
Rscript saha_r_methods.R --data-dir results_saha_sparsity/saha_dgp_seed_400_r --out results_saha_sparsity/saha_r_results_seed_400.csv --device cpu
```

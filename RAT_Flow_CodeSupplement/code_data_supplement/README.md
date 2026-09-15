# Code and Data Supplement

## Scope

This compact supplement records the computational path used for the simulation study in *Role-Aware Threshold Flow: Flow-Based Variational Inference for Bayesian Variable Selection*. It contains pseudocode and three selected tables sufficient to identify the data-generating settings, inspect the reported posterior/support summaries, and audit a representative multi-chain MH run. Raw observations, posterior draws, checkpoints, and logs are not included. No external repository is required or linked.

## Included tables

- `experiment_manifest.csv` lists the five simulation environments. Each row specifies the sample size, dimension, number of active predictors, noise variance, signal range, seeds, and train/validation/test split.
- `posterior_support_summary.csv` reports the selected Appendix C.1 results. Each row is an environment--configuration pair; `N` is the number of completed seeds, and every metric is followed by its 95% bootstrap confidence limits. Lower values are better for `D_SKL_A`, `D_JS_0`, and `RMSE_PIP`; higher values are better for `AUROC` and `AUPRC`.
- `mh_chain_summary.csv` gives per-chain diagnostics for one representative eight-chain MH run (baseline environment, simulation seed 400). It is included to document chain length, acceptance, adapted proposal scales, and runtime; it is not a cross-seed performance table.

The tables follow the workflow below: use the manifest to identify an environment, use the posterior/support table to locate its aggregated results, and use the MH chain table only as an implementation and timing check for the disclosed sampler.

## Function map

| File | Main function | Role |
|---|---|---|
| `simfun.py` | `simfun1()` | Generate the standardized Gaussian design, sparse coefficients, and Gaussian response. |
| `make_data.py` | `make_dataset()`, `export_one()` | Apply an environment setting and write data, truth, and manifest records. |
| `run.py` | `read_job()`, `read_data()`, `make_schedule()`, `main()` | Read one manifest row and launch one VI run. |
| `model.py` | `build_flow_vi()` | Build the diagonal base distribution, semantic flow, and relaxed DSS likelihood. |
| `framework.py` | `simflow_stagewise()` | Split data, train, select a checkpoint, sample the posterior, and save outputs. |
| `framework.py` | `train_flow_stagewise()` | Run Adam stages and checkpoint evaluations. |
| `metric.py` | `sample_posterior_latents()`, `hard_support_from_draws()` | Draw posterior samples and convert activation margins to inclusion draws. |
| `metric.py` | `recovery_metrics()` | Compare VI draws and PIPs with retained reference posterior summaries. |
| `artifact.py` | `save_run_artifacts()` | Write run configuration and seed-level summaries. |
| Multi-chain MH runner | `run_multichain_mh()` | Run eight sequential component-wise random-walk MH chains. |
| `postprocess_tables.py` | Aggregation loop | Combine seed-level results and calculate confidence intervals. |

## Pseudocode

### 1. Generate datasets

```text
FOR environment IN experiment_manifest:
    FOR seed IN 400,...,499:
        X <- iid N(0,1), then center and scale each column
        A <- sample 10 predictor indices without replacement
        beta[A] <- random signs * Uniform(beta_low, beta_high)
        y <- X beta + N(0, sigma2); center y

        WRITE data, coefficient truth, and manifest record
```

### 2. Fit a variational method

```text
RUN_VI(manifest_row, method_config):
    X, y, beta_true <- read_data(manifest_row)
    train, validation, test <- split 60%/20%/20% with seed 12345

    IF Mean-field:
        posterior <- diagonal Gaussian
    ELSE:
        posterior <- semantic flow with 68 role-aware cycles

    RUN one 200-epoch warm-up stage and five 500--1000-epoch stages
        temperature <- 0.8
        loss <- negative ELBO from 128 reparameterized draws
        Adam update with gradient norm clipped at 3
        every 25 epochs, evaluate checkpoint from 64 draws

    SELECT the lowest checkpoint score after epoch 25
    DRAW 4000 posterior samples
    gamma[j] <- 1(activation[j] > global_threshold)
    PIP[j] <- mean(gamma[j]); selected[j] <- 1(PIP[j] > 0.5)
    WRITE seed-level summaries
```

RAT-Flow uses an MLP 2/64 conditioner, ResCond uses two 64-unit residual blocks, and Deep MLP uses MLP 4/256. Other training settings are shared.

### 3. Run multi-chain MH

```text
RUN_MULTICHAIN_MH(manifest_row):
    target <- known-variance Gaussian likelihood + independent N(0,1) latents

    FOR chain c IN 1,...,8, sequentially:
        seed <- simulation_seed + 100000*c
        initialize all 2p+1 latents independently from N(0,1)
        initialize coordinate proposal SDs at 0.10

        FOR sweep IN 1,...,50000:
            update each coordinate by Gaussian random-walk MH
            during sweeps 1,...,10000, adapt every 100 sweeps
            toward acceptance 0.44

        RETAIN sweeps 10001,...,50000 without thinning
        WRITE acceptance, proposal-scale summary, and runtime
```

Retained reference posterior summaries provide the numerical posterior-reference and MCMC support entries in Appendix C.1. The disclosed eight-chain MH runs provide the algorithm specification and timing evidence. Inclusion is defined by the hard activation margin and requires no coefficient-magnitude cutoff.

### 4. Aggregate results

```text
FOR every completed VI seed:
    LOAD seed-level summaries and matching reference PIPs
    COMPUTE D_SKL_A, D_JS_0, RMSE_PIP, AUROC, and AUPRC

GROUP by environment and configuration
REPORT mean metric and 95% bootstrap interval
WRITE posterior_support_summary.csv
```

## Software and execution context

VI jobs use CPU PyTorch in the `nf311` Conda environment after loading `python3/3.10.9_anaconda2023.03_libmamba`. They request four CPU cores; flow jobs request 24 GB and mean-field jobs 12 GB. Multi-chain MH jobs use the cluster R module, one CPU core, and 24 GB. Exact package patch versions were not serialized in the seed-level artifacts.
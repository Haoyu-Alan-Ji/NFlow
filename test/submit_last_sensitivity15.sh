#!/bin/bash
set -euo pipefail

cd ~/NFlow
mkdir -p logs

ARRAY_SPEC="1-100%10"

submit_last () {
  local job_name="$1"
  local manifest="$2"
  local output_root="$3"
  local mcmc_root="$4"
  local config_name="$5"
  local coupling_type="$6"
  local conditioner_type="$7"
  local hidden_units="$8"
  local num_hidden_layers="$9"

  echo "[submit] ${job_name}"

  sbatch --array="${ARRAY_SPEC}" --job-name="${job_name}" \
    --export=ALL,MANIFEST="${manifest}",OUTPUT_ROOT="${output_root}",MCMC_ROOT="${mcmc_root}",CONFIG_NAME="${config_name}",COUPLING_TYPE="${coupling_type}",CONDITIONER_TYPE="${conditioner_type}",HIDDEN_UNITS="${hidden_units}",NUM_HIDDEN_LAYERS="${num_hidden_layers}",K_Q=64,K_G=4 \
    test/last_array.sh
}

# Baseline, n approximately p, sigma2 = 1
submit_last "last-base-mlp2" "data/n160p100/manifest_n160p100.csv" "data/n160p100/n160p100_last_output/last_default" "data/n160p100/n160p100_mcmc_output" "last_default" "semantic" "mlp" "64" "2"
submit_last "last-base-rescond" "data/n160p100/manifest_n160p100.csv" "data/n160p100/n160p100_last_output/rescond" "data/n160p100/n160p100_mcmc_output" "rescond" "semantic" "resnet" "64" "2"
submit_last "last-base-deepmlp" "data/n160p100/manifest_n160p100.csv" "data/n160p100/n160p100_last_output/deep_mlp" "data/n160p100/n160p100_mcmc_output" "deep_mlp" "semantic" "mlp" "256" "4"

# Low SNR
submit_last "last-snr-mlp2" "data/n160p100/manifest_n160p100_low_snr.csv" "data/n160p100/n160p100_last_output/last_default" "data/n160p100/n160p100_mcmc_output" "last_default" "semantic" "mlp" "64" "2"
submit_last "last-snr-rescond" "data/n160p100/manifest_n160p100_low_snr.csv" "data/n160p100/n160p100_last_output/rescond" "data/n160p100/n160p100_mcmc_output" "rescond" "semantic" "resnet" "64" "2"
submit_last "last-snr-deepmlp" "data/n160p100/manifest_n160p100_low_snr.csv" "data/n160p100/n160p100_last_output/deep_mlp" "data/n160p100/n160p100_mcmc_output" "deep_mlp" "semantic" "mlp" "256" "4"

# n > p
submit_last "last-n1000-mlp2" "data/n1000p100/manifest_n1000p100.csv" "data/n1000p100/n1000p100_last_output/last_default" "data/n1000p100/n1000p100_mcmc_output" "last_default" "semantic" "mlp" "64" "2"
submit_last "last-n1000-rescond" "data/n1000p100/manifest_n1000p100.csv" "data/n1000p100/n1000p100_last_output/rescond" "data/n1000p100/n1000p100_mcmc_output" "rescond" "semantic" "resnet" "64" "2"
submit_last "last-n1000-deepmlp" "data/n1000p100/manifest_n1000p100.csv" "data/n1000p100/n1000p100_last_output/deep_mlp" "data/n1000p100/n1000p100_mcmc_output" "deep_mlp" "semantic" "mlp" "256" "4"

# p >> n
submit_last "last-hd-mlp2" "data/n100p500/manifest_n100p500.csv" "data/n100p500/n100p500_last_output/last_default" "data/n100p500/n100p500_mcmc_output" "last_default" "semantic" "mlp" "64" "2"
submit_last "last-hd-rescond" "data/n100p500/manifest_n100p500.csv" "data/n100p500/n100p500_last_output/rescond" "data/n100p500/n100p500_mcmc_output" "rescond" "semantic" "resnet" "64" "2"
submit_last "last-hd-deepmlp" "data/n100p500/manifest_n100p500.csv" "data/n100p500/n100p500_last_output/deep_mlp" "data/n100p500/n100p500_mcmc_output" "deep_mlp" "semantic" "mlp" "256" "4"

# Weak signals
submit_last "last-weak-mlp2" "data/n160p100/manifest_n160p100_weak_signal.csv" "data/n160p100/n160p100_last_output/last_default" "data/n160p100/n160p100_mcmc_output" "last_default" "semantic" "mlp" "64" "2"
submit_last "last-weak-rescond" "data/n160p100/manifest_n160p100_weak_signal.csv" "data/n160p100/n160p100_last_output/rescond" "data/n160p100/n160p100_mcmc_output" "rescond" "semantic" "resnet" "64" "2"
submit_last "last-weak-deepmlp" "data/n160p100/manifest_n160p100_weak_signal.csv" "data/n160p100/n160p100_last_output/deep_mlp" "data/n160p100/n160p100_mcmc_output" "deep_mlp" "semantic" "mlp" "256" "4"
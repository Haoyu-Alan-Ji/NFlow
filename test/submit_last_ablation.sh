#!/bin/bash
set -euo pipefail

cd ~/NFlow
mkdir -p logs

ARRAY_SPEC="1-100%10"

submit_last () {
  local job_name="$1"
  local output_root="$2"
  local config_name="$3"
  local coupling_type="$4"
  local conditioner_type="$5"
  local hidden_units="$6"
  local num_hidden_layers="$7"
  local affine_layers_per_step="$8"

  echo "[submit] ${job_name}"

  sbatch --array="${ARRAY_SPEC}" --job-name="${job_name}" \
    --export=ALL,MANIFEST=data/n160p100/manifest_n160p100.csv,OUTPUT_ROOT="${output_root}",MCMC_ROOT=data/n160p100/n160p100_mcmc_output,CONFIG_NAME="${config_name}",COUPLING_TYPE="${coupling_type}",CONDITIONER_TYPE="${conditioner_type}",HIDDEN_UNITS="${hidden_units}",NUM_HIDDEN_LAYERS="${num_hidden_layers}",K_Q=64,K_G=4,AFFINE_LAYERS_PER_STEP="${affine_layers_per_step}" \
    test/last_array.sh
}

# Ablation study on baseline n160p100/simple.
# LaST-Flow baseline is produced by submit_last_sensitivity15.sh.

submit_last "abl-meanfield" "data/n160p100/n160p100_last_output/meanfield" "meanfield" "meanfield" "mlp" "64" "2" "3"
submit_last "abl-affine" "data/n160p100/n160p100_last_output/affine" "affine" "affine" "mlp" "64" "2" "3"
submit_last "abl-semantic-affine" "data/n160p100/n160p100_last_output/semantic_affine_control" "semantic_affine_control" "semantic_affine_control" "mlp" "64" "2" "3"

# Optional rerun of LaST baseline.
# submit_last "abl-last-default" "data/n160p100/n160p100_last_output/last_default" "last_default" "semantic" "mlp" "64" "2" "3"
#!/bin/bash
set -euo pipefail

cd ~/NFlow
mkdir -p data2/logs

ARRAY_SPEC="1-100%100"

submit_last () {
  local job_name="$1"
  local output_root="$2"
  local config_name="$3"
  local coupling_type="$4"
  local conditioner_type="$5"
  local hidden_units="$6"
  local num_hidden_layers="$7"
  local affine_layers_per_step="$8"
  local k_q="$9"
  local k_g="${10}"

  echo "[submit] ${job_name}"

  sbatch --array="${ARRAY_SPEC}" --job-name="${job_name}" \
    --export=ALL,MANIFEST=data/n160p100/manifest_n160p100.csv,OUTPUT_ROOT="${output_root}",MCMC_ROOT=data/n160p100,CONFIG_NAME="${config_name}",COUPLING_TYPE="${coupling_type}",CONDITIONER_TYPE="${conditioner_type}",HIDDEN_UNITS="${hidden_units}",NUM_HIDDEN_LAYERS="${num_hidden_layers}",K_Q="${k_q}",K_G="${k_g}",AFFINE_LAYERS_PER_STEP="${affine_layers_per_step}" \
    test/last_array.sh
}

# Ablation study on baseline n160p100/simple.
# RAT-Flow baseline is produced by submit_last_sensitivity15.sh.

submit_last "abl-meanfield" "data2/n160p100/n160p100_rat_k16_output/meanfield" "meanfield" "meanfield" "mlp" "64" "2" "3" "1" "1"
submit_last "abl-affine" "data2/n160p100/n160p100_rat_k16_output/affine" "affine" "affine" "mlp" "64" "2" "3" "12" "4"
submit_last "abl-semantic-affine" "data2/n160p100/n160p100_rat_k16_output/semantic_affine_control" "semantic_affine_control" "semantic_affine_control" "mlp" "64" "2" "3" "12" "4"

# Optional rerun of the RAT baseline.
# submit_last "abl-rat-k16" "data2/n160p100/n160p100_rat_k16_output/rat_k16" "rat_k16" "semantic" "mlp" "64" "2" "3" "12" "4"

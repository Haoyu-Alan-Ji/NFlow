#!/bin/bash
set -euo pipefail

cd ~/NFlow
mkdir -p data2/logs

ARRAY_SPEC="${ARRAY_SPEC:-1-100%100}"
STANDARD_MEM="${STANDARD_MEM:-24g}"
DEEP_MEM="${DEEP_MEM:-96g}"

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
  local memory="${10}"

  echo
  echo "[submit] ${job_name}"
  echo "manifest:         ${manifest}"
  echo "output_root:      ${output_root}"
  echo "mcmc_root:        ${mcmc_root}"
  echo "conditioner:      ${conditioner_type}"
  echo "hidden structure: ${num_hidden_layers}/${hidden_units}"
  echo "flow depth:       12+4"
  echo "memory per task:  ${memory}"
  echo "array:            ${ARRAY_SPEC}"

  sbatch \
    --array="${ARRAY_SPEC}" \
    --job-name="${job_name}" \
    --mem="${memory}" \
    --output="data2/logs/%x-%A_%a.out" \
    --error="data2/logs/%x-%A_%a.err" \
    --export="ALL,MANIFEST=${manifest},OUTPUT_ROOT=${output_root},MCMC_ROOT=${mcmc_root},CONFIG_NAME=${config_name},COUPLING_TYPE=${coupling_type},CONDITIONER_TYPE=${conditioner_type},HIDDEN_UNITS=${hidden_units},NUM_HIDDEN_LAYERS=${num_hidden_layers},K_Q=12,K_G=4" \
    test/last_array.sh
}


# ============================================================
# Baseline: n = 160, p = 100, sigma2 = 1
# ============================================================

submit_last \
  "last-base-mlp2" \
  "data/n160p100/manifest_n160p100.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rat_k16" \
  "data/n160p100" \
  "rat_k16" \
  "semantic" \
  "mlp" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-base-rescond" \
  "data/n160p100/manifest_n160p100.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rescond" \
  "data/n160p100" \
  "rescond" \
  "semantic" \
  "resnet" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-base-deepmlp" \
  "data/n160p100/manifest_n160p100.csv" \
  "data2/n160p100/n160p100_rat_k16_output/deep_mlp" \
  "data/n160p100" \
  "deep_mlp" \
  "semantic" \
  "mlp" \
  "256" \
  "4" \
  "${DEEP_MEM}"


# ============================================================
# Low SNR
# ============================================================

submit_last \
  "last-snr-mlp2" \
  "data/n160p100/manifest_n160p100_low_snr.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rat_k16" \
  "data/n160p100" \
  "rat_k16" \
  "semantic" \
  "mlp" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-snr-rescond" \
  "data/n160p100/manifest_n160p100_low_snr.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rescond" \
  "data/n160p100" \
  "rescond" \
  "semantic" \
  "resnet" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-snr-deepmlp" \
  "data/n160p100/manifest_n160p100_low_snr.csv" \
  "data2/n160p100/n160p100_rat_k16_output/deep_mlp" \
  "data/n160p100" \
  "deep_mlp" \
  "semantic" \
  "mlp" \
  "256" \
  "4" \
  "${DEEP_MEM}"


# ============================================================
# n > p: n = 1000, p = 100
# ============================================================

submit_last \
  "last-n1000-mlp2" \
  "data/n1000p100/manifest_n1000p100.csv" \
  "data2/n1000p100/n1000p100_rat_k16_output/rat_k16" \
  "data/n1000p100" \
  "rat_k16" \
  "semantic" \
  "mlp" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-n1000-rescond" \
  "data/n1000p100/manifest_n1000p100.csv" \
  "data2/n1000p100/n1000p100_rat_k16_output/rescond" \
  "data/n1000p100" \
  "rescond" \
  "semantic" \
  "resnet" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-n1000-deepmlp" \
  "data/n1000p100/manifest_n1000p100.csv" \
  "data2/n1000p100/n1000p100_rat_k16_output/deep_mlp" \
  "data/n1000p100" \
  "deep_mlp" \
  "semantic" \
  "mlp" \
  "256" \
  "4" \
  "${DEEP_MEM}"


# ============================================================
# p >> n: n = 100, p = 500
# ============================================================

submit_last \
  "last-hd-mlp2" \
  "data/n100p500/manifest_n100p500.csv" \
  "data2/n100p500/n100p500_rat_k16_output/rat_k16" \
  "data/n100p500" \
  "rat_k16" \
  "semantic" \
  "mlp" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-hd-rescond" \
  "data/n100p500/manifest_n100p500.csv" \
  "data2/n100p500/n100p500_rat_k16_output/rescond" \
  "data/n100p500" \
  "rescond" \
  "semantic" \
  "resnet" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-hd-deepmlp" \
  "data/n100p500/manifest_n100p500.csv" \
  "data2/n100p500/n100p500_rat_k16_output/deep_mlp" \
  "data/n100p500" \
  "deep_mlp" \
  "semantic" \
  "mlp" \
  "256" \
  "4" \
  "${DEEP_MEM}"


# ============================================================
# Weak signals
# ============================================================

submit_last \
  "last-weak-mlp2" \
  "data/n160p100/manifest_n160p100_weak_signal.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rat_k16" \
  "data/n160p100" \
  "rat_k16" \
  "semantic" \
  "mlp" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-weak-rescond" \
  "data/n160p100/manifest_n160p100_weak_signal.csv" \
  "data2/n160p100/n160p100_rat_k16_output/rescond" \
  "data/n160p100" \
  "rescond" \
  "semantic" \
  "resnet" \
  "64" \
  "2" \
  "${STANDARD_MEM}"

submit_last \
  "last-weak-deepmlp" \
  "data/n160p100/manifest_n160p100_weak_signal.csv" \
  "data2/n160p100/n160p100_rat_k16_output/deep_mlp" \
  "data/n160p100" \
  "deep_mlp" \
  "semantic" \
  "mlp" \
  "256" \
  "4" \
  "${DEEP_MEM}"

echo
echo "[done] Submitted all 15 sensitivity arrays."
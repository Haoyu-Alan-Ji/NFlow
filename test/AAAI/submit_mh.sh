#!/bin/bash
set -euo pipefail

cd ~/NFlow
mkdir -p data2/logs

configs=(
  n160p100
  low_snr
  weak_signal
  n100p500
  n1000p100
)

manifests=(
  data/n160p100/manifest_n160p100.csv
  data/n160p100/manifest_n160p100_low_snr.csv
  data/n160p100/manifest_n160p100_weak_signal.csv
  data/n100p500/manifest_n100p500.csv
  data/n1000p100/manifest_n1000p100.csv
)

output_roots=(
  data2/n160p100/n160p100_mh8_output
  data2/n160p100/n160p100_mh8_output
  data2/n160p100/n160p100_mh8_output
  data2/n100p500/n100p500_mh8_output
  data2/n1000p100/n1000p100_mh8_output
)

N_CHAINS="${N_CHAINS:-8}"
N_ITER="${N_ITER:-50000}"
WARMUP="${WARMUP:-10000}"
PROPOSAL_SCALE="${PROPOSAL_SCALE:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-12345}"
PRINT_EVERY="${PRINT_EVERY:-5000}"

MAX_PARALLEL="${MAX_PARALLEL:-100}"
WALLTIME="${WALLTIME:-96:00:00}"
MEMORY="${MEMORY:-24g}"

target="${1:-all}"
limit="${2:-}"
submitted=0

for i in "${!configs[@]}"; do
  config="${configs[$i]}"
  manifest="${manifests[$i]}"
  output_root="${output_roots[$i]}"

  if [[ "$target" != "all" &&
        "$target" != "$config" ]]; then
    continue
  fi

  n_tasks=$((
    $(wc -l < "$manifest") - 1
  ))

  if [[ -n "$limit" ]]; then
    n_tasks="$limit"
  fi

  echo
  echo "[submit] mh8-${config}"
  echo "manifest:             ${manifest}"
  echo "output_root:          ${output_root}"
  echo "tasks:                ${n_tasks}"
  echo "max parallel:         ${MAX_PARALLEL}"
  echo "chains per task:      ${N_CHAINS}"
  echo "iterations per chain: ${N_ITER}"
  echo "warmup per chain:     ${WARMUP}"
  echo "diagnostics:          disabled"
  echo "memory per task:      ${MEMORY}"
  echo "walltime:             ${WALLTIME}"

  sbatch \
    --job-name="mh8-${config}" \
    --time="${WALLTIME}" \
    --mem="${MEMORY}" \
    --array="1-${n_tasks}%${MAX_PARALLEL}" \
    --output="data2/logs/%x-%A_%a.out" \
    --error="data2/logs/%x-%A_%a.err" \
    --export="ALL,N_CHAINS=${N_CHAINS},N_ITER=${N_ITER},WARMUP=${WARMUP},PROPOSAL_SCALE=${PROPOSAL_SCALE},SPLIT_SEED=${SPLIT_SEED},PRINT_EVERY=${PRINT_EVERY}" \
    test/mh_array.sh \
    "${manifest}" \
    "${output_root}"

  submitted=1
done

if [[ "$submitted" -eq 0 ]]; then
  echo "Usage:" >&2
  echo "  bash test/submit_mh.sh all" >&2
  echo "  bash test/submit_mh.sh n160p100" >&2
  echo "  bash test/submit_mh.sh low_snr" >&2
  echo "  bash test/submit_mh.sh weak_signal" >&2
  echo "  bash test/submit_mh.sh n100p500" >&2
  echo "  bash test/submit_mh.sh n1000p100" >&2
  echo "Optional test limit:" >&2
  echo "  bash test/submit_mh.sh n160p100 1" >&2
  exit 2
fi
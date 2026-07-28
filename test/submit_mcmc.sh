#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

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

target=${1:-all}
submitted=0

for i in "${!configs[@]}"; do
  if [[ "$target" != "all" && "$target" != "${configs[$i]}" ]]; then
    continue
  fi

  manifest=${manifests[$i]}
  n_tasks=$(($(wc -l < "$manifest") - 1))

  sbatch \
    --job-name="mcmc_${configs[$i]}" \
    --array="1-${n_tasks}%50" \
    test/mcmc_array.sh \
    "$manifest"

  submitted=1
done

if [[ "$submitted" -eq 0 ]]; then
  echo "Usage: bash test/submit_mcmc.sh [all|n160p100|low_snr|weak_signal|n100p500|n1000p100]"
  exit 1
fi
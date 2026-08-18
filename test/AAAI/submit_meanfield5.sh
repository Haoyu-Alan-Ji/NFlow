#!/bin/bash

set -euo pipefail

cd ~/NFlow
mkdir -p data2/logs

TARGET="${1:-all}"
LIMIT="${2:-}"

if [[ "$TARGET" == "baseline" || "$TARGET" == "all" ]]; then
  MANIFEST="data/n160p100/manifest_n160p100.csv"
  N_TASKS=$(($(wc -l < "$MANIFEST") - 1))
  [[ -n "$LIMIT" ]] && N_TASKS="$LIMIT"

  echo
  echo "[submit] mf-base"
  echo "manifest:    $MANIFEST"
  echo "output_root: data2/n160p100/n160p100_rat_k16_output/meanfield"
  echo "tasks:       $N_TASKS"

  sbatch \
    --job-name="mf-base" \
    --array="1-${N_TASKS}%10" \
    --time="06:00:00" \
    --cpus-per-task=4 \
    --mem=12g \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=data2/n160p100/n160p100_rat_k16_output/meanfield,MCMC_ROOT=data/n160p100,CONFIG_NAME=meanfield,COUPLING_TYPE=meanfield,K_Q=1,K_G=1" \
    test/last_array.sh
fi

if [[ "$TARGET" == "low_snr" || "$TARGET" == "all" ]]; then
  MANIFEST="data/n160p100/manifest_n160p100_low_snr.csv"
  N_TASKS=$(($(wc -l < "$MANIFEST") - 1))
  [[ -n "$LIMIT" ]] && N_TASKS="$LIMIT"

  echo
  echo "[submit] mf-low-snr"
  echo "manifest:    $MANIFEST"
  echo "output_root: data2/n160p100/n160p100_rat_k16_output/meanfield"
  echo "tasks:       $N_TASKS"

  sbatch \
    --job-name="mf-low-snr" \
    --array="1-${N_TASKS}%10" \
    --time="06:00:00" \
    --cpus-per-task=4 \
    --mem=12g \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=data2/n160p100/n160p100_rat_k16_output/meanfield,MCMC_ROOT=data/n160p100,CONFIG_NAME=meanfield,COUPLING_TYPE=meanfield,K_Q=1,K_G=1" \
    test/last_array.sh
fi

if [[ "$TARGET" == "weak_signal" || "$TARGET" == "all" ]]; then
  MANIFEST="data/n160p100/manifest_n160p100_weak_signal.csv"
  N_TASKS=$(($(wc -l < "$MANIFEST") - 1))
  [[ -n "$LIMIT" ]] && N_TASKS="$LIMIT"

  echo
  echo "[submit] mf-weak"
  echo "manifest:    $MANIFEST"
  echo "output_root: data2/n160p100/n160p100_rat_k16_output/meanfield"
  echo "tasks:       $N_TASKS"

  sbatch \
    --job-name="mf-weak" \
    --array="1-${N_TASKS}%10" \
    --time="06:00:00" \
    --cpus-per-task=4 \
    --mem=12g \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=data2/n160p100/n160p100_rat_k16_output/meanfield,MCMC_ROOT=data/n160p100,CONFIG_NAME=meanfield,COUPLING_TYPE=meanfield,K_Q=1,K_G=1" \
    test/last_array.sh
fi

if [[ "$TARGET" == "n_gt_p" || "$TARGET" == "all" ]]; then
  MANIFEST="data/n1000p100/manifest_n1000p100.csv"
  N_TASKS=$(($(wc -l < "$MANIFEST") - 1))
  [[ -n "$LIMIT" ]] && N_TASKS="$LIMIT"

  echo
  echo "[submit] mf-n1000p100"
  echo "manifest:    $MANIFEST"
  echo "output_root: data2/n1000p100/n1000p100_rat_k16_output/meanfield"
  echo "tasks:       $N_TASKS"

  sbatch \
    --job-name="mf-n1000p100" \
    --array="1-${N_TASKS}%10" \
    --time="06:00:00" \
    --cpus-per-task=4 \
    --mem=12g \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=data2/n1000p100/n1000p100_rat_k16_output/meanfield,MCMC_ROOT=data/n1000p100,CONFIG_NAME=meanfield,COUPLING_TYPE=meanfield,K_Q=1,K_G=1" \
    test/last_array.sh
fi

if [[ "$TARGET" == "p_gt_n" || "$TARGET" == "all" ]]; then
  MANIFEST="data/n100p500/manifest_n100p500.csv"
  N_TASKS=$(($(wc -l < "$MANIFEST") - 1))
  [[ -n "$LIMIT" ]] && N_TASKS="$LIMIT"

  echo
  echo "[submit] mf-n100p500"
  echo "manifest:    $MANIFEST"
  echo "output_root: data2/n100p500/n100p500_rat_k16_output/meanfield"
  echo "tasks:       $N_TASKS"

  sbatch \
    --job-name="mf-n100p500" \
    --array="1-${N_TASKS}%10" \
    --time="06:00:00" \
    --cpus-per-task=4 \
    --mem=12g \
    --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=data2/n100p500/n100p500_rat_k16_output/meanfield,MCMC_ROOT=data/n100p500,CONFIG_NAME=meanfield,COUPLING_TYPE=meanfield,K_Q=1,K_G=1" \
    test/last_array.sh
fi

case "$TARGET" in
  all|baseline|low_snr|weak_signal|n_gt_p|p_gt_n)
    ;;
  *)
    echo "Usage:"
    echo "  bash test/submit_meanfield5.sh all"
    echo "  bash test/submit_meanfield5.sh baseline"
    echo "  bash test/submit_meanfield5.sh low_snr"
    echo "  bash test/submit_meanfield5.sh weak_signal"
    echo "  bash test/submit_meanfield5.sh n_gt_p"
    echo "  bash test/submit_meanfield5.sh p_gt_n"
    echo
    echo "Optional test limit:"
    echo "  bash test/submit_meanfield5.sh low_snr 1"
    exit 1
    ;;
esac

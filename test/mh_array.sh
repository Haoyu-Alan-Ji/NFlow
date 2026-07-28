#!/bin/bash -l
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24g

set -euo pipefail

cd ~/NFlow
mkdir -p data2/logs

module purge
module load R

manifest="$1"
output_root="$2"

N_CHAINS="${N_CHAINS:-8}"
N_ITER="${N_ITER:-50000}"
WARMUP="${WARMUP:-10000}"
PROPOSAL_SCALE="${PROPOSAL_SCALE:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-12345}"
PRINT_EVERY="${PRINT_EVERY:-5000}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

printf '[info] host: %s\n' "$(hostname)"
printf '[info] manifest: %s\n' "$manifest"
printf '[info] output root: %s\n' "$output_root"
printf '[info] array task: %s\n' "$SLURM_ARRAY_TASK_ID"
printf '[info] chains: %s\n' "$N_CHAINS"
printf '[info] iterations per chain: %s\n' "$N_ITER"
printf '[info] warmup per chain: %s\n' "$WARMUP"
printf '[info] fixed iteration stopping; no diagnostics\n'

/usr/bin/time -v Rscript test/run_mh_hard_dss.R \
  "$manifest" \
  "$SLURM_ARRAY_TASK_ID" \
  "$output_root" \
  "$N_CHAINS" \
  "$N_ITER" \
  "$WARMUP" \
  "$PROPOSAL_SCALE" \
  "$SPLIT_SEED" \
  "$PRINT_EVERY"
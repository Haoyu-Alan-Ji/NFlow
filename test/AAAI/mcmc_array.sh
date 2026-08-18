#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load R

manifest=$1

N_MCMC=${N_MCMC:-10000}
S_MAX=${S_MAX:-100}
BURNIN=${BURNIN:-2000}
THIN=${THIN:-1}
BETA_EPS=${BETA_EPS:-0.05}
SPLIT_SEED=${SPLIT_SEED:-12345}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript test/run.R \
  "$manifest" \
  "$SLURM_ARRAY_TASK_ID" \
  "$N_MCMC" \
  "$S_MAX" \
  "$BURNIN" \
  "$THIN" \
  "$BETA_EPS" \
  "$SPLIT_SEED"
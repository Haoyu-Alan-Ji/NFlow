#!/bin/bash -l

#SBATCH --job-name=mcmc-n160
#SBATCH --output=logs/mcmc-n160-%A_%a.out
#SBATCH --error=logs/mcmc-n160-%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --tmp=12g

#SBATCH --array=1-100%50

cd ~/NFlow

mkdir -p logs

module load R

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RUN_R=test/run.R
MANIFEST=data/n160p100/manifest_n160p100.csv

N_MCMC=10000
S_MAX=100
BURNIN=2000
THIN=1
BETA_EPS=0.5
SPLIT_SEED=12345

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "RUN_R=${RUN_R}"
echo "MANIFEST=${MANIFEST}"
echo "N_MCMC=${N_MCMC}"
echo "S_MAX=${S_MAX}"
echo "BURNIN=${BURNIN}"
echo "THIN=${THIN}"
echo "BETA_EPS=${BETA_EPS}"
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

test -f ${RUN_R}
test -f ${MANIFEST}

Rscript ${RUN_R} ${MANIFEST} ${SLURM_ARRAY_TASK_ID} \
  ${N_MCMC} ${S_MAX} ${BURNIN} ${THIN} ${BETA_EPS} ${SPLIT_SEED}
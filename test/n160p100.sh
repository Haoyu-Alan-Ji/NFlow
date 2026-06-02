#!/bin/bash -l

#SBATCH --job-name=last-n160p100
#SBATCH --output=logs/last-n160p100-%A_%a.out
#SBATCH --error=logs/last-n160p100-%A_%a.err

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --tmp=12g

#SBATCH --array=1-100%50

cd ~/NFlow

mkdir -p logs

module load python3/3.10.9_anaconda2023.03_libmamba
source $(conda info --base)/etc/profile.d/conda.sh
source activate nf311

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

MANIFEST=data/manifest_n160p100.csv
RUN_SCRIPT=run.py

SPLIT_SEED=12345

SUPPORT_THRESHOLD=0.5

TAU_START=0.5
TAU_END=0.1
BASE_LR=2e-5
STAGE_LR_DECAY=0.9
K_Q=32
K_G=8

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MANIFEST=${MANIFEST}"
echo "RUN_SCRIPT=${RUN_SCRIPT}"
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "SUPPORT_THRESHOLD=${SUPPORT_THRESHOLD}"
echo "TAU_START=${TAU_START}"
echo "TAU_END=${TAU_END}"
echo "BASE_LR=${BASE_LR}"
echo "STAGE_LR_DECAY=${STAGE_LR_DECAY}"
echo "K_Q=${K_Q}"
echo "K_G=${K_G}"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

test -f ${RUN_SCRIPT}
test -f ${MANIFEST}

python -u ${RUN_SCRIPT} ${MANIFEST} ${SLURM_ARRAY_TASK_ID} \
  --device cpu \
  --split-seed ${SPLIT_SEED} \
  --support-threshold ${SUPPORT_THRESHOLD} \
  --tau-start ${TAU_START} \
  --tau-end ${TAU_END} \
  --base-lr ${BASE_LR} \
  --stage-lr-decay ${STAGE_LR_DECAY} \
  --K-q ${K_Q} \
  --K-g ${K_G}
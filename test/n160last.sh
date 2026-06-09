#!/bin/bash -l

#SBATCH --job-name=n160last
#SBATCH --output=logs/n160last-%A_%a.out
#SBATCH --error=logs/n160last-%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --tmp=12g

#SBATCH --array=1-100%50

cd ~/NFlow

mkdir -p logs

module load python3/3.10.9_anaconda2023.03_libmamba
source $(conda info --base)/etc/profile.d/conda.sh
source activate nf311

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

MANIFEST=data/manifest_n160p100.csv
RUN_SCRIPT=run.py

MODE=lasso_recovery
OUTPUT_ROOT=data/n160p100_last_output/n160last

SPLIT_SEED=12345
SUPPORT_THRESHOLD=0.5

K_Q=64
K_G=4

TAU_START=0.8
TAU_END=0.8
Q_ENTROPY_WEIGHT=0.0

BASE_LR=1e-5
STAGE_LR_DECAY=1.0
MIN_LR_SCALE=1.0

WARMUP_EPOCHS=300
N_ANNEAL_STAGES=1
MIN_STAGE_EPOCHS=1200
MAX_STAGE_EPOCHS=1200

EVAL_EVERY=25
PRINT_EVERY=100

DIAG_R_TRAIN=256
DIAG_R_FINAL=4000

RECOVERY_MIN_EPOCH=100
RECOVERY_SCORE_COL=moment_recovery_score

ROW_ID=${SLURM_ARRAY_TASK_ID}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "ROW_ID=${ROW_ID}"
echo "MODE=${MODE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "K_Q=${K_Q}"
echo "K_G=${K_G}"
echo "TAU_START=${TAU_START}"
echo "TAU_END=${TAU_END}"
echo "START=$(date)"

test -f ${RUN_SCRIPT}
test -f ${MANIFEST}
test -d Python

python -u ${RUN_SCRIPT} ${MANIFEST} ${ROW_ID} \
  --mode ${MODE} \
  --device cpu \
  --output-root ${OUTPUT_ROOT} \
  --split-seed ${SPLIT_SEED} \
  --support-threshold ${SUPPORT_THRESHOLD} \
  --tau-start ${TAU_START} \
  --tau-end ${TAU_END} \
  --base-lr ${BASE_LR} \
  --stage-lr-decay ${STAGE_LR_DECAY} \
  --min-lr-scale ${MIN_LR_SCALE} \
  --warmup-epochs ${WARMUP_EPOCHS} \
  --n-anneal-stages ${N_ANNEAL_STAGES} \
  --min-stage-epochs ${MIN_STAGE_EPOCHS} \
  --max-stage-epochs ${MAX_STAGE_EPOCHS} \
  --eval-every ${EVAL_EVERY} \
  --print-every ${PRINT_EVERY} \
  --diag-R-train ${DIAG_R_TRAIN} \
  --diag-R-final ${DIAG_R_FINAL} \
  --recovery-min-epoch ${RECOVERY_MIN_EPOCH} \
  --recovery-score-col ${RECOVERY_SCORE_COL} \
  --q-entropy-weight ${Q_ENTROPY_WEIGHT} \
  --K-q ${K_Q} \
  --K-g ${K_G}

echo "END=$(date)"
echo "[done] n160last task ${SLURM_ARRAY_TASK_ID}"
#!/bin/bash -l

#SBATCH --job-name=rec32-n160p100
#SBATCH --output=logs/rec32-n160p100-%A_%a.out
#SBATCH --error=logs/rec32-n160p100-%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --tmp=12g

#SBATCH --array=1-40%20

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

MODE=recovery
MCMC_ROOT=outputs_mcmc/n160p100_input

SPLIT_SEED=12345
SUPPORT_THRESHOLD=0.5

K_Q=32
K_G=8

BASE_LR=1e-5
STAGE_LR_DECAY=0.95
MIN_LR_SCALE=0.9

EVAL_EVERY=25
PRINT_EVERY=100

DIAG_R_TRAIN=256
DIAG_R_FINAL=4000

RECOVERY_MIN_EPOCH=100
RECOVERY_SCORE_COL=moment_recovery_score

TASK0=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_INDEX=$((TASK0 / 5))
SEED_INDEX=$((TASK0 % 5))

CONFIG_ID_LIST=(E01 E02 E03 E04 E05 E06 E07 E08)

TAU_START_LIST=(0.8 0.8 0.8 0.8 1.2 1.2 1.2 1.2)
TAU_END_LIST=(0.8 0.8 0.6 0.6 1.2 1.2 0.8 0.8)

Q_ENTROPY_LIST=(0.0 0.1 0.0 0.1 0.0 0.1 0.0 0.1)

WARMUP_EPOCHS_LIST=(300 300 300 300 300 300 300 300)
N_ANNEAL_STAGES_LIST=(1 1 4 4 1 1 4 4)
MIN_STAGE_EPOCHS_LIST=(1200 1200 300 300 1200 1200 300 300)
MAX_STAGE_EPOCHS_LIST=(1200 1200 300 300 1200 1200 300 300)

CONFIG_ID=${CONFIG_ID_LIST[$CONFIG_INDEX]}
TAU_START=${TAU_START_LIST[$CONFIG_INDEX]}
TAU_END=${TAU_END_LIST[$CONFIG_INDEX]}
Q_ENTROPY_WEIGHT=${Q_ENTROPY_LIST[$CONFIG_INDEX]}
WARMUP_EPOCHS=${WARMUP_EPOCHS_LIST[$CONFIG_INDEX]}
N_ANNEAL_STAGES=${N_ANNEAL_STAGES_LIST[$CONFIG_INDEX]}
MIN_STAGE_EPOCHS=${MIN_STAGE_EPOCHS_LIST[$CONFIG_INDEX]}
MAX_STAGE_EPOCHS=${MAX_STAGE_EPOCHS_LIST[$CONFIG_INDEX]}

ROW_ID_LIST=(1 35 38 52 99)
ROW_ID=${ROW_ID_LIST[$SEED_INDEX]}

OUTPUT_ROOT=data/n160p100_last_output/recovery32_grid/${CONFIG_ID}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CONFIG_INDEX=${CONFIG_INDEX}"
echo "SEED_INDEX=${SEED_INDEX}"
echo "CONFIG_ID=${CONFIG_ID}"
echo "ROW_ID=${ROW_ID}"
echo "MODE=${MODE}"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MCMC_ROOT=${MCMC_ROOT}"
echo "K_Q=${K_Q}"
echo "K_G=${K_G}"
echo "TAU_START=${TAU_START}"
echo "TAU_END=${TAU_END}"
echo "Q_ENTROPY_WEIGHT=${Q_ENTROPY_WEIGHT}"
echo "BASE_LR=${BASE_LR}"
echo "STAGE_LR_DECAY=${STAGE_LR_DECAY}"
echo "MIN_LR_SCALE=${MIN_LR_SCALE}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "N_ANNEAL_STAGES=${N_ANNEAL_STAGES}"
echo "MIN_STAGE_EPOCHS=${MIN_STAGE_EPOCHS}"
echo "MAX_STAGE_EPOCHS=${MAX_STAGE_EPOCHS}"
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

test -f ${RUN_SCRIPT}
test -f ${MANIFEST}
test -d Python
test -d ${MCMC_ROOT}

python -u ${RUN_SCRIPT} ${MANIFEST} ${ROW_ID} \
  --mode ${MODE} \
  --device cpu \
  --output-root ${OUTPUT_ROOT} \
  --mcmc-root ${MCMC_ROOT} \
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
#!/bin/bash -l

#SBATCH --job-name=layer08-n160p100
#SBATCH --output=logs/layer08-n160p100-%A_%a.out
#SBATCH --error=logs/layer08-n160p100-%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --tmp=12g

#SBATCH --array=1-130%30

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

TAU_START=0.8
TAU_END=0.8
Q_ENTROPY_WEIGHT=0.0

BASE_LR=1e-5
STAGE_LR_DECAY=0.95
MIN_LR_SCALE=0.9

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

N_SEEDS=5

TASK0=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_INDEX=$((TASK0 / N_SEEDS))
SEED_INDEX=$((TASK0 % N_SEEDS))

CONFIG_ID_LIST=(
  L01 L02 L03 L04
  L05 L06 L07 L08
  L09 L10 L11 L12
  L13 L14 L15 L16
  L17 L18 L19 L20
  L21 L22 L23 L24
  L25 L26
)

K_Q_LIST=(
  16 16 16 16
  24 24 24 24
  32 32 32 32
  48 48 48 48
  64 64 64 64
  96 96 96 96
  64 96
)

K_G_LIST=(
  4 8 12 16
  4 8 12 16
  4 8 12 16
  4 8 12 16
  4 8 12 16
  4 8 12 16
  24 24
)

CONFIG_ID=${CONFIG_ID_LIST[$CONFIG_INDEX]}
K_Q=${K_Q_LIST[$CONFIG_INDEX]}
K_G=${K_G_LIST[$CONFIG_INDEX]}

# seed_400 -> row 1
# seed_434 -> row 35
# seed_437 -> row 38
# seed_451 -> row 52
# seed_498 -> row 99
ROW_ID_LIST=(1 35 38 52 99)
ROW_ID=${ROW_ID_LIST[$SEED_INDEX]}

OUTPUT_ROOT=data/n160p100_last_output/layer_fixed08/${CONFIG_ID}

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
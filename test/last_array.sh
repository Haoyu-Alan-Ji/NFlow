#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --array=1-100%10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd ~/NFlow
mkdir -p logs

PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

module load python3/3.10.9_anaconda2023.03_libmamba
source "$(conda info --base)/etc/profile.d/conda.sh"
source activate nf311

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

: "${MANIFEST:?Need MANIFEST}"
: "${OUTPUT_ROOT:?Need OUTPUT_ROOT}"
: "${MCMC_ROOT:?Need MCMC_ROOT}"

abs_path () {
  local x="$1"
  if [[ "$x" = /* ]]; then
    echo "$x"
  else
    echo "${PROJECT_ROOT}/${x}"
  fi
}

MANIFEST_ABS="$(abs_path "$MANIFEST")"
OUTPUT_ROOT_ABS="$(abs_path "$OUTPUT_ROOT")"
MCMC_ROOT_ABS="$(abs_path "$MCMC_ROOT")"

CONFIG_NAME="${CONFIG_NAME:-last_default}"
COUPLING_TYPE="${COUPLING_TYPE:-semantic}"
CONDITIONER_TYPE="${CONDITIONER_TYPE:-mlp}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-2}"
K_Q="${K_Q:-64}"
K_G="${K_G:-4}"
AFFINE_LAYERS_PER_STEP="${AFFINE_LAYERS_PER_STEP:-3}"

TAU_START="${TAU_START:-0.8}"
TAU_END="${TAU_END:-0.8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-200}"
N_ANNEAL_STAGES="${N_ANNEAL_STAGES:-5}"
MIN_STAGE_EPOCHS="${MIN_STAGE_EPOCHS:-500}"
MAX_STAGE_EPOCHS="${MAX_STAGE_EPOCHS:-1000}"
BASE_LR="${BASE_LR:-5e-5}"
STAGE_LR_DECAY="${STAGE_LR_DECAY:-0.7}"
MIN_LR_SCALE="${MIN_LR_SCALE:-0.2}"
EVAL_EVERY="${EVAL_EVERY:-25}"
PRINT_EVERY="${PRINT_EVERY:-100}"
DIAG_R_TRAIN="${DIAG_R_TRAIN:-64}"
DIAG_R_FINAL="${DIAG_R_FINAL:-4000}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.5}"
RECOVERY_MIN_EPOCH="${RECOVERY_MIN_EPOCH:-25}"

ROW_ID="${SLURM_ARRAY_TASK_ID}"

# Compatibility for datasets whose MCMC output is stored as
#   <mcmc_root>/seed_400
# rather than
#   <mcmc_root>/simple/seed_400.
if [[ ! -e "${MCMC_ROOT_ABS}/simple" && -d "${MCMC_ROOT_ABS}/seed_400" ]]; then
  ln -s . "${MCMC_ROOT_ABS}/simple" 2>/dev/null || true
fi

echo "[info] host: $(hostname)"
echo "[info] project_root: ${PROJECT_ROOT}"
echo "[info] row_id: ${ROW_ID}"
echo "[info] manifest: ${MANIFEST_ABS}"
echo "[info] output_root: ${OUTPUT_ROOT_ABS}"
echo "[info] mcmc_root: ${MCMC_ROOT_ABS}"
echo "[info] config_name: ${CONFIG_NAME}"
echo "[info] coupling_type: ${COUPLING_TYPE}"
echo "[info] conditioner_type: ${CONDITIONER_TYPE}"
echo "[info] hidden_units: ${HIDDEN_UNITS}"
echo "[info] num_hidden_layers: ${NUM_HIDDEN_LAYERS}"
echo "[info] K_q/K_g: ${K_Q}/${K_G}"

HELP="$(python test/run.py --help 2>&1 || true)"

cmd=(python test/run.py "${MANIFEST_ABS}" "${ROW_ID}")

add_arg () {
  local opt="$1"
  local val="$2"
  if echo "${HELP}" | grep -q -- "${opt}"; then
    cmd+=("${opt}" "${val}")
  fi
}

add_arg "--config-name" "${CONFIG_NAME}"
add_arg "--output-root" "${OUTPUT_ROOT_ABS}"
add_arg "--mcmc-root" "${MCMC_ROOT_ABS}"
add_arg "--coupling-type" "${COUPLING_TYPE}"
add_arg "--conditioner-type" "${CONDITIONER_TYPE}"
add_arg "--hidden-units" "${HIDDEN_UNITS}"
add_arg "--num-hidden-layers" "${NUM_HIDDEN_LAYERS}"
add_arg "--K-q" "${K_Q}"
add_arg "--K-g" "${K_G}"
add_arg "--affine-layers-per-step" "${AFFINE_LAYERS_PER_STEP}"

add_arg "--tau-start" "${TAU_START}"
add_arg "--tau-end" "${TAU_END}"
add_arg "--warmup-epochs" "${WARMUP_EPOCHS}"
add_arg "--n-anneal-stages" "${N_ANNEAL_STAGES}"
add_arg "--min-stage-epochs" "${MIN_STAGE_EPOCHS}"
add_arg "--max-stage-epochs" "${MAX_STAGE_EPOCHS}"
add_arg "--base-lr" "${BASE_LR}"
add_arg "--stage-lr-decay" "${STAGE_LR_DECAY}"
add_arg "--min-lr-scale" "${MIN_LR_SCALE}"
add_arg "--eval-every" "${EVAL_EVERY}"
add_arg "--print-every" "${PRINT_EVERY}"
add_arg "--diag-R-train" "${DIAG_R_TRAIN}"
add_arg "--diag-R-final" "${DIAG_R_FINAL}"
add_arg "--support-threshold" "${SUPPORT_THRESHOLD}"
add_arg "--recovery-min-epoch" "${RECOVERY_MIN_EPOCH}"

echo "[cmd] ${cmd[*]}"
"${cmd[@]}"

echo "[done] row_id=${ROW_ID}, config=${CONFIG_NAME}"
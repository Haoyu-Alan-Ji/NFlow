#!/bin/bash -l

#SBATCH --job-name=post-n160last
#SBATCH --output=logs/post-n160last-%j.out
#SBATCH --error=logs/post-n160last-%j.err

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --tmp=8g

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

MODE=lasso_recovery

LAST_ROOT=data/n160p100_last_output/n160last
MCMC_ROOT=outputs_mcmc/n160p100_input
OUT_DIR=data/posterior_figure_candidates/n160p100_last

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "START=$(date)"
echo "MODE=${MODE}"
echo "LAST_ROOT=${LAST_ROOT}"
echo "MCMC_ROOT=${MCMC_ROOT}"
echo "OUT_DIR=${OUT_DIR}"

test -f postprocess_figure.py
test -d ${LAST_ROOT}
test -d ${MCMC_ROOT}

python -u postprocess_figure.py \
  --mode ${MODE} \
  --last-root ${LAST_ROOT} \
  --mcmc-root ${MCMC_ROOT} \
  --out-dir ${OUT_DIR} \
  --top-active-vars 8 \
  --top-zero-vars 30 \
  --top-n 20 \
  --fig4-n-per-group 5 \
  --kde-grid-1d 256 \
  --kde-grid-2d 45 \
  --min-mcmc-active-pip 0.5

echo "END=$(date)"
echo "[done] postprocess n160p100 last"
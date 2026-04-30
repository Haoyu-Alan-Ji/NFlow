#!/bin/bash -l

#SBATCH --job-name=nf-all
#SBATCH --output=logs/nf-all-%A_%a.out
#SBATCH --error=logs/nf-all-%A_%a.err

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12g
#SBATCH --tmp=12g

#SBATCH --array=1-300%20

cd ~/NFlow

mkdir -p logs
mkdir -p data/hpc_benchmark

module load python3/3.10.9_anaconda2023.03_libmamba
source $(conda info --base)/etc/profile.d/conda.sh
source activate nf311

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

N_SEEDS=100
SETTINGS=("simple" "block_corr" "jitter")

TASK_ID=${SLURM_ARRAY_TASK_ID}

SETTING_INDEX=$(( (TASK_ID - 1) / N_SEEDS ))
SEED=$(( (TASK_ID - 1) % N_SEEDS + 1 ))

SETTING=${SETTINGS[$SETTING_INDEX]}
OUT_ROOT=data/hpc_benchmark

echo "TASK_ID=${TASK_ID}"
echo "SETTING_INDEX=${SETTING_INDEX}"
echo "SETTING=${SETTING}"
echo "SEED=${SEED}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "HOSTNAME=$(hostname)"

python -u test/HPC_benchmark.py --seed ${SEED} --setting ${SETTING} --out-root ${OUT_ROOT} --device cpu
#!/bin/bash -l

#SBATCH --job-name=lasso-ckpt64x4
#SBATCH --output=logs/lasso-ckpt64x4-%j.out
#SBATCH --error=logs/lasso-ckpt64x4-%j.err

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --tmp=4g

cd ~/NFlow

mkdir -p logs

module load python3/3.10.9_anaconda2023.03_libmamba
source $(conda info --base)/etc/profile.d/conda.sh
source activate nf311

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "START=$(date)"

test -f train_checkpoint_lasso.py
test -d data/n160p100_last_output/recovery64_4_lasso100/F01

python -u train_checkpoint_lasso.py

echo "END=$(date)"
echo "[done] lasso checkpoint rule"
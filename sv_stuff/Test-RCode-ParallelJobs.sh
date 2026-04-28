#!/bin/bash -l
#SBATCH --job-name=Test-Rcode-ID-%j.txt    # jobname 
#SBATCH --output=m/Test-Rcode-ID-o-%j.txt  # job rub mirrow
#SBATCH --error=m/Test-Rcode-ID-e-%j.txt   # error massage

#SBATCH --time=1:00:00                     # 1h of computing 
#SBATCH --ntasks=1                         # 1 core 
#SBATCH --mem=2g                           # max of 1G memory
#SBATCH --tmp=2g                           # max 1G of temporary mem available
#SBATCH --mail-type=ALL                    # send email when job aborts/begins/ends
#SBATCH --mail-user=ventz001@umn.edu       # email address to be used
#SBATCH --array=1-5

# SBATCH -p small                           # select partition
# sbatch Test-RCode.sh    # submit 
# squeue --user=ventz001  # check status

# change directory
cd /projects/standard/ventz001/ventz001/F01-Current/Test-code         

module load R                                                       # load R
srun R --vanilla < Test-RCode-ParallelJobs.R  $SLURM_ARRAY_TASK_ID  # run script in R


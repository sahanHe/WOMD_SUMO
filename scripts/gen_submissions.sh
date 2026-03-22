#!/bin/bash

#SBATCH --job-name=submissions
#SBATCH --mail-user=erdao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5g
#SBATCH --array=0-149  # How many workers to run in parallel
#SBATCH --time=00-0:20:00
#SBATCH --account=henryliu98
#SBATCH --partition=standard
#SBATCH --output=/home/erdao/SUMOBaseline/outputs/slurm-logs/%x-%j.log

cd /home/erdao/SUMOBaseline

# module purge
# module load python3.10-anaconda/2023.03

srun python3 src/wosac/gen_submission.py --config=configs/gl/cfg_submission_validation_1006.yml --shard_idx=$SLURM_ARRAY_TASK_ID

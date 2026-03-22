#!/bin/bash

#SBATCH --job-name=test-final0321
#SBATCH --mail-user=erdao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7g
#SBATCH --array=0-1
#SBATCH --time=00-16:00:00
#SBATCH --account=henryliu1
#SBATCH --partition=standard
#SBATCH --output=/home/erdao/SUMOBaseline/outputs/slurm-logs/%x-%j.log

cd /home/erdao/SUMOBaseline

# module purge
# module load python3.10-anaconda/2023.03

srun python3 src/wosac/gen_rollouts.py --config=configs/gl/cfg_rollouts_0223test.yml --batch_idx=$SLURM_ARRAY_TASK_ID
#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
#SBATCH --mincpus=39
#SBATCH -n 16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode058


echo "Running on node: $SLURM_JOB_NODELIST ;;  in directory $PWD"

python3 SNLI.py
#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
#SBATCH --mincpus=39
#SBATCH -n 16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --constraint="2080ti|phase3|2080ti,phase3"



source ./sota/bin/activate
cd explain 

python trainer.py --bert_path ../roberta-base --data_dir ../multinli_1.0_all --task mnli --save_path ../save_mnli --gpus=0,1,2,3  --precision 8 --lr=2e-5 --batch_size=5 --lamb=1.0 --workers=40 --max_epoch=20
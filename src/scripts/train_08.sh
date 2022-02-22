#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=task08_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task08_HepaticVessel/   \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --clip_range -1 231 \
  --mean_std 86.50 46.06 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 4

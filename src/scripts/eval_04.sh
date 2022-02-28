#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=task04_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task04_Hippocampus/   \
  --split_json dataset_5slices.json \
  --img_size 240 240 5 \
  --clip_range -1000000 1000000 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 4 \
  --evaluation 1 \
  --model_path MedicalSegmentation/1gf36cve/checkpoints/epoch=100-step=23431.ckpt

#!/bin/bash

#SBATCH --job-name=msd_2d
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=task09_2d_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task09_Spleen/   \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --clip_range -41 176 \
  --mean_std 104.90 38.06 \
  --in_channels 1 \
  --out_channels 2 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 4 \
  --force_2d 1 \
  --evaluation 1 \
  --model_path MedicalSegmentation/2a54nzdl/checkpoints/epoch=134-step=11879.ckpt

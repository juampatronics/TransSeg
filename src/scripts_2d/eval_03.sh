#!/bin/bash

#SBATCH --job-name=msd_2d
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=9gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=task03_2d_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task03_Liver/   \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --clip_range -27 210 \
  --mean_std 104.12 42.11 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 4 \
  --force_2d 1 \
  --evaluation 1 \
  --model_path MedicalSegmentation/2edbfahy/checkpoints/epoch=16-step=24734.ckpt

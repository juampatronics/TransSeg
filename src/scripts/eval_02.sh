#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=task02_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task02_Heart/   \
  --split_json dataset_5slices.json \
  --img_size 320 320 5 \
  --clip_range -1000000 1000000 \
  --in_channels 1 \
  --out_channels 2 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 4 \
  --evaluation 1 \
  --model_path MedicalSegmentation/3vz3a5dl/checkpoints/epoch=304-step=17689.ckpt > eval_3d_task02.log

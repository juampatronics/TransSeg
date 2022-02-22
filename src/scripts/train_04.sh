python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task04_Hippocampus/   \
  --split_json dataset_5slices.json \
  --img_size 240 240 5 \
  --clip_range -1000000 1000000 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 4 \
  --eval_batch_size 4

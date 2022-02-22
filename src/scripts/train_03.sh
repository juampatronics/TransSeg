python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task03_Liver/   \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --clip_range -27 210 \
  --mean_std 104.12 42.11 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 4 \
  --eval_batch_size 4

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task06_Lung \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --unit_range -1000 750 \
  --in_channels 1 \
  --out_channels 2 \
  --max_steps 25000 \
  --train_batch_size 2 \
  --eval_batch_size 2
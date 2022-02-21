python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/Task05_Prostate/   \
  --split_json dataset_5slices.json \
  --img_size 320 320 5 \
  --unit_range 0 2048 \
  --in_channels 2 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 4 \
  --eval_batch_size 4
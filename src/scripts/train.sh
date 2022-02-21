conda activate beit

python main.py \
  --data_dir /sailhome/yuhuiz/develop/data/MedicalImages/msd/processed/TaskTBD \
  --split_json dataset_5slices.json \
  --img_size Height Width 5 \
  --unit_range Lower Upper \
  --in_channels SeeData \
  --out_channels NumLabels \
  --max_steps 25000 \ # See training curve
  --train_batch_size 2 \ # Run on 8 GPUs
  --eval_batch_size 2
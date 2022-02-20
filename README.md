# Effectively Adapting Large-scale Pre-trained Vision Transformers to 3D Medical Image Segmentation

## Prerequisites

- Linux
- Python >= 3.8
- PyTorch >= 1.7.1
- PyTorch Lightning >= 1.4.9

## Getting Started

### Installation

Please install dependencies by

```bash
conda env create -f environment.yml
```

### Dataset and Model

- We use the **[BCV](https://www.synapse.org/\#!Synapse:syn3193805/wiki/217789)**, **[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)**, **[MSD Spleen](https://drive.google.com/file/d/1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE/view?usp=sharing)**, **[MSD Brain](https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=sharing)**, and **[MSD Hepatic Vessel](https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=sharing)** dataset, please register and download these datasets.
- Decompress each dataset and move it to the corresponding the `data/` folder (e.g., move **BCV** to `data/bcv30/`)
- Run the pre-processing script *split_data_to_sliced_nii.py* in each folder to generate processed data. (e.g., **BCV** will be processed to `data/bcv30/bcv18-12-5slices/`).
- Run the weight downloading script *download_weights.sh* in the `backbones/encoders/pretrained_models/`folder.

### Training

1. To train the **window-based** segmentation model, run:

```bash
python main.py \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --data_dir data/bcv30/bcv18-12-5slices/ \
  --split_json dataset_5slices.json \
  --accumulate_grad_batches 1 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --warmup_steps 20 \
  --encoder beit \
  --val_check_interval 1.0 \
  --max_steps 25000 \
  --use_pretrained 1 \
  --loss_type dicefocal
```

To change the dataset, modify `--data_dir`

To change the different encoder, set `--encoder swint/videoswint/dino`.

To disable pre-trained weight initialization, set `--use_pretrained 0`.

To adjust the number of training steps for **MSD Brain** and **MSD Hepatic Vessel**, set `--max_steps 250000`.


2. To train the **slice-based** segmentation model, simply add `--force_2d 1` and run:

```bash
python main.py \
  ...same command as above... \
  --force_2d 1
```

Results are displayed at the end of training and can also be found at `wandb/` (open with `wandb` web or local client).

Model files are saved in `MedicalSegmentation/` folder. 

3. To train the **volume-based** segmentation model, follow the tutorial of the **official UNETR release** and set the correct training and validation set:

```
https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb
```

Will clean and integrate volume-based method in the final code release.

### Evaluation

To evaluate any trained segmentation model, simply add `--evaluation 1 --model_path <path/to/checkpoint>` and run:

```
python main.py \
  --evaluation 1 \
  --model_path <path/to/model_file> \
  ...same command as above...
```

The results are displayed at the end of evaluation and can also be found at `wandb/` (open with `wandb` web or local client).

The predictions are saved in `dumps/` (open with `pickle.Unpickler`) .


## ❗ Common Q&A

If you have any questions, please let us know.

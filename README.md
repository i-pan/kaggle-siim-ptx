# Kaggle SIIM-ACR Pneumothorax Segmentation (#8/1475)

## Hardware
- Ubuntu 16.04 LTS 
- 64 GB RAM / 2 TB HDD
- 1x NVIDIA Titan V100 32GB 
- 1x Titan V 12GB

## Software
- Python 3.7.4
- CUDA 10.0
- cuDNN 7.6
- PyTorch 1.1

## Model Checkpoints 
_PLEASE NOTE: Some of the model checkpoints are unfortunately corrupted. Thus certain commands in the inference scripts will not work as intended. Training scripts are available to retrain all the models._

Download from Kaggle:
```
kaggle datasets download vaillant/siim-ptx-checkpoints
```

Models should be unzipped into `./segment/checkpoints/` in order to run code as is. There should be 3 folders:
```
./segment/checkpoints/TRAIN_V100/
./segment/checkpoints/TRAIN_SEGMENT/
./segment/checkpoints/TRAIN_DEEPLABXY/
```

## Instructions
See `entry_points.md` for reproducing results. Relative filepaths and directories are used, so the code should work as is. 

Note that `TRAIN_V100` and `TRAIN_DEEPLABXY` models require V100 32GB GPUs to train with the current configurations. If you wish to train these models on a lower capacity GPU, I suggest using the following flag options: 

`--grad-accum 8 --batch-size 2` or `--grad-accum 16 --batch-size 1`

Model performance is not guaranteed to be the same with these modifications. 

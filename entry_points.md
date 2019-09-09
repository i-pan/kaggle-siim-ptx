## Setup environment

```
conda create -n siim-ptx python=3.7 pip

conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch

# Install mmdetection
git clone https://github.com/open-mmlab/mmdetection/
cd mmdetection
pip install Cython
python setup.py develop
# pip install -v -e .

conda install pandas scikit-learn scikit-image
pip install albumentations pretrainedmodels pydicom adabound
```

## Download data

Data should be downloaded into:
  `./data/dicom-images-train/`
  `./data/dicom-images-stage2/`

Scripts to help with data downloading are available in `./etl/`, but make sure that data are in the appropriate directories. Note that we did not retrain models on stage 2 train. A list of image IDs to exclude from the stage 2 train data is available in `./stage1test.txt`.

## Process data

```
cd ./etl/
python 0_convert_data_to_png.py
python 1_get_png_masks_and_assign_labels.py
python 2_create_data_splits.py 
```

## Train models

```
cd ./segment/scripts/
bash TRAIN_V100.sh 
bash TRAIN_SEGMENT.sh
bash TRAIN_DEEPLABXY.sh
```

## Predict on stage 2 test data

```
cd ./segment/scripts/
bash STAGE2_PREDICT_V100.sh 
bash STAGE2_PREDICT_SEGMENT.sh
bash STAGE2_PREDICT_DEEPLABXY.sh
```

## Create submission

```
cd ./submit/
python create_submission_partitioned.py
```

Submissions will be in `./submissions/` as `submission0.csv` (best) and `submission1.csv`. 

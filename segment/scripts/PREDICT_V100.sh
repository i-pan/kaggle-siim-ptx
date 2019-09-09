module load anaconda/3-5.2.0
source activate siim-ptx 
source deactivate
source activate siim-ptx
PYTHONPATH=''
cd scratch/siim-ptx/segment/bash-scripts/


python PredictDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_V100Flip/o0/i0_resnet50.csv \
    --gn \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python PredictDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_V100Flip/o0/i1_resnet101.csv \
    --gn \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python PredictDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_V100Flip/o0/i2_resnext50.csv \
    --gn \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python PredictDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_V100Flip/o0/i3_resnext101.csv \
    --gn \
    --class-mode \
    --batch-size 1 --imsize-x 960 --imsize-y 960 \
    --gpu 2
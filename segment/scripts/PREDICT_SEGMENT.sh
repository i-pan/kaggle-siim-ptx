module load anaconda/3-5.2.0
source activate siim-ptx 
source deactivate
source activate siim-ptx
PYTHONPATH=''
cd scratch/siim-ptx/segment/bash-scripts/


python PredictDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i5o0/resnet50 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_SEGMENTFlip/o0/i5_resnet50.pkl \
    --gn \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 0

python PredictDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i6o0/resnet101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_SEGMENTFlip/o0/i6_resnet101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 0

python PredictDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i7o0/resnext50 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_SEGMENTFlip/o0/i7_resnext50.pkl \
    --gn \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 0

python PredictDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i8o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_SEGMENTFlip/o0/i8_resnext101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 0
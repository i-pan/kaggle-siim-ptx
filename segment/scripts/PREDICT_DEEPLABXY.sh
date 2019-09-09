module load anaconda/3-5.2.0
source activate siim-ptx 
source deactivate
source activate siim-ptx
PYTHONPATH=''
cd scratch/siim-ptx/segment/bash-scripts/


python PredictDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i0o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_DEEPLABXYFlip/o0/i0_resnext101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 1

python PredictDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i1o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_DEEPLABXYFlip/o0/i1_resnext101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 1

python PredictDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i2o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_DEEPLABXYFlip/o0/i2_resnext101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 1

python PredictDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i3o0/resnext101 \
    ../../data/pngs/test \
    ../lb-predictions/TRAIN_DEEPLABXYFlip/o0/i3_resnext101.pkl \
    --gn \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 1


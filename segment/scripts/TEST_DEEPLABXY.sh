python TestDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i0o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_DEEPLABXYFlip/o0/i0_resnext101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 0

python TestDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i1o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_DEEPLABXYFlip/o0/i1_resnext101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python TestDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i2o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_DEEPLABXYFlip/o0/i2_resnext101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python TestDeepLabSnapshotV3.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i3o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_DEEPLABXYFlip/o0/i3_resnext101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2


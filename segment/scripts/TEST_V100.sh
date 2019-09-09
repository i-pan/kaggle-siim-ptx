python TestDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_V100Flip/o0/i0_resnet50.csv \
    --gn \
    --class-mode \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 3

python TestDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_V100Flip/o0/i1_resnet101.csv \
    --gn \
    --class-mode \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 3

python TestDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_V100Flip/o0/i2_resnext50.csv \
    --gn \
    --class-mode \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 3

python TestDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_V100Flip/o0/i3_resnext101.csv \
    --gn \
    --class-mode \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 960 --imsize-y 960 \
    --gpu 3
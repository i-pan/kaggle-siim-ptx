python TrainDeepLab.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 4.9 --gn \
    --inner-fold 0 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0

python TrainDeepLab.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 4.9 --gn \
    --inner-fold 1 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 1

python TrainDeepLab.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 4.9 --gn \
    --inner-fold 2 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 2

python TrainDeepLab.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 4.9 --gn \
    --inner-fold 3 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 960 --imsize-y 960 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 3
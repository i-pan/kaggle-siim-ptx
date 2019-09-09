python TrainDeepLabV2.py resnet50_gn_ws \
   ../checkpoints/TRAIN_DEEPLABXY/i0o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 7.9 --gn \
    --inner-fold 0 --outer-fold 0 \
    --thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024  \
    --cosine-anneal --total-epochs 80 --num-snapshots 4 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0 --seed 86 --verbosity 20

python TrainDeepLabV2.py resnext101_gn_ws \
   ../checkpoints/TRAIN_DEEPLABXY/i1o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 7.9 --gn \
    --inner-fold 1 --outer-fold 0 \
    --thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024  \
    --cosine-anneal --total-epochs 80 --num-snapshots 4 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0 --seed 87

python TrainDeepLabV2.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i2o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 7.9 --gn \
    --inner-fold 2 --outer-fold 0 \
    --thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024  \
    --cosine-anneal --total-epochs 80 --num-snapshots 4 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0 --seed 88

python TrainDeepLabV2.py resnext101_gn_ws \
    ../checkpoints/TRAIN_DEEPLABXY/i3o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --loss weighted_bce --pos-frac 0.1 --neg-frac 7.9 --gn \
    --inner-fold 3 --outer-fold 0 \
    --thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95 \
    --grad-accum 4 \
    --batch-size 4 --imsize-x 1024 --imsize-y 1024  \
    --cosine-anneal --total-epochs 80 --num-snapshots 4 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0 --seed 89
python TrainDeepLab.py resnet50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i5o0/resnet50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --pos-only --loss soft_dice --gn \
    --inner-fold 5 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 16 \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 0 --verbosity 50

python TrainDeepLab.py resnet101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i6o0/resnet101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --pos-only --loss soft_dice --gn \
    --inner-fold 1 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 16 \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 1 --verbosity 50

python TrainDeepLab.py resnext50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i7o0/resnext50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --pos-only --loss soft_dice --gn \
    --inner-fold 7 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 16 \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 2 --verbosity 50

python TrainDeepLab.py resnext101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i8o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    --pos-only --loss soft_dice --gn \
    --inner-fold 8 --outer-fold 0 \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --grad-accum 16 \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --cosine-anneal --total-epochs 100 --num-snapshots 5 \
    --optimizer sgd --initial-lr 1e-2 --momentum 0.9 --eta-min 1e-4 \
    --save_best --gpu 3 --verbosity 50
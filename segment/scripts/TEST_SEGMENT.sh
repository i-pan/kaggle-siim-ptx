python TestDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i5o0/resnet50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_SEGMENTFlip/o0/i5_resnet50.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 2

python TestDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i6o0/resnet101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_SEGMENTFlip/o0/i6_resnet101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 2

python TestDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i7o0/resnext50 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_SEGMENTFlip/o0/i7_resnext50.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1280 --imsize-y 1280 \
    --gpu 2

python TestDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_SEGMENT/i8o0/resnext101 \
    ../../data/pngs/train \
    ../../data/masks/train \
    ../local-cv-predictions/TRAIN_SEGMENTFlip/o0/i8_resnext101.pkl \
    --gn \
    --outer-fold 0 --outer-only \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2
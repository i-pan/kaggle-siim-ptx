python Stage2PredictDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i0_resnet50.csv \
    --gn --group 0 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i1_resnet101.csv \
    --gn --group 0 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i2_resnext50.csv \
    --gn --group 0 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i3_resnext101.csv \
    --gn --group 0 \
    --class-mode \
    --batch-size 1 --imsize-x 960 --imsize-y 960 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i0_resnet50.csv \
    --gn --group 1 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i1_resnet101.csv \
    --gn --group 1 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i2_resnext50.csv \
    --gn --group 1 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i3_resnext101.csv \
    --gn --group 1 \
    --class-mode \
    --batch-size 1 --imsize-x 960 --imsize-y 960 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnet50_gn_ws \
    ../checkpoints/TRAIN_V100/i0o0/resnet50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i0_resnet50.csv \
    --gn --group 2 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnet101_gn_ws \
    ../checkpoints/TRAIN_V100/i1o0/resnet101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i1_resnet101.csv \
    --gn --group 2 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext50_gn_ws \
    ../checkpoints/TRAIN_V100/i2o0/resnext50 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i2_resnext50.csv \
    --gn --group 2 \
    --class-mode \
    --batch-size 1 --imsize-x 1024 --imsize-y 1024 \
    --gpu 2

python Stage2PredictDeepLabSnapshot.py resnext101_gn_ws \
    ../checkpoints/TRAIN_V100/i3o0/resnext101 \
    ../../data/pngs/stage2 \
    ../stage2-predictions/TRAIN_V100Flip/o0/i3_resnext101.csv \
    --gn --group 2 \
    --class-mode \
    --batch-size 1 --imsize-x 960 --imsize-y 960 \
    --gpu 2
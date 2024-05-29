#!/bin/bash

# Main constant training variables 
WEIGHT_DEACAY=0.0005
MOMENTUM=0.9
BATCH_SIZE=128
NETWORK="resnet18"
TAU=3
MAX_EXPERTS=1
GMMS=1
DATASET="pathmnist"
NEPOCHS=1
NUM_WORKERS=4

GPU=0

# Searched hyperparamiters
SEEDs=(0 1 2)
LRs=(0.1 0.05 0.01 0.005 0.001)
ALPHAs=(0 0.25 0.5 0.9 0.99 0.999 1)

for SEED in ${SEEDs[@]}; do
    for LR in ${LRs[@]}; do
        for ALPHA in ${ALPHAs[@]}; do
            EXP_NAME="online_gridsearch_alpha_"$ALPHA"_lr_"$LR"_seed_"$SEED

            echo "========== | ========== | ========== | ========== | =========="
            echo "weightDecay = $WEIGHT_DEACAY," \
                "momentum = $MOMENTUM," \
                "batchSize = $BATCH_SIZE," \
                "network = $NETWORK," \
                "tau = $TAU" \
                "maxExperts = $MAX_EXPERTS," \
                "lrMin = $LR_MIN," \
                "gmms = $GMMS," \
                "dataset = $DATASET," \
                "lr = $LR," \
                "alpha = $ALPHA"
            python src/main_incremental.py --approach seed --pretrained --skip-ft-pretraining --gmms $GMMS --max-experts $MAX_EXPERTS --use-multivariate --nepochs $NEPOCHS --gpu $GPU --tau $TAU --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --datasets $DATASET --lr $LR --weight-decay $WEIGHT_DEACAY --alpha $ALPHA --use-test-as-val --network $NETWORK --momentum $MOMENTUM --exp-name $EXP_NAME --seed $SEED
        done
    done
done
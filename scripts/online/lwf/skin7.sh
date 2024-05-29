#!/bin/bash

# Main constant training variables 
WEIGHT_DEACAY=0.0005
MOMENTUM=0.9
BATCH_SIZE=128
NETWORK="resnet18"
DATASET="skin7"
NEPOCHS=1
NUM_WORKERS=4

GPU=0

# Searched hyperparamiters
SEEDs=(0 1 2)
LRs=(0.1 0.05 0.01 0.005 0.001)
LAMBDAs=(0.1 1 5 10 25 50 100)

for SEED in ${SEEDs[@]}; do
    for LR in ${LRs[@]}; do
        for LAMBDA in ${LAMBDAs[@]}; do
            EXP_NAME="online_gridsearch_lambda_"$LAMBDA"_lr_"$LR"_seed_"$SEED

            echo "========== | ========== | ========== | ========== | =========="
            echo "weightDecay = $WEIGHT_DEACAY," \
                "momentum = $MOMENTUM," \
                "batchSize = $BATCH_SIZE," \
                "network = $NETWORK," \
                "dataset = $DATASET," \
                "lr = $LR," \
                "lambda = $LAMBDA" \
                "seed = $SEED"
            python src/main_incremental.py --approach lwf --pretrained --nepochs $NEPOCHS --gpu $GPU --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --datasets $DATASET --lr $LR --weight-decay $WEIGHT_DEACAY --lamb $LAMBDA --use-test-as-val --network $NETWORK --momentum $MOMENTUM --exp-name $EXP_NAME --seed $SEED
        done
    done
done
#!/bin/bash

# Main constant training variables 
WEIGHT_DEACAY=0.0005
MOMENTUM=0.9
BATCH_SIZE=128
NETWORK="resnet18"
TAU=3
MAX_EXPERTS=1
GMMS=1
DATASET="cifar100"
NEPOCHS=200
NUM_WORKERS=4
NUM_TASKS=10

LR=0
ALPHA=0

GPU=0
SEED=0

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

# 1. Full covariance matrix
EXP_NAME="full_cov_matrix"
python src/main_incremental.py --approach seed --gmms $GMMS --max-experts $MAX_EXPERTS --use-multivariate --nepochs $NEPOCHS --gpu $GPU --tau $TAU --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --datasets $DATASET --num-tasks $NUM_TASKS --lr $LR --weight-decay $WEIGHT_DEACAY --alpha $ALPHA --use-test-as-val --network $NETWORK --momentum $MOMENTUM --exp-name $EXP_NAME --seed $SEED

# 2. Diagonal of cavariance matrix
EXP_NAME="diagonal_cov_matrix"
python src/main_incremental.py --approach seed --gmms $GMMS --max-experts $MAX_EXPERTS --nepochs $NEPOCHS --gpu $GPU --tau $TAU --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --datasets $DATASET --num-tasks $NUM_TASKS --lr $LR --weight-decay $WEIGHT_DEACAY --alpha $ALPHA --use-test-as-val --network $NETWORK --momentum $MOMENTUM --exp-name $EXP_NAME --seed $SEED

# 3. Nearest mean classifier
EXP_NAME="nearest_mean_classifier"
python src/main_incremental.py --approach seed --gmms $GMMS --max-experts $MAX_EXPERTS --use-nmc --nepochs $NEPOCHS --gpu $GPU --tau $TAU --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --datasets $DATASET --num-tasks $NUM_TASKS --lr $LR --weight-decay $WEIGHT_DEACAY --alpha $ALPHA --use-test-as-val --network $NETWORK --momentum $MOMENTUM --exp-name $EXP_NAME --seed $SEED

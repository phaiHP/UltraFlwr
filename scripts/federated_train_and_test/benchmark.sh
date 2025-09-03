#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# Clear previous run data
echo "Cleaning up previous run data..."
if [ -d "logs" ]; then
    rm -rf logs/*
    echo "Cleared logs folder"
fi
if [ -d "results" ]; then
    rm -rf results/*
    echo "Cleared results folder"
fi
if [ -d "weights" ]; then
    rm -rf weights/*
    echo "Cleared weights folder"
fi
echo "Cleanup completed"
echo ""

# Configuration variables
NUM_CLIENTS=3
DATASET_NAME_LIST=("redistributed_CAMMA_cholec")
STRATEGY_LIST=(
    "FedAvg"
)
# STRATEGY_LIST=("FedBackboneAvg" "FedNeckMedian")

# Training parameters
BATCH_SIZE=8
EPOCHS=20
FL_ROUNDS=20
SEED_OFFSET=0

# Partition the data, comment out if already partitioned
# python3 FedYOLO/data_partitioner/fed_split.py >> logs/data_partition_log.txt 2>&1

# Loop over each dataset and strategy
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        
        echo "===================================================================="
        echo "Running with DATASET_NAME=${DATASET_NAME} and STRATEGY=${STRATEGY}"
        echo "===================================================================="
        
        # Modify the config.py file
        sed -i "s/^DETECTION_CLIENTS = {.*}/DETECTION_CLIENTS = {'${DATASET_NAME}': ${NUM_CLIENTS}}/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'strategy': .*/    'strategy': '${STRATEGY}',/" $CLIENT_CONFIG_FILE
        
        # Update training parameters
        sed -i "s/^\s*'rounds': .*/    'rounds': ${FL_ROUNDS},/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'batch_size': .*/    'batch_size': ${BATCH_SIZE},/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'epochs': .*/    'epochs': ${EPOCHS},/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'seed_offset': .*/    'seed_offset': ${SEED_OFFSET}/" $CLIENT_CONFIG_FILE
        
        # Run the base bash file
        bash "scripts/federated_train_and_test/train.sh"

        # newline
        echo ""
        echo ""
        
    done
done

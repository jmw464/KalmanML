#!/bin/bash

DATA_DIR=/global/cfs/cdirs/atlas/jmw464/mlkf_data/new
DATASET=ttbar_200p_100e

MODE=$1

echo "##############################"
echo "Working on ${DATASET}..."
echo "##############################"

if [[ $MODE == "generate" ]]; then
    echo "RUNNING IN MODE \"generate\" - Creating training data..."
    python -m scripts.policy.generate_multiplets --in_dir ${DATA_DIR}/${DATASET}/preprocessed/ --out_dir ${DATA_DIR}/${DATASET}/policy/
elif [[ $MODE == "train" ]]; then
    echo "RUNNING IN MODE \"train\" - Training policy network..."
else
    echo "ERROR: Invalid mode. Please specify either \"generate\" or \"train\"."
fi
#!/bin/bash

DATA_DIR=/global/cfs/cdirs/atlas/jmw464/mlkf_data/new
DATASET=ttbar_200p_100e
OVERWRITE=1

if [[ $OVERWRITE == 1 ]]; then
    OVERWRITE_FLAG="-o"
else
    OVERWRITE_FLAG=""
fi

echo "##############################"
echo "Working on ${DATASET}..."
echo "##############################"

echo "------------------------------"

echo "RUNNING import_hits"
python scripts/process/import_hits.py --in_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/preprocessed/ ${OVERWRITE_FLAG}

echo "------------------------------"

echo "RUNNING import_particles"
python scripts/process/import_particles.py --in_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/preprocessed/ ${OVERWRITE_FLAG}

echo "------------------------------"

echo "RUNNING import_tracks"
python scripts/process/import_tracks.py --in_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/preprocessed/ ${OVERWRITE_FLAG}

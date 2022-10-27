#!/bin/bash
set -xe

DATASET=$1
K=$2

python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.5-cg2 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.cgraph_partitions 2
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.5-cg3 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.cgraph_partitions 3
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.5-cg4 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.cgraph_partitions 4
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.5-cg5 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.cgraph_partitions 5
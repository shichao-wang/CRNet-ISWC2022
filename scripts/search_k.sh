#!/bin/bash
set -x

DATASET=$1

python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k50-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 50
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k45-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 45
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k40-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 40
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k35-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 35
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k30-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 30
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k25-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 25
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k20-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 20
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k15-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 15
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k10-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 10
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k5-wl0.5-cg1  configs/cenet/cenetcp-$DATASET.yml --model.k 5
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k3-wl0.5-cg1  configs/cenet/cenetcp-$DATASET.yml --model.k 3
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k1-wl0.5-cg1  configs/cenet/cenetcp-$DATASET.yml --model.k 1
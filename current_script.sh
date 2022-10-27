#!/bin/bash
set -xe

#SBATCH --gres=gpu:1

DATASET="ICEWS14"

# python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k20-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 20
# python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k15-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 15
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k10-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 10
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k5-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 5
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k3-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 3
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k1-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k 1
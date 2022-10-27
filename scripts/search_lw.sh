
#!/bin/bash
set -xe

DATASET=$1
K=$2

python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.0-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.0
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.1-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.1
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.3-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.3
# python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.5-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.5
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.7-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.7
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl0.9-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 0.9
python bin/train_model.py --save-folder-path ./tmp_models/$DATASET/cenetcp-k$K-wl1.0-cg1 configs/cenet/cenetcp-$DATASET.yml --model.k $K --model.logit_weight 1.0
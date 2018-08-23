#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3
python2 tools/train_net.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_2x.yaml \
    OUTPUT_DIR trained_models/retinanet_X-101-64x4d-FPN_anchor_scale/

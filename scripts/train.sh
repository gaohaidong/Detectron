#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/train_net.py \
    --cfg configs/trafficSign_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN.yaml \
    OUTPUT_DIR trained_models/trafficSign_quater_0525

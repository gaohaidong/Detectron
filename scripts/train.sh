#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/train_net.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi_conv.yaml \
    OUTPUT_DIR trained_models/e2e_faster_rcnn_X-101-64x4d-FPN_roi_28_anchor30/

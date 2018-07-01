#!/usr/bin/env bash

python2 tools/train_net.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml \
    OUTPUT_DIR trained_models/e2e_faster_rcnn_X-101-32x8d-FPN_2x/
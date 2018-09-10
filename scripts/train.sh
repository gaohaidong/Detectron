#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
python2 tools/train_net.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_new.yaml \
    OUTPUT_DIR trained_models/retinanet_X-101-64x4d-FPN_60cls_newanno_dataaug_new_0830/

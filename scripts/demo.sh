#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi28.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --wts roi28_new.pkl \
    --csv_res  new_roi28.csv \
    --img_pad 0\
    /data/01SmartTraffic_datacastle/data/test_a/

python2 tools/eval_val.py \
    --val_file test_data_roi28_80k.csv \
    --thresh 0.9
python2 tools/eval_val.py \
    --val_file test_data_roi28_80k.csv \
    --thresh 0.95
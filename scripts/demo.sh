#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi28.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --wts trained_models/e2e_faster_rcnn_X-101-64x4d-FPN_roi28-test/train/traffic_train:traffic_val:traffic_test/generalized_rcnn/model_iter49999.pkl \
    --csv_res test_data_roi28_50k.csv \
    /data/01SmartTraffic_datacastle/data/test_a/

python2 tools/eval_val.py \
    --val_file test_data_roi28_50k.csv \
    --thresh 0.9
python2 tools/eval_val.py \
    --val_file test_data_roi28_50k.csv \
    --thresh 0.95
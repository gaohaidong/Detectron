#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi28.yaml \
    --output-dir train-visualizations \
    --image-ext jpg \
    --img_pad 0 \
    --wts trained_models/e2e_faster_rcnn_X-101-64x4d-FPN_roi_28_fc_1024/train/traffic_train:traffic_val/generalized_rcnn/model_iter59999.pkl \
    --csv_res train_data_101_roi28.csv \
    /data/01SmartTraffic_datacastle/data/train_1w/
python2 tools/eval_val.py

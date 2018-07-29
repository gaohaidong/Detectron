
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi28.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --wts trained_models/e2e_faster_rcnn_X-101-64x4d-FPN_roi28/train/traffic_train:traffic_val:traffic_test/generalized_rcnn/model_iter79999.pkl \
    --csv_res test_8w.csv \
    --img_pad 0\
    /data/01SmartTraffic/data/test_a/

python2 tools/eval_val.py \
    --val_file test_8w.csv \
    --thresh 0.95
python2 tools/eval_val.py \
    --val_file test_8w.csv \
    --thresh 0.99
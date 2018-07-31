
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/traffic_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_roi28.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --wts model_final.pkl \
    --csv_res test_res.csv \
    --img_pad 0\
    /data/01SmartTraffic/data/test_a/

python2 tools/eval_val.py \
    --val_file test_res.csv \
    --thresh 0.95
python2 tools/eval_val.py \
    --val_file test_final.csv \
    --thresh 0.99
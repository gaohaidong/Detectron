
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/cloth_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --csv_res cloth-x101-crop.csv \
    --wts trained_models/e2e_faster_rcnn_X-101-64x4d-FPN/train/cloth_train/generalized_rcnn/model_iter129999.pkl \
    /data/02Cloth/crop_test/

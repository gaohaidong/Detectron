 #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --wts models/e2e_keypoint_rcnn_R-50-FPN_s1x/model_final.pkl \
demo
#!/usr/bin/env bash
CSVFILE=round1_test_b2.csv
export CUDA_VISIBLE_DEVICES=2
python2 tools/infer_simple.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_2x.yaml \
    --output-dir val47-visualizations \
    --image-ext jpg \
    --csv_res  $CSVFILE\
    --wts ret-x-101-new.pkl \
    --save_im True \
    /data/02cloth/XuelangFiles/xuelang_round1_test_b #xuelang_round2_test_a_20180809

python2 tools/get_cloth_res_round2_det_prob.py --csv_file $CSVFILE

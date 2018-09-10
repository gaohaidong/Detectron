#!/usr/bin/env bash
CSVFILE=res/round2_test_b/newanno_60cls_aug_new/ori_det.csv
#TEST_DIR=XuelangFiles/xuelang_round2_test_b
TEST_DIR=xuelang_round2_test_b_201808031
export CUDA_VISIBLE_DEVICES=2
python2 tools/infer_simple.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_new.yaml \
    --output-dir val-visualizations \
    --image-ext jpg \
    --csv_res  $CSVFILE\
    --wts trained_models/retinanet_X-101-64x4d-FPN_60cls_newanno_dataaug_new_0830/train/cloth_train/retinanet/model_iter109999.pkl \
    /data/02cloth/$TEST_DIR

python2 tools/get_cloth_res_round2_det_prob.py --csv_file $CSVFILE

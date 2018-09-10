#!/usr/bin/env bash
RES_DIR=res/round2_test_b/newanno_60cls_aug
CSVFILE=$RES_DIR/crop4_det.csv
#TEST_DIR=crop_round2_testa_size_4
TEST_DIR=crop_round2_testb_size_4
export CUDA_VISIBLE_DEVICES=2
python2 tools/infer_simple.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_crop4.yaml \
    --output-dir val60-visualizations \
    --image-ext jpg \
    --csv_res  $CSVFILE\
    --wts ret-x-101-cloth.pkl \
    /data/02cloth/$TEST_DIR

python2 tools/get_cloth_res_round2_det_prob_patch.py --csv_file $CSVFILE --crop_size 4


#python2 tools/merge_cloth_res_round2_det_prob.py --res_dir $RES_DIR


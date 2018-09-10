#!/usr/bin/env bash
RES_DIR=res/round2_test_b/newanno_60cls_aug
CSVFILE=$RES_DIR/crop3_det.csv
#TEST_DIR=crop_round2_testa_size_3
TEST_DIR=crop_round2_testb_size_3
export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/cloth_cfgs/retinanet_X-101-64x4d-FPN_crop3.yaml \
    --output-dir val60-visualizations \
    --image-ext jpg \
    --csv_res  $CSVFILE\
    --wts trained_models/retinanet_X-101-64x4d-FPN_60cls_newanno_dataaug/train/cloth_train/retinanet/model_iter89999.pkl \
    /data/02cloth/$TEST_DIR

python2 tools/get_cloth_res_round2_det_prob_patch.py --csv_file $CSVFILE --crop_size 3


#python2 tools/merge_cloth_res_round2_det_prob.py --res_dir $RES_DIR


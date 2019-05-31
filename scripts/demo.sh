 #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/dishwash_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN.yaml \
    --output-dir dishwash \
    --image-ext jpg \
    --thresh 0.3 \
    --wts trained_models/dishwash_aug_new/train/dishwash_train/generalized_rcnn/model_final.pkl \
    --step 1 \
web_dishwash/

#/data/08hanzi/traindataset/verifyImage

# /data/08hanzi/test_dataset/
# /data/08hanzi/traindataset/trian_Image
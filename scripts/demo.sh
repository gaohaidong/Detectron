 #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/infer_simple.py \
    --cfg configs/hanzi_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml \
    --output-dir hanzi_test_im \
    --image-ext jpg \
    --thresh 0.9 \
    --wts trained_models/hanzi/anno/train/hanzi_train/generalized_rcnn/model_iter89999.pkl \
/data/08hanzi/test_dataset/
# /data/08hanzi/traindataset/trian_Image
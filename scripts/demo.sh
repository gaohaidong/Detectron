 #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/trafficSign_cfgs/e2e_faster_rcnn_X-101-64x4d-FPN.yaml \
    --output-dir trafficSign \
    --image-ext jpg \
    --thresh 0.01 \
    --wts trained_models/trafficSign/train/trafficSign_train/generalized_rcnn/model_0523_quater.pkl \
    --csv 6_res_0523_quater.csv \
    --step 2 \
/data/11trafficSign/Test_fix/

#/data/08hanzi/traindataset/verifyImage

# /data/08hanzi/test_dataset/
# /data/08hanzi/traindataset/trian_Image
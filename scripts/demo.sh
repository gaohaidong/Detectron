#!/usr/bin/env bash

python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir detectron-visualizations \
    --image-ext jpg \
    --wts detectron-download-cache/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train\:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
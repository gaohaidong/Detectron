#!/usr/bin/env python

##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization results',
        default='.',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--source', 
        dest='source',
        help='source for surveillance video', 
        # default='http://hls.open.ys7.com/openlive/acd9b6ecc9a4478c81e2b829a919eeaf.hd.m3u8',
        # default='rtmp://rtmp.open.ys7.com/openlive/acd9b6ecc9a4478c81e2b829a919eeaf',
        # default='rtsp://admin:gt121314@172.16.5.7:554/onvif1',
        # default='rtsp://10.186.88.1:554/onvif3',
        default = 'rtsp://admin:NWVNNC@192.168.1.104/Streaming/Channels/1', 
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"    
    cap = cv2.VideoCapture(args.source)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    logger.info('Surveillance video size {}'.format(size))
    out = cv2.VideoWriter('{}/181224.avi'.format(args.output_dir), fourcc, 25, size)
    out1 = cv2.VideoWriter('{}/181224_ori.avi'.format(args.output_dir), fourcc, 25, size)
  
    while (cap.isOpened()):
        ret, im = cap.read()
        if ret == False:
            break
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        

        im_visual = vis_utils.vis_one_image_opencv(
            im,  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh
        )
        out1.write(im)
        qrcode_size = 300
        im_qrcode_old = im_visual[0:qrcode_size, 0:qrcode_size]
        im_qrcode = im_qrcode_old
        
        if cls_keyps is not None:
            mid_chest = [(cls_keyps[1][0][0][5] + cls_keyps[1][0][0][4]) / 2 , (cls_keyps[1][0][1][5] + cls_keyps[1][0][1][4]) / 2]
            shoulder_width = abs(cls_keyps[1][0][0][5] - cls_keyps[1][0][0][4])
            im_qrcode = im[int(mid_chest[1] - shoulder_width * 0.2):int(mid_chest[1] + shoulder_width * 1.), \
            int(mid_chest[0] - shoulder_width * 0.5) : int(mid_chest[0] + shoulder_width * 0.5)]
            if im_qrcode.size > 10:
                im_qrcode = cv2.resize(im_qrcode, (qrcode_size, qrcode_size))
                # cv2.imshow('im_qrcode', im_qrcode).


                # cv2.imwrite('im_qrcode.png', im_qrcode)
                # im_qrcode_old = im_qrcode
                # # cv2.waitKey(1)
                # import zbar
                # from PIL import Image
                # scanner = zbar.ImageScanner()
                # scanner.parse_config("enable")
                # img = Image.open('im_qrcode.png').convert('L')
                # qrCode = zbar.Image(300, 300, 'Y800', img.tobytes())
                # scanner.scan(qrCode)
                # data = 'qrcode:'
                # for s in qrCode:
                #     data += s.data
                # del img
                # print(data)
                from pyzbar import pyzbar
                barcodes = pyzbar.decode(im_qrcode)
                for barcode in barcodes:
                    (x, y, w, h) = barcode.rect
                    im_qrcode = cv2.rectangle(im_qrcode, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    barcodeData = barcode.data.decode("utf-8")
                    # barcodeType = barcode.type
            
                    print(barcodeData)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    im_visual = cv2.putText(im_visual, 'qrcode:' + barcodeData, (0, qrcode_size+30), font, 1.2, (0, 255, 0), 2)
                im_visual[0:qrcode_size, 0:qrcode_size] = im_qrcode
        im_visual = cv2.resize(im_visual,(1920,1080))
        cv2.imshow('im_visual', im_visual)
        cv2.waitKey(1)
        out.write(im_visual)
    out.release()
    out1.release()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)

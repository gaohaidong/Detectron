# -------------------------------
# Configuration
# -------------------------------

CAFFE2_ROOT = "/env/pytorch/caffe2"
CAFFE_MODELS = "./"
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
import os
from caffe2.python import core, workspace
print("Required modules imported.")

MODEL = '/workspace/caffe_classification', 'init_net.pb', 'predict_net.pb', 400
IMAGE_DIR = '/data/02cloth/crop_round2_testa_size_5/'
import cv2, os


# -------------------------------
# Pre-processing image
# -------------------------------

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    imgScaled = skimage.transform.resize(img, (input_width, input_height))

    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
print "Functions set."

# set paths and variables from model choice and prep image
CAFFE2_ROOT = os.path.expanduser(CAFFE2_ROOT)
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images

mean = np.array([120,120,120])
mean = mean[:, np.newaxis, np.newaxis]
print "mean was set to: ", mean

INPUT_IMAGE_SIZE = MODEL[3]

# make sure all of the files are around...
if not os.path.exists(CAFFE2_ROOT):
    print("Houston, you may have a problem.")
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
print 'INIT_NET = ', INIT_NET
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
print 'PREDICT_NET = ', PREDICT_NET
if not os.path.exists(INIT_NET):
    print(INIT_NET + " not found!")
else:
    print "Found ", INIT_NET, "...Now looking for", PREDICT_NET
    if not os.path.exists(PREDICT_NET):
        print "Caffe model file, " + PREDICT_NET + " was not found!"
    else:
        print "All needed files found! Loading the model in the next block."


for im in os.listdir(IMAGE_DIR):
    IMAGE_LOCATION = os.path.join(IMAGE_DIR, im)
    img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    print "After rescale: " , img.shape

    # switch to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)


    # switch to BGR
    img = img[(2, 1, 0), :, :]

    # remove mean for better results

    img = img * 255 - mean

    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)
    print "NCHW: ", img.shape

    # -------------------------------

    # -------------------------------
    with open(INIT_NET) as f:
        init_net = f.read()
    with open(PREDICT_NET) as f:
        predict_net = f.read()

    p = workspace.Predictor(init_net, predict_net)

    # -------------------------------

    # -------------------------------
    # run the net and return prediction
    results = p.run([img]) #

    # turn it into something we can play with and examine which is in a multi-dimensional array
    results = np.asarray(results)
    print "results shape: ", results.shape
    # results shape:  (1, 1, 1000, 1, 1)
    print results
    res_cls = np.argmax(results[0][0])
    with open('cloth_cls_res.txt', 'a') as f:
        f.write('{}\t{}\t{}\n'.format(im, res_cls, max(results[0][0])))

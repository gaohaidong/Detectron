# -------------------------------
# Configuration
# -------------------------------

CAFFE2_ROOT = "/env/pytorch/caffe2"
CAFFE_MODELS = "./"
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.python import core, workspace
import urllib2
print("Required modules imported.")

MODEL = 'bvlc_googlenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 224



# -------------------------------
# Pre-processing image
# -------------------------------
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))

    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
print "Functions set."

# set paths and variables from model choice and prep image
CAFFE2_ROOT = os.path.expanduser(CAFFE2_ROOT)
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])

mean = np.array([104.00698793,116.66876762,122.67891434])
mean = mean[:, np.newaxis, np.newaxis]
print "mean was set to: ", mean

INPUT_IMAGE_SIZE = MODEL[4]

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
IMAGE_DIR = 'low_dets'
from eval_det import read_csv

annos_ft, num = read_csv('new_roi28.csv', 0.0)
annos = dict()
for im in annos_ft.keys():
    annos[im] = []
    for box in annos_ft[im]:
        box = map(int, box)
        annos[im].append([box[0], box[1], box[2], box[3]])

with open('new_res_ori.csv', 'w') as f:
    f.write('name,coordinate')
    for im in annos.keys():
        f.write('\n{},'.format(im))
        for box in annos[im]:
            f.write('{}_{}_{}_{};'.format(box[0], box[1], box[2], box[3]))

for im in os.listdir(IMAGE_DIR):
    IMAGE_LOCATION = os.path.join(IMAGE_DIR, im)
    img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    print "After crop: " , img.shape

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
    with open('cls_res.txt', 'a') as f:
        f.write('{}\t{}\t{}\n'.format(im, res_cls, max(results[0][0])))
        items = im[:-4].split('_')
    if res_cls == 0 :
        print annos[items[0] + '.jpg']
        print([items[1], items[2], items[3], items[4]])
        annos[items[0] + '.jpg'].remove([int(items[1]), int(items[2]), int(items[3]), int(items[4])])
with open('new_res.csv', 'w') as f:
    f.write('name,coordinate')
    for im in annos.keys():
        f.write('\n{},'.format(im))
        for box in annos[im]:
            f.write('{}_{}_{}_{};'.format(box[0], box[1], box[2], box[3]))

    # output
    # 985  ::  0.979059
    # daisy
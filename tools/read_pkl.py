import cPickle as pickle
import yaml
import os
# without momentum
def read_pkl(pklfile, file_out):
    if os.path.isfile(pklfile):
        pkl = pickle.load(open(pklfile, 'rb'))
        new_pkl = dict()
        new_pkl['cfg'] = pkl['cfg']
        new_pkl['blobs'] = dict()
        for item in pkl['blobs'].keys():
            if not item.endswith('_momentum'):
                new_pkl['blobs'][item] = pkl['blobs'][item]
        with open(file_out, 'wb') as f:
            pickle.dump(
                new_pkl, f,
                pickle.HIGHEST_PROTOCOL
            )

def read_cfg_from_pkl(pklfile, file_out):
    if os.path.isfile(pklfile):
        pkl = pickle.load(open(pklfile, 'rb'))
        with open(file_out, 'w') as f:
            yaml.dump(pkl['cfg'], f)


def read_pkl_in_backbone(pklfile, imagenet_pklfile, file_out):
    if os.path.isfile(pklfile):
        imagenet_pkl = pickle.load(open(imagenet_pklfile, 'rb'))
    if os.path.isfile(pklfile):
        pkl = pickle.load(open(pklfile, 'rb'))
        new_pkl = dict()
        new_pkl['cfg'] = pkl['cfg']
        new_pkl['blobs'] = dict()
        for item in pkl['blobs'].keys():
            if item in imagenet_pkl['blobs'].keys():
                new_pkl['blobs'][item] = pkl['blobs'][item]
        with open(file_out, 'wb') as f:
            pickle.dump(
                new_pkl, f,
                pickle.HIGHEST_PROTOCOL
            )

def read_pkl_blobs(pklfile):
    if os.path.isfile(pklfile):
        pkl = pickle.load(open(pklfile, 'rb'))
        for item in pkl['blobs'].keys():
            if item != 'weight_order':
                print item, pkl['blobs'][item].shape


if __name__ == '__main__':
    read_pkl('trained_models/retinanet_X-101-64x4d-FPN_60cls_newanno_dataaug/train/cloth_train/retinanet/model_iter89999.pkl', 'ret-x-101-cloth.pkl')
    # read_pkl_in_backbone('trained_models/retinanet_X-101-64x4d-FPN_2x/train/cloth_train/retinanet/model_final.pkl',
    #          'imagenet_models/X-101-64x4d.pkl',
    #          'ret-x-101-backbone.pkl')
    # read_pkl_blobs('ret-x-101-new.pkl')
    # read_cfg_from_pkl('ret-x-101-new.pkl', 'ret-x-101-new.yaml')
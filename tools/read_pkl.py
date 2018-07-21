import cPickle as pickle
import os
change_names = ['cls_score', 'bbox_pred']
def read_pkl(pklfile, file_out):
    if os.path.isfile(pklfile):
        pkl = pickle.load(open(pklfile, 'rb'))
        print(pkl['cfg'])
        return
        for item in pkl['blobs'].keys():
            for name in change_names:
                if name + '_' in item:
                    pkl['blobs'][item.replace(name + '_', name)] = pkl['blobs'][item]
                    pkl['blobs'].pop(item)
            if 'fc' in item and 'a' not in item:
                print item
                pkl['blobs'].pop(item)
        with open(file_out, 'wb') as f:
            pickle.dump(
                pkl, f,
                pickle.HIGHEST_PROTOCOL
            )



if __name__ == '__main__':
    read_pkl('roi28.pkl', 'roi28_new.pkl')
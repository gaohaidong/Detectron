import numpy as np
import argparse, sys
def read_cls(cls_csv):
    norm_dict = dict()
    with open(cls_csv) as f:
        for line in f.readlines():
            infos = line.strip().split(',')
            print infos[0]
            norm_dict[infos[0]] = round(float(infos[1]), 6)
    return norm_dict




if __name__ == '__main__':
    cls_csv = 'cls_im_norm_probs.csv'
    norm_dict = read_cls(cls_csv)
    res_csv = 'merged.csv'
    merged_csv = '{}_{}.csv'.format(res_csv[:-4], cls_csv[:-4])

    with open(merged_csv, 'w') as fw:
        with open(res_csv) as f:
            for line in f.readlines():
                infos = line.strip().split(',')
                if infos[0] in norm_dict.keys():
                    fw.write('{},{}\n'.format(infos[0], norm_dict[infos[0]]))
                else:
                    fw.write('{}\n'.format(line.strip()))

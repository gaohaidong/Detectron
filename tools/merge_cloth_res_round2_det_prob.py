import numpy as np
import argparse, sys


defect_codes_imgs = dict()
merged_codes_imgs = dict()
labels = ['norm'] + ['defect_{}'.format(i) for i in range(1,11)]

if __name__ == '__main__':

    res_csvs = ['cloth-ret-x-101-round2_scale_round2_det_prob_mergejingwei.csv', 'cloth-ret-x-101-round2_crop_2_scale_round2_det_prob_crop_2.csv']

    for res_csv in res_csvs:
        with open(res_csv) as f:
            for line in f.readlines():
                infos = line.strip().split(',')
                items = infos[0].split('|')
                if items[0][-4:] == '.jpg':
                    print items[0]
                    if items[0] not in defect_codes_imgs.keys():
                        defect_codes_imgs[items[0]] = [[] for i in range(11)]
                    thresh = float(infos[1])
                    defect_codes_imgs[items[0]][labels.index(items[1])].append(thresh)
    for img in defect_codes_imgs.keys():
        merged_codes_imgs[img] = [[] for i in range(11)]
        for i in range(1,11):
            merged_codes_imgs[img][i] = max(defect_codes_imgs[img][i])
        # merged_codes_imgs[img][0] = 1 - max([merged_codes_imgs[img][i] for i in range(1,11)])
        merged_codes_imgs[img][0] = 1.0
        for i in range(1, 11):
            merged_codes_imgs[img][0] *= 1 - merged_codes_imgs[img][i]

    merged_csv = 'merged_scale.csv'

    with open(merged_csv, 'w') as f:
        f.write('filename|defect,probability\n')

        for img in merged_codes_imgs.keys():
            for i in range(11):
                f.write('{}|{}, {}\n'.format(img, labels[i], round(merged_codes_imgs[img][i], 6)))
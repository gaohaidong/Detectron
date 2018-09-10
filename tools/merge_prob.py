import numpy as np
import argparse, sys, os


defect_codes_imgs = dict()
merged_codes_imgs = dict()
labels = ['norm'] + ['defect_{}'.format(i) for i in range(1,11)]
def prob_merge(defect_codes_imgs, weight):
    res = 0.0
    for i in range(len(weight)):
        res += defect_codes_imgs[i] * weight[i]
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--res_dir',
        dest='res_dir',
        help='csv file dir to val',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == '__main__':

    defect_codes = {'auc': 0, 'zhadong': 1, 'maoban': 2, 'cadong': 3, 'maodong': 4, 'zhixi': 5, 'diaojing': 6,
                    'quejing': 7, 'tiaohua': 8, 'youwuzi': 9, 'qita':10}
    labels = ['norm'] + ['defect_{}'.format(i) for i in range(1, 11)]
    ori_res = [['maoban', 0.41666666666666663], ['cadong', 0.846332730681288], ['zhadong', 0.5583103764921947],
               ['diaojing', 0.7863608969451316], ['quejing', 0.7296538973009561], ['maodong', 0.7627145118310217],
               ['zhixi', 0.6165038582874313], ['tiaohua', 0.49188519275612685], ['youwuzi', 0.3133066318623216],
               ['qita', 0.4980703525524249], ['auc', 0.8989664082687353]]
    crop2_res = [['maoban', 0.5], ['cadong', 0.5736907378815937], ['zhadong', 0.3732718894009216],
                 ['diaojing', 0.5126101969375285], ['quejing', 0.3716968994649375], ['maodong', 0.4841923452339297],
                 ['zhixi', 0.56259008950857], ['tiaohua', 0.6970004375331467], ['youwuzi', 0.5245043861431656],
                 ['qita', 0.364777239160452], ['auc', 0.9026593257089411]]

    crop3_res = [['maoban', 1.0], ['cadong', 0.9217347425771483], ['zhadong', 0.2893772893772894],
                 ['diaojing', 0.9355347445295328], ['quejing', 0.8871572871572871], ['maodong', 0.742229135673408],
                 ['zhixi', 0.8524487981347469], ['tiaohua', 0.9241497833955492], ['youwuzi', 0.6543171098028807],
                 ['qita', 0.6312945665307854], ['auc', 0.9635874246339361]]
    weights = [[0 for j in range(3)] for i in range(11)]
    for i in range(11):
        print defect_codes[ori_res[i][0]], defect_codes[crop2_res[i][0]]
        weights[defect_codes[ori_res[i][0]]][0] = ori_res[i][1]
        weights[defect_codes[crop2_res[i][0]]][1] = crop2_res[i][1]
        weights[defect_codes[crop3_res[i][0]]][2] = crop3_res[i][1]
    for i in range(11):
        sum_weight = sum(weights[i])
        weights[i][0] = weights[i][0] / sum_weight
        weights[i][1] = weights[i][1] / sum_weight
        weights[i][2] = weights[i][2] / sum_weight
    print weights
    print len(weights)

    args = parse_args()
    res_csvs = [os.path.join(args.res_dir, 'ori.csv'), os.path.join(args.res_dir, 'crop_2.csv'),  os.path.join(args.res_dir, 'crop_3.csv')]

    for res_csv in res_csvs:
        with open(res_csv) as f:
            for line in f.readlines():
                infos = line.strip().split(',')
                items = infos[0].split('|')
                if items[0][-4:] == '.jpg':
                    if items[0] not in defect_codes_imgs.keys():
                        defect_codes_imgs[items[0]] = [[] for i in range(11)]
                    thresh = float(infos[1])
                    defect_codes_imgs[items[0]][labels.index(items[1])].append(thresh)

    for img in defect_codes_imgs.keys():
        merged_codes_imgs[img] = [[] for i in range(11)]
        for i in range(11):
            print img, i, weights[i]
            merged_codes_imgs[img][i]\
                = prob_merge(defect_codes_imgs[img][i], weights[i])
        # merged_codes_imgs[img][0] = 1 - max([merged_codes_imgs[img][i] for i in range(1,11)])

    merged_csv = os.path.join(args.res_dir, 'merged_prob.csv')
    # merged_csv = '{}_{}.csv'.format(res_csvs[0], res_csvs[1])

    with open(merged_csv, 'w') as f:
        f.write('filename|defect,probability\n')

        for img in merged_codes_imgs.keys():
            for i in range(11):
                f.write('{}|{}, {}\n'.format(img, labels[i], round(merged_codes_imgs[img][i], 6)))
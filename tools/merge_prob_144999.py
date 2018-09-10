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
    ori_res = [['maoban', 0.14835164835164835], ['cadong', 0.7531398080573936], ['zhadong', 0.6711509715994021], ['diaojing', 0.7807869391774523], ['quejing', 0.7359290848993366], ['maodong', 0.7627420257323039], ['zhixi', 0.6932603355832541], ['tiaohua', 0.579794542777399], ['youwuzi', 0.33732688387764426], ['qita', 0.5458185711976459], ['auc', 0.906815245478037]]

    crop2_res = [['maoban', 1.0], ['cadong', 0.5834323408962974], ['zhadong', 0.5925925925925927], ['diaojing', 0.5712872891436368], ['quejing', 0.2891999119750509], ['maodong', 0.6656948742978154], ['zhixi', 0.6124667196944561], ['tiaohua', 0.9004236855098923], ['youwuzi', 0.5692887484666137], ['qita', 0.4406729805512218], ['auc', 0.8949662004166491]]


    crop3_res = [['maoban', 1.0], ['cadong', 0.9381008600611208], ['zhadong', 0.7555555555555555], ['diaojing', 0.9388358869066431], ['quejing', 0.7826617826617828], ['maodong', 0.6898700989413062], ['zhixi', 0.8841587283586135], ['tiaohua', 0.9117790997161475], ['youwuzi', 0.5765109479398532], ['qita', 0.7657648342047786], ['auc', 0.9757859603789847]]


    weights = [[0 for j in range(3)] for i in range(11)]
    for i in range(11):
        print defect_codes[ori_res[i][0]], defect_codes[crop2_res[i][0]]
        weights[defect_codes[ori_res[i][0]]][0] = ori_res[i][1]
        weights[defect_codes[crop2_res[i][0]]][1] = crop2_res[i][1]
        weights[defect_codes[crop3_res[i][0]]][2] = crop3_res[i][1]

    hard_weights = [[0, 0, 0] for i in range(11)]
    for i in range(11):
        for j in range(3):
            if weights[i][j] == max(weights[i]):
                hard_weights[i][j] = 1
                break
        # sum_weight = sum(weights[i])
        # weights[i][0] = weights[i][0] / sum_weight
        # weights[i][1] = weights[i][1] / sum_weight
        # weights[i][2] = weights[i][2] / sum_weight
    weights = hard_weights
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
            print img, i
            print defect_codes_imgs[img][i]
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
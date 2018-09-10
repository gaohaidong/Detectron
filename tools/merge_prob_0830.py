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
    ori_res = [['maoban', 0.01891891891891892], ['cadong', 0.7085624541708788], ['zhadong', 0.27945845004668535], ['diaojing', 0.5755037969586065], ['quejing', 0.5152767932219987], ['maodong', 0.615263157461947], ['zhixi', 0.38434152163964524], ['tiaohua', 0.10655782679750815], ['youwuzi', 0.3855714903255646], ['qita', 0.2770212854666483], ['auc', 0.8926571920757972]]

    crop2_res = [['maoban', 1.0], ['cadong', 0.5210943942117005], ['zhadong', 0.425], ['diaojing', 0.35474903180363115], ['quejing', 0.4545483516317186], ['maodong', 0.6727051294224773], ['zhixi', 0.36463095149642333], ['tiaohua', 0.2752288850089609], ['youwuzi', 0.47232036883861667], ['qita', 0.23976355158199256], ['auc', 0.8386208069384801]]

    crop3_res = [['maoban', 1.0], ['cadong', 0.9429332622622478], ['zhadong', 0.2972582972582972], ['diaojing', 0.7895570601438515], ['quejing', 0.7779852681813466], ['maodong', 0.7486305312127347], ['zhixi', 0.645548572864241], ['tiaohua', 0.6223056496008006], ['youwuzi', 0.5655931406327447], ['qita', 0.5187418299150541], ['auc', 0.9286391042205016]]

    crop4_res = [['maoban', 1.0], ['cadong', 0.9300470319270557], ['zhadong', 0.3176638176638177], ['diaojing', 0.7942477842751883], ['quejing', 0.6905809297113644], ['maodong', 0.743782007752596], ['zhixi', 0.7235843893824836], ['tiaohua', 0.5745079511220829], ['youwuzi', 0.5679005195989563], ['qita', 0.36435025050670367], ['auc', 0.9124784668389327]]

    crop5_res = [['maoban', 0.7], ['cadong', 0.9100491615120672], ['zhadong', 0.12158498435870699], ['diaojing', 0.6308828114551828], ['quejing', 0.41383057439902526], ['maodong', 0.6980397296685036], ['zhixi', 0.6716741374309684], ['tiaohua', 0.4906241566436029], ['youwuzi', 0.5136204719515362], ['qita', 0.315812321690556], ['auc', 0.8721145564168833]]

    weights = [[0 for j in range(2)] for i in range(11)]
    for i in range(11):
        # print defect_codes[ori_res[i][0]], defect_codes[crop2_res[i][0]]
        weights[defect_codes[ori_res[i][0]]][0] = ori_res[i][1]
        weights[defect_codes[crop2_res[i][0]]][1] = crop2_res[i][1]
        # weights[defect_codes[crop3_res[i][0]]][2] = crop3_res[i][1]
        # weights[defect_codes[crop4_res[i][0]]][3] = crop4_res[i][1]
        # weights[defect_codes[crop5_res[i][0]]][4] = crop5_res[i][1]

    hard_weights = [[0, 0] for i in range(11)]
    for i in range(11):
        # for j in range(3):
        #     if weights[i][j] == max(weights[i]):
        #         hard_weights[i][j] = 1
        #         break
        sum_weight = sum(weights[i])
        for j in range(2):
            hard_weights[i][j] = weights[i][j] / sum_weight
    weights = hard_weights
    print weights
    print len(weights)

    args = parse_args()
    res_csvs = [os.path.join(args.res_dir, 'ori.csv'),
                os.path.join(args.res_dir, 'crop_2.csv')
                # os.path.join(args.res_dir, 'crop_3.csv'),
                # os.path.join(args.res_dir, 'crop_4.csv'),
                # os.path.join(args.res_dir, 'crop_5.csv')
                ]

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

    merged_csv = os.path.join(args.res_dir, 'merged_prob_12.csv')
    # merged_csv = '{}_{}.csv'.format(res_csvs[0], res_csvs[1])

    with open(merged_csv, 'w') as f:
        f.write('filename|defect,probability\n')

        for img in merged_codes_imgs.keys():
            for i in range(11):
                f.write('{}|{}, {}\n'.format(img, labels[i], round(merged_codes_imgs[img][i], 6)))
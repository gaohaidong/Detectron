import numpy as np
res_csv = 'cloth-ret-x-101.csv'
round2_csv = '{}_round2_prob.csv'.format(res_csv[:-4])
defect_codes = {'zhadong':1, 'maoban':2, 'cadong':3, 'maodong':4, 'zhixi':5, 'diaojing':6, 'quejing':7, 'tiaohua':8, 'youzi':9, 'wuzi':9}
defect_codes_imgs = [[] for i in range(11)]
labels = ['norm'] + ['defect_{}'.format(i) for i in range(1,11)]
imgs = []
with open(res_csv) as f:
    for line in f.readlines():
        items = line.strip().split(',')
        if items[0][-4:] == '.jpg':
            imgs.append(items[0])
            boxes = items[1].split(';')
            threshs = [float(box.split('_')[-1]) for box in boxes if box != '']
            if threshs != [] and max(threshs) > 0.1:
                defect_code = boxes[np.argmax(threshs)].split('_')[-2]
                if defect_code in defect_codes.keys():
                    defect_codes_imgs[defect_codes[defect_code]].append(items[0])
                else:
                    defect_codes_imgs[10].append(items[0])
            else:
                print items[0],'norm'
                defect_codes_imgs[0].append(items[0])

with open(round2_csv, 'w') as f:
    f.write('filename|defect,probability\n')
    for i in range(11):
        for img in imgs:
            if img in defect_codes_imgs[i]:
                f.write('{}|{}, 1\n'.format(img, labels[i]))
            else:
                f.write('{}|{}, 0\n'.format(img, labels[i]))
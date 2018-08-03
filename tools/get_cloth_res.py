res_csv = 'cloth-x101-crop-test-b.csv'
tianchi_csv = 'cloth-x101-crop-tc-tx.csv'
im_threshs = dict()
with open(res_csv) as f:
    for line in f.readlines():
        items = line.strip().split(',')
        if items[0][-4:] == '.jpg':
            patch_id = int(items[0][items[0].rfind('_') + 1 : items[0].rfind('.')])
            if patch_id > 8:
                continue
            threshs = []
            # threshs = [box.split('_')[-1] for box in items[1].split(';') if box != '']
            for box in items[1].split(';'):
                if box == '':
                    continue
                bbox = map(float, box.split('_'))

                if patch_id % 3 != 0 and bbox[1] + bbox[3] < 160:
                    continue
                if patch_id >= 3 and bbox[0] + bbox[2] < 213:
                    continue
                if patch_id % 3 != 2 and bbox[1] > 800:
                    continue
                if patch_id < 6 and bbox[0] > 1067:
                    continue
                '''
                H1, W1 = 1920/2, 2560/2
                HT = 1920 / 12
                WT = 2560 / 12
                x1, y1 = bbox[0], bbox[1]
                x2 = x1 + bbox[2] - 1
                y2 = y1 + bbox[3] - 1
                if patch_id % 3 == 0:
                    if y1 > H1 - HT:
                        continue
                if patch_id % 3 == 1:
                    if y2 < HT or y1 > H1 - HT:
                        continue
                if patch_id % 3 == 2:
                    if y2 < HT:
                        continue

                if patch_id / 3 == 0:
                    if x1 > W1 - WT:
                        continue
                if patch_id / 3 == 1:
                    if x2 < WT or x1 > W1 - WT:
                        continue
                if patch_id / 3 == 2:
                    if x2 < WT:
                        continue
                '''
                threshs.append(bbox[-1])
            im = items[0][:items[0].rfind('_')]
            if im not in im_threshs:
                im_threshs[im] = []
            for thresh in threshs:
                im_threshs[im].append(thresh)
with open(tianchi_csv, 'w') as fw:
    fw.write('filename,probability')
    for im in im_threshs.keys():
        threshs = im_threshs[im]
        if threshs == []:
            threshs.append(0.000001)
        fw.write('\n{}.jpg,{}'.format(im, round(max(threshs), 6)))




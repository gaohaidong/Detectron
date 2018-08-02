res_csv = 'cloth-x101-crop.csv'
tianchi_csv = 'cloth-x101-crop-tc.csv'
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
                if patch_id % 3 != 0 and bbox[1] + bbox[3] < 320:
                    continue
                if patch_id >= 3 and bbox[0] + bbox[2] < 420:
                    continue
                if patch_id % 3 != 2 and bbox[1] > 320:
                    continue
                if patch_id < 6 and bbox[0] > 420:
                    continue
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
        fw.write('\n{},{}'.format(im, round(max(threshs), 6)))




def read_csv(anno_file, conf_thresh=0.99):
    annos = dict()
    num = 0
    with open(anno_file) as f:
        for line in f.readlines():
            items = line.split(',')
            if items[0][-3:] == 'jpg':
                bboxes = []
                for bbox_item in items[1].strip().split(';'):
                    bbox = []
                    if bbox_item == '':
                        continue
                    info = bbox_item.split('_')
                    if len(info) == 5:
                        if float(info[-1]) < conf_thresh:
                            continue
                    for i in range(4):
                        bbox.append(float(info[i]))
                    bboxes.append(bbox)
                    num += 1
                annos[items[0]] = bboxes
    return annos, num

thresh=0.9

annos, _ = read_csv('test_data_101_roi28_focal_150k.csv', thresh)
with open('test_data_101_roi28_focal_150k_{}.csv'.format(thresh), 'w') as f:
    f.write('name,coordinate')
    for img in annos.keys():
        f.write('\n{},'.format(img))
        for bbox in annos[img]:
            f.write('{}_{}_{}_{};'.format(bbox[0], bbox[1], bbox[2], bbox[3]))
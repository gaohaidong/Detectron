import argparse, sys
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

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--val_file',
        dest='val_file',
        help='csv file to val',
        default=None,
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='thresh for result',
        default=0.9,
        type=float
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    annos, _ = read_csv(args.val_file, args.thresh)
    with open('{}_{}.csv'.format(args.val_file[:-4], args.thresh), 'w') as f:
        f.write('name,coordinate')
        for img in annos.keys():
            f.write('\n{},'.format(img))
            for bbox in annos[img]:
                f.write('{}_{}_{}_{};'.format(bbox[0], bbox[1], bbox[2], bbox[3]))
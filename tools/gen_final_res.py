
from eval_det import read_csv
thresh = 0.9

annos_ft, num = read_csv('new_roi28.csv',thresh)
annos = dict()
for im in annos_ft.keys():
    annos[im] = []
    for box in annos_ft[im]:
        box = map(int, box)
        annos[im].append([box[0], box[1], box[2], box[3]])
with open('cls_res.txt') as f:
    for line in f.readlines():
        info = line.strip().split('\t')
        items = info[0][:-4].split('_')
        res_cls = int(info[1])
        res_score = float(info[2])
        if res_cls == 0 and annos_ft[items[0] + '.jpg'] < 0.999:
            annos[items[0] + '.jpg'].remove([int(items[1]), int(items[2]), int(items[3]), int(items[4])])
with open('new_res_remove_tri_{}.csv'.format(thresh), 'w') as f:
    f.write('name,coordinate')
    for im in annos.keys():
        f.write('\n{},'.format(im))
        for box in annos[im]:
            f.write('{}_{}_{}_{};'.format(box[0], box[1], box[2], box[3]))
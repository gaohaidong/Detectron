# -*- coding: UTF-8 -*-
import os
import pdb
import numpy as np
# from PIL import Image
# from skimage import io, color, filters
from cv2 import *
from matplotlib import pyplot as plt
import xml.dom.minidom as xmldom
from metrics import calc_rate, calc_auc
from utils_xml import get_bbox_xml
import argparse, sys

def get_r1_csv(res_pth):
    res_dct = {}
    with open(res_pth, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            items = lines[i].strip().split('\t')
            res_dct[items[0]] = float(items[1])
    return res_dct


def get_r1_ans(xml_dir):
    xml_dct = {}
    for item in os.listdir(xml_dir):
        sub_dir = os.path.join(xml_dir, item)
        if os.path.isdir(sub_dir):
            for temp in os.listdir(sub_dir):
                xml_pth = os.path.join(sub_dir, temp)
                if temp[-3:] == 'xml':
                    jpg_pth = temp[0:-3] + 'jpg'
                    if jpg_pth not in xml_dct:
                        xml_dct[jpg_pth] = 1
    return xml_dct


def get_r2_csv(res_pth):
    res_dct = {}
    with open(res_pth, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            items = lines[i].strip().split(',')
            parts = items[0].split('|')
            if parts[1] not in res_dct:
                res_dct[parts[1]] = {}
            # [defect_class][image_name] = probability
            res_dct[parts[1]][parts[0]] = float(items[1].strip())
    return res_dct


def get_r2_ans(xml_dir, cls_dct):
    xml_dct = {}
    for item in os.listdir(xml_dir):
        item = item.encode('utf-8')
        sub_dir = os.path.join(xml_dir, item)
        if os.path.isdir(sub_dir):
            def_cls = cls_dct[item].decode('utf-8')
            if def_cls not in xml_dct:
                xml_dct[def_cls] = {}
            for temp in os.listdir(sub_dir):
                xml_pth = os.path.join(sub_dir, temp)
                if temp[-3:] == 'xml':
                    jpg_pth = temp[0:-3] + 'jpg'
                    if jpg_pth not in xml_dct:
                        xml_dct[def_cls][jpg_pth] = 1
    return xml_dct

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    parser.add_argument(
        '--val_csv',
        dest='val_csv',
        help='csv file dir to val',
        default='merged.csv',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    xml_dir = '/home/ubuntu/02data/02cloth/XuelangFiles/xuelang_round1_answer_b/'
    xml_dct = get_r1_ans(xml_dir)

    res_pth = args.val_csv
    res_dct = get_r2_csv(res_pth)

    # step 2: get the provided result
    '''
    cls_dct['norm'] = set(['正常'])
    cls_dct['defect_1'] = set(['扎洞'])
    cls_dct['defect_2'] = set(['毛斑'])
    cls_dct['defect_3'] = set(['擦洞'])
    cls_dct['defect_4'] = set(['毛洞'])
    cls_dct['defect_5'] = set(['织稀'])
    cls_dct['defect_6'] = set(['吊经'])
    cls_dct['defect_7'] = set(['缺经'])
    cls_dct['defect_8'] = set(['跳花'])
    cls_dct['defect_9'] = set(['黄渍', '污渍', '油渍'])
    others = ['剪洞', '吊弓', '薄段', '扎梳', '缺纬', '错纱', '弓纱']
    others += ['回边', '破洞', '楞断', '织入', '粗纱', '错经', '夹纱']
    others += ['擦伤', '线印', '厚段', '破边', '边扎洞', '换纱印', '纬粗纱']
    cls_dct['defect_10'] = set(others)
    '''
    cn_dct = {}
    cn_dct['zhengchang'] = 'norm'
    cn_dct['jiama'] = 'defect_10'
    cn_dct['zhiru'] = 'defect_10'
    cn_dct['shuangsha'] = 'defect_10'
    cn_dct['baoduan'] = 'defect_10'
    cn_dct['jiasha'] = 'defect_10'
    cn_dct['cashang'] = 'defect_10'
    cn_dct['zhuwang'] = 'defect_10'
    cn_dct['qianjie'] = 'defect_10'
    cn_dct['zhixi'] = 'defect_5'
    cn_dct['camao'] = 'defect_10'
    cn_dct['jiandong'] = 'defect_10'
    cn_dct['tiaohua'] = 'defect_8'
    cn_dct['youwu'] = 'defect_9'
    cn_dct['xianyin'] = 'defect_10'
    cn_dct['zhasha'] = 'defect_10'
    cn_dct['zhashu'] = 'defect_10'
    cn_dct['cujie'] = 'defect_10'
    cn_dct['xiuyin'] = 'defect_10'
    cn_dct['diaowei'] = 'defect_10'
    cn_dct['cuosha'] = 'defect_10'
    cn_dct['cuojing'] = 'defect_10'
    cn_dct['huibian'] = 'defect_10'
    cn_dct['diaogong'] = 'defect_10'
    cn_dct['quewei'] = 'defect_10'
    cn_dct['maodong'] = 'defect_4'
    cn_dct['pobian'] = 'defect_10'
    cn_dct['youzi'] = 'defect_9'
    cn_dct['erduo'] = 'defect_10'
    cn_dct['huangzi'] = 'defect_9'
    cn_dct['biandong'] = 'defect_10'
    cn_dct['jiedong'] = 'defect_10'
    cn_dct['gongsha'] = 'defect_10'
    cn_dct['maoli'] = 'defect_10'
    cn_dct['jinsha'] = 'defect_10'
    cn_dct['maoban'] = 'defect_2'
    cn_dct['houduan'] = 'defect_10'
    cn_dct['wuzi'] = 'defect_9'
    cn_dct['quejing'] = 'defect_7'
    cn_dct['lengduan'] = 'defect_10'
    cn_dct['cadong'] = 'defect_3'
    cn_dct['zhadong'] = 'defect_1'
    cn_dct['podong'] = 'defect_10'
    cn_dct['cusha'] = 'defect_10'
    cn_dct['diaojing'] = 'defect_6'
    cn_dct['bianbaiyin'] = 'defect_10'
    cn_dct['houbaoduan'] = 'defect_10'
    cn_dct['zhengniyin'] = 'defect_10'
    cn_dct['mingqianxian'] = 'defect_10'
    cn_dct['weicusha'] = 'defect_10'
    cn_dct['huanshayin'] = 'defect_10'
    cn_dct['jingcusha'] = 'defect_10'
    cn_dct['bianquejing'] = 'defect_10'
    cn_dct['shaomaohen'] = 'defect_10'
    cn_dct['bianzhenyan'] = 'defect_10'
    cn_dct['weicujie'] = 'defect_10'
    cn_dct['bianzhadong'] = 'defect_10'
    cn_dct['jingtiaohua'] = 'defect_10'
    cn_dct['bianquewei'] = 'defect_10'
    cn_dct['yuanzhubiyin'] = 'defect_10'
    xml_dct = get_r2_ans(xml_dir, cn_dct)
    ook_lst = []
    for i in range(1, 11):
        ook_lst += xml_dct['defect_{0}'.format(i)].keys()
    ook_set = set(ook_lst)
    for jpg in res_dct['norm'].keys():
        if jpg not in set(ook_set):
            xml_dct['norm'][jpg] = 1

    # step 3: get the probs and labels
    auc = 0.0
    aps = 0.0
    score_aps = []

    fig = plt.figure(frameon=False)
    py_dct = {     'norm':'zhengchang', 'defect_1':'zhadong', 'defect_2':'maoban',   'defect_3':'cadong',
               'defect_4':'maodong',    'defect_5':'zhixi',   'defect_6':'diaojing', 'defect_7':'quejing',
               'defect_8':'tiaohua',    'defect_9':'youwuzi', 'defect_10':'qita'}
    for def_cls in res_dct.keys():
        probs, labels = [], []
        tmp_res_dct = res_dct[def_cls]
        tmp_xml_dct = xml_dct[def_cls]
        for key in tmp_res_dct.keys():
            probs.append(tmp_res_dct[key])
            labels.append(1 if key in tmp_xml_dct else -1)
        tpr, fpr, pre = calc_rate(probs, labels)

        if def_cls == 'norm':
            val = calc_auc(fpr, tpr)
            #print('auc_{0}={1}'.format(def_cls, val))
            auc = val
            plt.subplot(3, 4, 1)
            plt.plot(fpr, tpr)
        else:
            val = calc_auc(tpr, pre)
            #print('ap_{0}(1)={2}'.format(def_cls, py_dct[def_cls], val))
            aps += val
            plt.subplot(3, 4, int(def_cls[7:]) + 1)
            plt.plot(tpr, pre)
            score_aps.append([py_dct[def_cls], val])
        plt.title('{2}_{0}={1}'.format(py_dct[def_cls], round(val, 3), 'auc' if def_cls=='norm' else 'ap'))

    sorted_aps = sorted(score_aps, key=lambda x:x[1], reverse=True)
    score_aps.append(['auc', auc])
    print score_aps

    aps *= 0.1
    score = auc * 0.7 + aps * 0.3
    plt.subplot(3, 4, 12)
    plt.title('{0}*0.7+{1}*0.3={2}'.format(round(auc, 3), round(aps, 3), round(score, 3)))
    print(score)
    plt.show()
    fig.savefig('{}'.format(args.val_csv.replace('csv', 'png')))
    plt.close('all')
    # with open('eval_{}_res.csv'.format(args.val_csv[:-4])), 'w') as f:
    #     f.write('auc, {}\n'.format(auc))
    #     for items in score_aps:
    #         f.write('{},{}\n'.format(items[0], items[1]))
    #     f.write('score, {}\n'.format(score))
    # pdb.set_trace()
    # exit(0)

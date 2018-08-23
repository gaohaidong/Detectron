# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_traffic_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'car'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_cloth_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__',
        # round1 47 cls
        'lengduan', 'tiaohua', 'wuzi', 'diaojing', 'jinsha', 'jiandong', 'xiuyin', 'huibian', 'quewei', 'maoban',
        'huangzi', 'zhashu', 'qianjie', 'camao', 'bianzhenyan', 'bianquewei', 'maodong', 'zhengneyin', 'erduo',
        'bianzhadong', 'cadong', 'zhiru', 'maoli', 'zhixi', 'pobian', 'diaogong', 'diaowei', 'xianyin', 'quejing',
        'podong', 'weicusha', 'jiedong', 'bianbaiyin', 'cashang', 'cusha', 'jingtiaohua', 'jingcusha', 'zhadong',
        'gongsha', 'zhasha', 'youzi', 'houbaoduan', 'bianquejing', 'mingqianxian', 'jiama', 'houduan'
        # round2 60 cls
        # 'jiama', 'bianbaiyin', 'houbaoduan', 'zhengneyin', 'zhiru', 'mingqianxian', 'shuangsha', 'baoduan',
        # 'bianquewei', 'jiasha', 'cashang', 'weicusha', 'zhuwang', 'huanshayin', 'qianjie', 'zhixi', 'camao', 'jiandong',
        # 'tiaohua', 'yuanzhubiyin', 'youwu', 'xianyin', 'zhasha', 'zhashu', 'jingcusha', 'cujie', 'xiuyin', 'diaowei',
        # 'cuosha', 'cuojing', 'huibian', 'bianquejing', 'shaomaohen', 'diaogong', 'quewei', 'bianzhenyan', 'maodong',
        # 'pobian', 'weicujie', 'bianzhadong', 'jingtiaohua', 'youzi', 'erduo', 'huangzi', 'biandong', 'jiedong',
        # 'gongsha', 'maoli', 'jinsha', 'maoban', 'houduan', 'wuzi', 'quejing', 'lengduan', 'cadong', 'zhadong', 'podong',
        # 'cusha', 'diaojing'
        # merge 54 cls
        # 'jiama', 'bianbaiyin', 'houbaoduan', 'zhengneyin', 'zhiru', 'mingqianxian', 'shuangsha', 'baoduan',
        # 'jiasha', 'cashang', 'weicusha', 'zhuwang', 'huanshayin', 'qianjie', 'zhixi', 'camao', 'jiandong',
        # 'tiaohua', 'yuanzhubiyin', 'youwu', 'xianyin', 'zhasha', 'zhashu', 'jingcusha', 'cujie', 'xiuyin', 'diaowei',
        # 'cuosha', 'cuojing', 'huibian', 'shaomaohen', 'diaogong', 'quewei', 'bianzhenyan', 'maodong',
        # 'pobian', 'weicujie', 'wuzi', 'erduo', 'biandong', 'jiedong',
        # 'gongsha', 'maoli', 'jinsha', 'maoban', 'houduan', 'quejing', 'lengduan', 'cadong', 'zhadong', 'podong',
        # 'cusha', 'diaojing'

        ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds
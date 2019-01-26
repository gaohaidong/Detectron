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
        '__background__', 'huibian','qianjie','bianzhadong','quewei','diaojing','cusha','xianyin','quejing','diaogong','zhixi','pobian','lengduan','cashang','bianquejing','maoban','wuzi','mingqianxian','houduan','bianzhenyan','gongsha','zhengneyin','cadong','jiandong','jiama','jingtiaohua','maodong','zhiru','youzi','camao','zhadong','tiaohua','diaowei','houbaoduan','xiuyin','bianquewei','erduo','jiedong','maoli','podong','huangzi','jinsha','zhashu','zhasha','bianbaiyin','jingcusha','weicusha'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_steel_dataset():
    ds = AttrDict()
    classes = [
        '__background__', 'steel'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_hanzi_dataset():
    ds = AttrDict()
    classes = [
        '__background__', 'hanzi'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds
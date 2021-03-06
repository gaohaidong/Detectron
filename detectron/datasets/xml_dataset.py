# -*- coding: utf-8 -*-
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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import detectron.utils.env as envu
envu.set_up_matplotlib()
# COCO API
from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import _ANN_FN
from detectron.datasets.dataset_catalog import _DATASETS
from detectron.datasets.dataset_catalog import _IM_DIR
from detectron.datasets.dataset_catalog import _ANN_DIR
from detectron.datasets.dataset_catalog import _IM_PREFIX
from detectron.utils.timer import Timer
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils
import detectron.datasets.dummy_datasets as dummy_datasets

logger = logging.getLogger(__name__)


def get_polygon_from_obj(objs):
    polygons = []
    for obj in objs:
        points = obj.findall('point')
        poly = []
        for pp in points:
            pos = map(int, pp.text.split(','))
            poly.append(pos[0])
            poly.append(pos[1])
        polygons.append(poly)
    return polygons


def get_segmentation_area(segms):
    import cv2
    def calc_area(segm):
        points = segm.findall('point')
        if len(points) == 0:
            return 0
        contour = []
        for point in points:
            contour.append(map(int, point.text.split(',')))
        contour = np.asarray(contour)
        area = cv2.contourArea(contour)
        return area

    polygon_area = 0
    if len(segms) == 0:
        return 0
    for segm in segms:
        polygon_area += calc_area(segm)
    return polygon_area

class XMLDataset(object):
    """A class representing a xml dataset."""

    def __init__(self, name):
        assert name in _DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(_DATASETS[name][_IM_DIR]), \
            'Image directory \'{}\' not found'.format(_DATASETS[name][_IM_DIR])
        assert os.path.exists(_DATASETS[name][_ANN_FN]), \
            'Annotation file \'{}\' not found'.format(_DATASETS[name][_ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = _DATASETS[name][_IM_DIR]
        self.image_prefix = (
            '' if _IM_PREFIX not in _DATASETS[name] else _DATASETS[name][_IM_PREFIX]
        )

        self.debug_timer = Timer()
        # Set up dataset classes
        if 'traffic' in self.name:
            dummy_dataset = dummy_datasets.get_traffic_dataset()
        elif 'bupi' in self.name:
            dummy_dataset = dummy_datasets.get_cloth_dataset()
        elif 'steel' in self.name:
            dummy_dataset = dummy_datasets.get_steel_dataset()
        elif 'hanzi' in self.name:
            dummy_dataset = dummy_datasets.get_hanzi_dataset()
        categories = dummy_dataset.classes.values()
        category_ids = range(len(categories))
        logger.info('categories\t{}'.format(categories))
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = categories  
        self.num_classes = len(self.classes)
        self.keypoints = None
        # self._init_keypoints()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))


    # load_image_info(self)
    def load_image_set_index(self):
        image_set_index_file = os.path.join(_DATASETS[self.name][_ANN_FN])
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        logger.info('image_set_index_file \t{}'.format(image_set_index_file))
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
            logger.info('number of images\t{}'.format(len(image_set_index)))
            return image_set_index

    def image_path_from_index(self,index):
        image_path = os.path.join(_DATASETS[self.name][_IM_DIR], index + '.jpg')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=2,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        cache_path = cfg.TRAIN.DATASET_CACHE_PATH
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        cache_file = os.path.join(cache_path, self.name + '.pkl')
        if os.path.isfile(cache_file):
            roidb = pickle.load(open(cache_file,'rb'))
            logger.info('load roidb from %s' % cache_file)
            return roidb

        image_ids = self.load_image_set_index()
        roidb = [{'index': index} for index in image_ids]

        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry in roidb:
                self._add_gt_annotations(entry)
            logger.debug(
                '_add_gt_annotations took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        with open(cache_file, 'wb') as fw:
            pickle.dump(roidb, fw)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        entry['image'] = self.image_path_from_index(entry['index'])
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]



    def load_objs_from_index(self, entry):
        import xml.etree.ElementTree as ET
        index = entry['index']
        tree = ET.parse(os.path.join(_DATASETS[self.name][_ANN_DIR], index + '.xml'))
        size = tree.find('size')
        entry['width'] = int(size.find('width').text)
        entry['height'] = int(size.find('height').text)
        objs = tree.findall('object')

        valid_objs = []
        for obj in objs:
            cls_name = obj.find('name').text.lower()
            if cls_name not in self.classes:
                logger.info('obj class {} not in classname list {}'.format(cls_name, self.classes))
                continue

            tmp_obj = dict()
            tmp_obj['cls_name'] = cls_name
            bndbox = obj.find('bndbox')
            x1, y1, x2, y2 = map(int, [xx.text for xx in bndbox])
            tmp_obj['bbox'] = [x1, y1, x2, y2]

            segms = obj.find('segmentation')
            if segms == None or len(segms) == 0:
                polygons = None
                area = 0
            else:
                polygons = get_polygon_from_obj(segms)
                area = get_segmentation_area(segms)

            tmp_obj['area'] = area
            tmp_obj['segmentation'] = polygons

            tmp_obj['iscrowd'] = False
            valid_objs.append(tmp_obj)
        return valid_objs

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        objs = self.load_objs_from_index(entry)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue

            x1, y1, x2, y2 = obj['bbox']
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )

            if x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)
        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            if obj['cls_name'] == 'ignore':
                cls = -1
            else:
                cls = self._class_to_ind[obj['cls_name']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = 0
            is_crowd[ix] = obj['iscrowd']


            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        raise NotImplementedError

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

'''
Date: 2022-01-11 22:46:33
Company: Luokung Technology Corp.
LastEditors: Will Cheng Yong chengyong@pku.edu.cn
LastEditTime: 2022-09-29 16:03:23
'''
import os
from enum import Enum
import time

import torch
import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from tools import logger

class Predictor(DefaultPredictor):
    def __init__(self, model_cfg, model_params, score_thresh_hold=0.2):
        cfg = get_cfg()
        cfg.merge_from_file(model_cfg)
        cfg.MODEL.WEIGHTS = model_params
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_hold
        super(Predictor, self).__init__(cfg)

class ClassName(Enum):
    SIDE = 0
    CENTER = 1
    SIDE_CUT = 2

class Direction(Enum):
    RIGHT     = 'right'
    LEFT      = 'left'
    CENTER    = 'center'
    RIGHT_CUT = 'right_cut'
    LEFT_CUT  = 'left_cut'
    UNKNOWN   = 'unknown'
    def __str__(self):
        return str(self.value)

def get_valid_rear(rear_p0, rear_p1):
    x0, y0 = rear_p0
    x1, y1 = rear_p1
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))

def get_direction(rear_p0, rear_p1, side_p0, side_p1, cls_name: ClassName=ClassName.SIDE):
    if cls_name == ClassName.SIDE:
        if side_p0[0] > rear_p1[0] and side_p1[0] > rear_p1[0]:
            direction = Direction.RIGHT
        elif side_p0[0] < rear_p0[0] and side_p1[0] < rear_p0[0]:
            direction = Direction.LEFT
        else:
            direction = Direction.UNKNOWN

        if direction in [Direction.RIGHT, Direction.LEFT]:
            if abs(rear_p1[0] - rear_p0[0]) < 10:
                if direction == Direction.RIGHT:
                    direction = Direction.RIGHT_CUT
                else:
                    direction = Direction.LEFT_CUT

    elif cls_name == ClassName.CENTER:
        direction = Direction.CENTER
    elif cls_name == ClassName.SIDE_CUT:
        if side_p0[0] > rear_p1[0] and side_p1[0] > rear_p1[0]:
            direction = Direction.RIGHT_CUT
        elif side_p0[0] < rear_p0[0] and side_p1[0] < rear_p0[0]:
            direction = Direction.LEFT_CUT
        else:
            direction = Direction.UNKNOWN
    else:
        direction = Direction.UNKNOWN
    return direction

def get_side_pts(side_p0, side_p1):
    # should share the x
    x_axis = (side_p0[0] + side_p1[0]) // 2
    return ((x_axis, side_p0[1]), (x_axis, side_p1[1]))

class Cuboid:
    def __init__(self, cls_name: ClassName, rear_p0, rear_p1, side_p0, side_p1, score=0.0):
        self.rear_p0 = rear_p0
        self.rear_p1 = rear_p1
        self.side_p0 = side_p0
        self.side_p1 = side_p1
        self.score = float(score)
        self.direction = get_direction(rear_p0, rear_p1, side_p0, side_p1, cls_name)
        self.front = self._init_face()
        self.back = self._init_face()
        self.middle_points = self._init_middle_line()
        self.get_eight_pts()
        self.scaled_pts = []

    def set_scaled_pts(self, scaled_pts):
        # assert len(scaled_pts) == 20 # 10 pts
        self.scaled_pts = scaled_pts

    def get_scaled_pts(self):
        # assert len(self.scaled_pts) == 20 # 10 pts
        return self.scaled_pts

    def is_valid_cuboid(self, tolerance=10):
        if self.front['top_left']:
            if abs(self.front['top_left'][0] - self.front['top_right'][0]) < tolerance:
                print('error')
                return False
            if abs(self.front['top_left'][1] - self.front['btm_left'][1]) < tolerance:
                print('error')
                return False
            return True
        else:
            print('error')
            return False

    def _init_face(self):
        return {
            "top_left": None,
            "btm_left": None,
            "top_right": None,
            "btm_right": None
        }

    def _init_middle_line(self):
        rear_x0, rear_y0 = self.rear_p0
        rear_x1, rear_y1 = self.rear_p1
        width = abs(rear_x0 - rear_x1) // 2
        x = rear_x0 + width
        return ((x, rear_y0), (x, rear_y1))

    def _flatten(self, pts):
        return [int(element) for tupl in pts for element in tupl]

    def get_eight_pts(self):
        rear_x0, rear_y0 = self.rear_p0
        rear_x1, rear_y1 = self.rear_p1
        side_x0, side_y0 = self.side_p0
        side_x1, side_y1 = self.side_p1
        back_width = abs(side_y0 - side_y1)
        if self.direction in [Direction.RIGHT, Direction.LEFT, Direction.RIGHT_CUT, Direction.LEFT_CUT, Direction.CENTER]:
            self.front['top_left'] = (rear_x0, rear_y0)
            self.front['btm_left'] = (rear_x0, rear_y1)
            self.front['top_right'] = (rear_x1, rear_y0)
            self.front['btm_right'] = (rear_x1, rear_y1)
            if self.direction in [Direction.RIGHT, Direction.RIGHT_CUT]:
                self.back['top_right'] = (side_x0, side_y0)
                self.back['btm_right'] = (side_x1, side_y1)
                self.back['top_left'] = ((side_x0 - back_width), side_y0)
                self.back['btm_left'] = ((side_x0 - back_width), side_y1)
            elif self.direction in [Direction.LEFT, Direction.LEFT_CUT]:
                self.back['top_left'] = (side_x0, side_y0)
                self.back['btm_left'] = (side_x1, side_y1)
                self.back['top_right'] = ((side_x0 + back_width), side_y0)
                self.back['btm_right'] = ((side_x0 + back_width), side_y1)

    def to_list(self):
        if self.direction in [Direction.LEFT]:
            return self._flatten([self.front['top_left'], self.front['btm_left'], self.front['top_right'], self.front['btm_right'],
                    self.back['top_left'], self.back['btm_left'],
                    self.middle_points[0], self.middle_points[1]])
        elif self.direction in [Direction.RIGHT]:
            return self._flatten([self.front['top_left'], self.front['btm_left'], self.front['top_right'], self.front['btm_right'],
                    self.back['top_right'], self.back['btm_right'],
                    self.middle_points[0], self.middle_points[1]])
        elif self.direction in [Direction.CENTER]:
            return self._flatten([self.front['top_left'], self.front['btm_left'], self.front['top_right'], self.front['btm_right'],
                    self.middle_points[0], self.middle_points[1]])
        return None

def do_inference(predictor, image, yolo_score=0.0):
    start = time.time()
    results = []
    predictions = predictor(image)
    predictions = predictions["instances"].to(torch.device("cpu"))
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints.tolist() if predictions.has("pred_keypoints") else None
    # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

    for kpts, score, cls_id in zip(keypoints, scores, classes):
        rear_p0, rear_p1, side_p0, side_p1 = kpts
        rear_p0 = np.int0(rear_p0[:2])
        rear_p1 = np.int0(rear_p1[:2])
        side_p0 = np.int0(side_p0[:2])
        side_p1 = np.int0(side_p1[:2])
        rear_p0, rear_p1 = get_valid_rear(rear_p0, rear_p1)
        side_p0, side_p1 = get_side_pts(side_p0, side_p1)

        if cls_id == 0:
            cls_name = ClassName.SIDE
        elif cls_id == 1:
            cls_name = ClassName.CENTER
        else:
            cls_name = ClassName.SIDE_CUT
        results.append(Cuboid(cls_name, rear_p0, rear_p1, side_p0, side_p1, yolo_score))
    end = time.time()
    logger.info("[Inference] Get {} objects, takes {} seconds.".format(len(results), end - start))
    return results
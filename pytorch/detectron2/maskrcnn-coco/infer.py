'''
Date: 2021-12-15 17:07:09
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 23:00:09
'''
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from skimage import measure

from utils import logger

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0, slice_num=5):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        contour = contour[::slice_num]
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def init_model(cfg_file, score_thres=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    predictor = DefaultPredictor(cfg)
    logger.info("Init model successfully.")
    return predictor

def do_inference(predictor, image):
    outputs = predictor(image)
    fields = outputs['instances'].get_fields()
    scores = fields['scores'].cpu().tolist()
    pred_masks = fields['pred_masks'].cpu()
    pred_classes = fields['pred_classes'].cpu().tolist()
    assert len(scores) == len(pred_masks) == len(pred_classes)

    polygons = []
    for i in range(pred_masks.shape[0]):
        polygons.append(np.int0(binary_mask_to_polygon(pred_masks[i, :, :], slice_num=7)[0]).tolist())
    logger.info("Do inference.")
    return scores, pred_classes, polygons

import json
import logging.config
from collections import defaultdict

import cv2
import numpy as np
import yaml

logging.config.fileConfig("/opt/nuclio/logger_cfg.cfg")
logger = logging.getLogger('Admin_Client')

def parse_yaml(yaml_file):
    try:
        with open(yaml_file, "r") as stream:
            logger.info("Parse {}".format(yaml_file))
            return yaml.safe_load(stream)
    except Exception as e:
        # logger.error(e)
        raise ValueError(e)

def list_to_mapping(label_list):
    mapping = defaultdict(int)
    for item in label_list:
        mapping[int(item["id"])] = str(item["name"])
    return mapping

def formatter(scores, pred_classes, pred_boxes, label_mapping, threshold=0.5):
    results = []
    for score, cls_id, box in zip(scores, pred_classes, pred_boxes):
        if score < threshold:
            continue

        x1, y1, x2, y2 = np.int0(box.tolist())
        results.append({
            "confidence": score,
            "label": label_mapping[cls_id+1], # cause' the label_id start from 1
            "points": [int(x1), int(y1), int(x2), int(y2)],
            "type": "rectangle"
        })
    return results

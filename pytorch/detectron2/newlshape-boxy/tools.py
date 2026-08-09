'''
Date: 2022-01-11 22:49:28
Company: Luokung Technology Corp.
LastEditors: Will Cheng Yong chengyong@pku.edu.cn
LastEditTime: 2022-09-29 16:04:47
'''
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

def formatter(cuboids, label_mapping, threshold=0.8):
    results = []
    for cuboid in cuboids:
        cls_id = 0 # only one class in this model
        cuboid_pts = cuboid.get_scaled_pts()
        if cuboid.score < threshold or cuboid_pts is None:
            continue

        results.append({
            "confidence": cuboid.score,
            "label": label_mapping[cls_id+1], # cause' the label_id start from 1
            "points": cuboid_pts,
            "type": "newlshape"
        })
    return results

'''
Author: Will Cheng Yong chengyong@pku.edu.cn
Date: 2022-10-02 16:18:53
LastEditors: Will Cheng Yong chengyong@pku.edu.cn
LastEditTime: 2022-10-08 16:28:58
FilePath: /lkdi-auto-annotation/detector/bbox/open-pcdet-kitti/utils.py
Description:

Copyright (c) 2022 by Will Cheng Yong chengyong@pku.edu.cn, All Rights Reserved.
'''
import logging.config
from collections import defaultdict

import yaml

logging.config.fileConfig("/opt/nuclio/logger_cfg.cfg")
logger = logging.getLogger('Admin_Client')

def parse_yaml(yaml_file):
    try:
        with open(yaml_file, "r") as stream:
            logger.info("Parse {}".format(yaml_file))
            return yaml.safe_load(stream)
    except Exception as e:
        raise ValueError(e)

def list_to_mapping(label_list):
    mapping = defaultdict(int)
    for item in label_list:
        mapping[int(item["id"])] = str(item["name"])
    return mapping

def formatter(boxes, scores, labels, label_mapping, threshold=0.5):
    results = []
    for box, score, cls_id in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x, y, z, l, w, h, ry = box
        results.append({
            "confidence": score,
            "label": label_mapping[cls_id], # already start from 1
            "points": [float(x), float(y), float(z), float(l), float(w), float(h), float(ry)],
            "type": "cuboid"
        })

    return results
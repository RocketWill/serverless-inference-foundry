'''
Date: 2022-01-11 22:49:28
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:48:25
'''
import json
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
        # logger.error(e)
        raise ValueError(e)

def list_to_mapping(label_list):
    mapping = defaultdict(int)
    for item in label_list:
        mapping[int(item["id"])] = str(item["name"])
    return mapping

def formatter(lanes, label_mapping):
    results = []
    for lane in lanes:
        cls_id = 0 # only one class in this model
        results.append({
            "confidence": 0, # no score for this function
            "label": label_mapping[cls_id+1], # cause' the label_id start from 1
            "points": lane,
            "type": "polyline"
        })
    return results

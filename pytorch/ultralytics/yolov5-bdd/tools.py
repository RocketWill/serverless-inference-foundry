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
        raise ValueError(e)

def list_to_mapping(label_list):
    mapping = defaultdict(int)
    for item in label_list:
        mapping[int(item["id"])] = str(item["name"])
    return mapping

def formatter(dets, label_mapping, threshold=0.5):
    results = []
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < threshold:
            continue
        results.append({
            "confidence": conf,
            "label": label_mapping[cls_id+1], # cause' the label_id start from 1
            "points": [int(x1), int(y1), int(x2), int(y2)],
            "type": "rectangle"
        })
    return results
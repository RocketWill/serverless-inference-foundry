'''
Date: 2022-01-20 19:50:46
Company: Luokung Technology Corp.
LastEditors: Will Cheng Yong
LastEditTime: 2022-01-22 00:24:15
'''
import json
import logging.config
from collections import defaultdict
from enum import Enum

import yaml

logging.config.fileConfig("/opt/nuclio/logger_cfg.cfg")
logger = logging.getLogger('Admin_Client')

class SIGN(Enum):
    LOW_SPEED_LIMIT   = 0 # 低限速路牌
    WEIGHT_LIMIT_AXLE = 1 # 矢量限制轴重标志牌
    HEIGHT_LIMIT      = 2 # 限高标志牌
    WIDTH_LIMIT       = 3 # 限宽标志牌
    HIGH_SPEED_LIMIT  = 4 # 高限速标志牌
    WEIGHT_LIMIT      = 5 # 限重标志牌
    LIFT_SPEED_LIMIT  = 6 # 解除限速标志牌
    BLUE              = 7 # 弱规律-蓝色指示标志牌
    YELLOW            = 8 # 弱规律-黄色三角形警告标志牌
    CIRCLE            = 9 # 弱规律-圆形禁止标志牌
    OTHER             = 10 # 其它标志牌

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

def load_names(name_file):
    with open(name_file) as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def get_sign_rough_category(name):
    if len(name) < 2:
        cate = SIGN.OTHER
    prefix = name[:2]
    if prefix == "il":
        cate = SIGN.LOW_SPEED_LIMIT
    elif prefix == "pa":
        cate = SIGN.WEIGHT_LIMIT_AXLE
    elif prefix == "ph":
        cate = SIGN.HEIGHT_LIMIT
    elif prefix == "pw":
        cate = SIGN.WIDTH_LIMIT
    elif prefix == "pl":
        cate = SIGN.HIGH_SPEED_LIMIT
    elif prefix == "pm":
        cate = SIGN.WEIGHT_LIMIT
    elif prefix == "pr":
        cate = SIGN.LIFT_SPEED_LIMIT
    elif prefix in ["ip", "pb", "pc", "pd", "pe", "pg", "pn", "ps"]:
        cate = SIGN.OTHER
    elif prefix[0] == "i":
        cate = SIGN.BLUE
    elif prefix[0] == "w":
        cate = SIGN.YELLOW
    elif prefix[0] == "p":
        cate = SIGN.CIRCLE
    else:
        cate = SIGN.OTHER
    return cate

def formatter(
    dets,
    label_mapping,
    name_file="/opt/nuclio/cfg/roadSign213.names",
    threshold=0.5
):
    names = load_names(name_file)
    results = []
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        sign_name = names[int(cls_id)]
        cat_id = get_sign_rough_category(sign_name).value
        if conf < threshold:
            continue
        results.append({
            "confidence": conf,
            "label": label_mapping[cat_id+1], # cause' the label_id start from 1
            "points": [int(x1), int(y1), int(x2), int(y2)],
            "type": "rectangle"
        })
    return results
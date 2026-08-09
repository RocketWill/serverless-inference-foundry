'''
Author: ChengYong chengy@luokung.com
Date: 2022-10-11 22:58:22
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 23:06:50
FilePath: /serverless/pytorch/detectron2/maskrcnn-coco/utils.py
Description:

Copyright (c) 2022 by ChengYong chengy@luokung.com, All Rights Reserved.
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

def formatter(scores, pred_classes, polygons, label_mapping, threshold=0.5):
    results = []
    for score, cls_id, polygon in zip(scores, pred_classes, polygons):
        if score < threshold:
            continue
        results.append({
            "confidence": score,
            "label": label_mapping[cls_id+1], # cause' the label_id start from 1
            "points": polygon,
            "type": "polygon"
        })
    return results

class RequestParser:
    def parse_boundary(self, content_type):
        if isinstance(content_type, str):
            boundary = '--' + content_type.replace('multipart/form-data; boundary=', '')
            boundary = boundary.encode()
        elif isinstance(content_type, bytes):
            boundary = b'--' + content_type.replace(b'multipart/form-data; boundary=', b'')
        else:
            raise ValueError("Invalid content_type type.")
        return boundary

    def parse_image_meta(self, image_meta_data):
        image_meta_data = image_meta_data.split(b'\r\n\r\n')
        image_meta_data = [ele for ele in image_meta_data if not b'Content-Disposition' in ele]
        assert len(image_meta_data) == 1, "Length of image meta data contents should be 1."
        image_meta_data = image_meta_data[0].replace(b'\r\n', b'')
        return json.loads(image_meta_data.decode('utf-8'))

    def parse_image_data(self, image_data, image_meta):
        image_info = image_meta["image"]
        h, w, _ = np.int0([image_info["height"], image_info["width"], image_info["channel"]])
        image_data = image_data.split(b'\r\n\r\n')
    image_data = [ele for ele in image_data if ele != b'' and b'form-data' not in ele and b'stream' not in ele]
        assert len(image_data) == 1, "Length of image data data contents should be 1."
        image_data = image_data[0].rsplit(b'\r\n', 1)[0] # cause' there are many '\r\n' in image data, so we just remove the last
        flag = False
        image = None
        for c in [4, 3, 1]:
            if flag: break
            try:
                image = np.fromstring(image_data, np.uint8).reshape(h, w, c)
                flag = True
            except Exception as e:
                logger.error(str(e))
                print(e)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB

    def get_image_meta_and_data(self, event):
        content_type = event.content_type
        boundary = self.parse_boundary(content_type)
        form_data = event.body
        content = form_data.split(boundary)
        content = [ele for ele in content if b'form-data' in ele]
        assert len(content) == 2, "Length of content not match."
        meta_data, image_data = content[0], content[1]
        image_meta = self.parse_image_meta(meta_data)
        image = self.parse_image_data(image_data, image_meta)
        return image_meta, image

    def __call__(self, event):
        return self.get_image_meta_and_data(event)

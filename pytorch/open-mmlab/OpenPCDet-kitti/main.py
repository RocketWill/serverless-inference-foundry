'''
Author: Will Cheng Yong chengyong@pku.edu.cn
Date: 2022-10-02 16:18:53
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:23:27
FilePath: /serverless/pytorch/open-mmlab/OpenPCDet-kitti/main.py
Description:

Copyright (c) 2022 by Will Cheng Yong chengyong@pku.edu.cn, All Rights Reserved.
'''
import os
import json
import base64

import nuclio_sdk
import numpy as np

from infer import InferDataset, init_model, do_inference
from utils import logger, parse_yaml, list_to_mapping, formatter

headers={
    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
    'Access-Control-Allow-Credentials':'false',
    'Access-Control-Allow-Origin': "*",
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
}

def init_context(context):
    context.logger.info("Init context...  0%")
    weights = os.path.join("/opt/nuclio/weights", os.environ['WEIGHTS_FILE'])
    config_file = "/opt/nuclio/cfgs/kitti_models/PartA2_free.yaml"
    classes_info = json.loads(parse_yaml("/opt/nuclio/function.yaml")["metadata"]["annotations"]["spec"])
    cfg, model = init_model(weights, config_file)
    model_info = {
        "model": model,
        "cfg": cfg,
        "classes": classes_info,
        "class_map": list_to_mapping(classes_info),
    }
    context.model_info = model_info
    context.logger.info("Init context...100%")

def handler(context: nuclio_sdk.Context, event: nuclio_sdk.Event):
    path = event.path

    if event.method in [b"OPTIONS", "OPTIONS"]:
        return context.Response(body=None,
                                headers=headers,
                                status_code=200)

    if event.method in [b"GET", "GET"]:
        try:
            response_body = {"code": 200, "message": "ok", "result": []}
            response_body["result"] = context.model_info["classes"]
            return context.Response(body=response_body,
                                    headers=headers,
                                    content_type='application/json',
                                    status_code=response_body["code"])
        except Exception as e:
            logger.error(e)

    if event.method in [b"POST", "POST"]:
        response_body = {"code": 200, "message": "ok", "result": []}
        try:
            data = event.body
            buf = base64.decodebytes(data["image"].encode())
            points = np.frombuffer(buf, dtype=np.float32).reshape(-1, 4)

            pcd = InferDataset(
                context.model_info["cfg"].DATA_CONFIG,
                context.model_info["cfg"].CLASS_NAMES,
                points,
            )

            threshold = float(data.get("threshold", 0.5))
            boxes, scores, labels = do_inference(pcd, context.model_info["model"])
            response_body["result"] = formatter(boxes, scores, labels, context.model_info["class_map"], threshold)
        except Exception as e:
            context.logger.error(str(e))
            response_body["message"] = str(e)
            response_body["code"] = 500
        return context.Response(body=response_body,
                                headers=headers,
                                content_type='application/json',
                                status_code=200)
'''
Author: ChengYong chengy@luokung.com
Date: 2022-10-11 20:13:42
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 21:25:24
FilePath: /serverless/pytorch/ultralytics/yolov5-coco/main.py
Description:

Copyright (c) 2022 by ChengYong chengy@luokung.com, All Rights Reserved.
'''
import os
import json
import base64
import io

import nuclio_sdk
from PIL import Image

from infer import init_model, do_inference
from tools import logger, parse_yaml, list_to_mapping, formatter

headers={
    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
    'Access-Control-Allow-Credentials':'false',
    'Access-Control-Allow-Origin': "*",
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
}

def init_context(context):
    context.logger.info("Init context...  0%")
    weights = os.path.join("/opt/nuclio/weights", os.environ['WEIGHTS_FILE'])
    imgsz = [640, 640]
    classes_info = json.loads(parse_yaml("/opt/nuclio/function.yaml")["metadata"]["annotations"]["spec"])
    model_info = {
        "model": init_model(weights, imgsz), # model, stride, device
        "classes": classes_info,
        "class_map": list_to_mapping(classes_info),
        "imgsz": [640, 640]
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
            buf = io.BytesIO(base64.b64decode(data["image"]))
            threshold = float(data.get("threshold", 0.5))
            image = Image.open(buf)
            model, stride, device = context.model_info["model"]
            dets = do_inference(model, image, context.model_info["imgsz"], stride, device)
            response_body["result"] = formatter(dets, context.model_info["class_map"], threshold)
        except Exception as e:
            context.logger.error(str(e))
            response_body["message"] = str(e)
            response_body["code"] = 500
        return context.Response(body=response_body,
                                headers=headers,
                                content_type='application/json',
                                status_code=200)
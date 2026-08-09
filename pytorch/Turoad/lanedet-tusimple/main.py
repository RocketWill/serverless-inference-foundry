'''
Date: 2022-01-11 22:36:10
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:51:44
'''
import sys
import os
import json
import base64
import io
import time
import traceback
import base64
import nuclio_sdk

import cv2
from PIL import Image
import numpy as np

from infer import init_model, do_inference
from utils import logger, parse_yaml, list_to_mapping, formatter

headers={
    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
    'Access-Control-Allow-Credentials':'false',
    'Access-Control-Allow-Origin': "*",
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
}

def init_context(context):
    context.logger.info("Init context...  0%")
    classes_info = json.loads(parse_yaml("/opt/nuclio/function.yaml")["metadata"]["annotations"]["spec"])
    class Args:
        config = "/app/configs/laneatt/resnet34_tusimple.py"
        load_from = os.path.join("/opt/nuclio/weights", os.environ['WEIGHTS_FILE'])
        img = None
        show = False
        savedir = None
    args = Args()
    model_info = {
        "model": init_model(args),
        "classes": classes_info,
        "class_map": list_to_mapping(classes_info)
    }
    context.model_info = model_info
    context.logger.info("Init context...100%")
    logger.info("Init context success!")

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
            threshold = float(data.get("threshold", 0.5)) # threshold is not needed in this function
            image = Image.open(buf)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # convert PIL to opencv
            lanes = do_inference(context.model_info["model"], image)
            response_body["result"] = formatter(lanes, context.model_info["class_map"])
        except Exception as e:
            logger.error(traceback.format_exc())
            response_body["message"] = "OOPS！自动化标注遇到问题喽，请确认 Label 是否已成功创建。"
            response_body["code"] = 500
        return context.Response(body=response_body,
                                headers=headers,
                                content_type='application/json',
                                status_code=200)

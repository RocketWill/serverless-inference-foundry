'''
Date: 2022-01-11 22:36:10
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:37:21
'''
import os
import json
import base64
import io
import traceback
import base64
import nuclio_sdk

from PIL import Image
from detectron2.data.detection_utils import convert_PIL_to_numpy

from two_stage_infer import TwoStageDetection
from tools import logger, parse_yaml, list_to_mapping, formatter

headers={
    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
    'Access-Control-Allow-Credentials':'false',
    'Access-Control-Allow-Origin': "*",
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
}

def init_context(context):
    context.logger.info("Init context...  0%")
    classes_info = json.loads(parse_yaml("/opt/nuclio/function.yaml")["metadata"]["annotations"]["spec"])
    config = {
        "first": {
            "weights": os.path.join("/opt/nuclio/weights", os.environ['DET_WEIGHTS_FILE']),
            "imgsz": [640, 640]
        },
        "second": {
            "cfg_file": "/opt/nuclio/configs/train.yaml",
            "weights": os.path.join("/opt/nuclio/weights", os.environ['KPT_WEIGHTS_FILE']),
        }
    }
    model_info = {
        "model": TwoStageDetection(config),
        "classes": classes_info,
        "class_map": list_to_mapping(classes_info)
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
            threshold = float(data.get("threshold", 0.6))
            image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
            cuboids = context.model_info["model"].infer(image, threshold)
            response_body["result"] = formatter(cuboids, context.model_info["class_map"], threshold)
        except Exception as e:
            context.logger.error(traceback.format_exc())
            response_body["message"] = "OOPS！自动化标注遇到问题喽，请确认 Label 是否已成功创建。"
            response_body["code"] = 500
        return context.Response(body=response_body,
                                headers=headers,
                                content_type='application/json',
                                status_code=200)

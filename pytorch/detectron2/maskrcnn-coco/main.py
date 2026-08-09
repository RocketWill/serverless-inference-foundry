'''
Date: 2021-12-15 16:31:30
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 23:09:10
'''
import os
import json
import base64
import io
import base64
import nuclio_sdk

from PIL import Image
from detectron2.data.detection_utils import convert_PIL_to_numpy

from infer import init_model, do_inference
from utils import logger, parse_yaml, list_to_mapping, formatter, RequestParser

headers={
    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
    'Access-Control-Allow-Credentials':'false',
    'Access-Control-Allow-Origin': "*",
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
}

def init_context(context):
    context.logger.info("Init context...  0%")
    classes_info = json.loads(parse_yaml("/opt/nuclio/function.yaml")["metadata"]["annotations"]["spec"])
    model_info = {
        "model": init_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 0.3),
        "classes": classes_info,
        "class_map": list_to_mapping(classes_info)
    }
    context.model_info = model_info
    context.request_parser = RequestParser()
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
            image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
            scores, pred_classes, polygons = do_inference(context.model_info["model"], image)
            response_body["result"] = formatter(scores, pred_classes, polygons, context.model_info["class_map"], threshold)
        except Exception as e:
            context.logger.error(str(e))
            response_body["message"] = str(e)
            response_body["code"] = 500
        return context.Response(body=response_body,
                                headers=headers,
                                content_type='application/json',
                                status_code=200)

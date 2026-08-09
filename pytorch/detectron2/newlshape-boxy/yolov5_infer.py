"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import traceback

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
def init_model(
    weights,
    imgsz,
    device="",
    half=False,
):
    try:
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half: model.half()  # to FP16
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        return model, stride, device
    except Exception as e:
        logging.error(e)
        return None, None, None

@torch.no_grad()
def do_inference(
    model,
    source,
    imgsz,
    stride,
    device,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    classes=None,
    half=False,
    augment=False,
    agnostic_nms=False
):
    try:
        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        img0 = np.array(source)
        img0 = img0[:, :, ::-1].copy()
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        if device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        for img, im0 in [[img, img0]]:
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(img, augment=augment, visualize=False)[0]
            # pred[..., 0] *= imgsz[1]/640  # x
            # pred[..., 1] *= imgsz[0]/640  # y
            # pred[..., 2] *= imgsz[1]/640  # w
            # pred[..., 3] *= imgsz[0]/640  # h
            pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            t = tuple(x / 1 * 1E3 for x in dt)  # speeds per image
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0.shape).round()
            logging.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            results = np.float32(pred[0].cpu().tolist()).tolist()
            return results
    except Exception as e:
        logging.error(traceback.format_exc())
        raise ValueError(traceback.format_exc())

# def main():
#     weights = "/outputs/yolov5m_bdd.pt"
#     source = "/outputs/cars.jpeg"
#     imgsz = [640, 640]
#     model, stride, device = init_model(weights, imgsz)
#     if model and stride and device:
#         return do_inference(model, source, imgsz, stride, device)
#     else:
#         logging.error("Init model fail.")
#         return []

# if __name__ == "__main__":
#     results = main()
#     print(results)

#     image = cv2.imread("/outputs/cars.jpeg")
#     for res in results:
#         x1, y1, x2, y2, conf, cls = np.int0(res)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
#     cv2.imwrite("/outputs/cars_.jpeg", image)
'''
Date: 2022-01-20 05:01:07
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 21:38:30
'''
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from tools import logger

def init_model(cfg, imgsz, weights):
    device = select_device("")
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    return model, device

def padding_image(img0, img_size, auto_size):
    img = letterbox(img0, new_shape=img_size, auto_size=auto_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img

def do_inference(model, device, imgsz, image, conf_thres=0.4, iou_thres=0.5):
    # Set Dataloader
    dataset = [[None, padding_image(image, imgsz, 64), image]]
    results = []
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() # fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred2 = model(img, augment=False)
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                results.extend(det.cpu().tolist())
    return results

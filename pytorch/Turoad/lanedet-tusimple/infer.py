'''
Date: 2022-01-13 23:33:16
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:57:09
'''
import os
import os.path as osp

import numpy as np
import torch
import cv2
from pathlib import Path

import torch
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network

from utils import logger

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': None, 'ori_img': ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        # print([np.int0(lane).tolist() for lane in lanes])
        # imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)
    def _flatten(self, pts):
        return [element for tupl in pts for element in tupl]

    def to_list(self, data, scaler, ori_shape):
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        lanes = [np.int0(np.array(lane)/scaler) for lane in lanes]
        results = []
        h, w = ori_shape
        for lane in lanes:
            single_lane_result = []
            for pt in lane:
                if (pt[0] > 0 and pt[1] > 0) and (pt[0] < w and pt[1] < h):
                    single_lane_result.append([int(pt[0]), int(pt[1])])
            if len(single_lane_result) > 1:
                results.append(self._flatten(single_lane_result))
        # return [self._flatten(np.int0(np.array(lane)/scaler).tolist()) for lane in lanes]
        return results

    def run(self, data, scaler, ori_shape):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        return self.to_list(data, scaler, ori_shape)

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths

def preprocess_image(image):
    ori_h, ori_w, _ = image.shape
    scaler = min(1280/ori_w, 720/ori_h)
    blank_image = np.zeros((720, 1280, 3), np.uint8) # 'cause we use TuSimple dataset, img_sz is (720, 1280)
    target_size = (int(ori_w*scaler), int(ori_h*scaler))
    resized_image = cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
    blank_image[0:resized_image.shape[0], 0:resized_image.shape[1]] = resized_image
    return blank_image, scaler

def init_model(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    return Detect(cfg)

def do_inference(detector, image):
    resized_image, scaler = preprocess_image(image)
    return detector.run(resized_image, scaler, image.shape[:2])

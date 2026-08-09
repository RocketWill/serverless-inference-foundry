'''
Author: ChengYong chengy@luokung.com
Date: 2022-10-11 22:27:39
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 22:29:34
FilePath: /serverless/pytorch/detectron2/newlshape-boxy/two_stage_infer.py
Description:

Copyright (c) 2022 by ChengYong chengy@luokung.com, All Rights Reserved.
'''
import cv2

from yolov5_infer import init_model as yolo_init, do_inference as yolo_infer
from d2_infer import Predictor as D2Predictor, do_inference as d2_inference

class TwoStageDetection:
    def __init__(self, config):
        self.config = config
        self.yolo_detector = yolo_init(
            config["first"]["weights"],
            config["first"]["imgsz"])
        self.kpts_detector = D2Predictor(
            config["second"]["cfg_file"],
            config["second"]["weights"])

    def _add_padding(self, imgsz, bbox):
        x0, y0, x1, y1 = bbox
        full_h, full_w, _ = imgsz
        crop_h, crop_w = abs(x0 - x1), abs(y0 - y1)
        padd_h, padd_w = int(0.25 * crop_h / 2), int(0.25 * crop_w / 2)
        x0 = max(x0 - padd_w, 0)
        y0 = max(y0 - padd_h, 0)
        x1 = min(x1 + padd_w, full_w)
        y1 = min(y1 + padd_h, full_h)
        return [int(ele) for ele in [x0, y0, x1, y1]]

    def _yolo_infer(self, image, thresh):
        model, stride, device = self.yolo_detector
        dets = yolo_infer(model, image, self.config["first"]["imgsz"], stride, device)
        results = []
        for det in dets:
            x0, y0, x1, y1, conf, cls_id = det
            if conf < thresh: continue
            if cls_id not in [2, 3, 4]: continue # we only need [car, bus, truck]
            results.append([int(x0), int(y0), int(x1), int(y1), float(conf), int(cls_id)])
        return results

    def _d2_infer(self, cropped_image, yolo_score):
        cuboids = d2_inference(self.kpts_detector, cropped_image, yolo_score)
        return cuboids

    def infer(self, image, thresh=0.6):
        results = []
        yolo_dets = self._yolo_infer(image, thresh)
        for det in yolo_dets:
            x0, y0, x1, y1, conf, cls_id = det
            x0, y0, x1, y1 = self._add_padding(image.shape, [x0, y0, x1, y1])
            cropped_image = image[y0:y1, x0:x1]

            if cropped_image.shape[0] < 1 or cropped_image.shape[1] < 1:
                continue

            cuboids = self._d2_infer(cropped_image, conf)
            if len(cuboids) < 1: continue
            cuboid = cuboids[0]
            crop_pts = cuboid.to_list()
            if not crop_pts: continue
            scaled_pts = []
            for idx, p in enumerate(crop_pts):
                if idx % 2 == 0:
                    scaled_pts.append(p + x0)
                else:
                    scaled_pts.append(p + y0)
            cuboid.set_scaled_pts(scaled_pts)
            results.append(cuboid)
        return results

    def visualize(self, image, cuboids):
        for cuboid in cuboids:
            sp = cuboid.get_scaled_pts()
            aabb = [(sp[0], sp[1]), (sp[6], sp[7])]
            if cuboid.direction.value == 'right':
                rear = [(sp[8], sp[9]), (sp[10], sp[11])]
            elif cuboid.direction.value == 'left':
                rear = [(sp[12], sp[13]), (sp[14], sp[15])]
            else:
                rear = None
            cv2.rectangle(image, aabb[0], aabb[1], (0,255,0), 1)
            if rear:
                cv2.circle(image, rear[0], 2, (0,0,255), -1)
                cv2.circle(image, rear[1], 2, (0,0,255), -1)
        import time
        cv2.imwrite("{}.png".format(time.time()), image)
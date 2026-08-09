'''
Date: 2021-12-15 17:07:09
Company: Luokung Technology Corp.
LastEditors: ChengYong chengy@luokung.com
LastEditTime: 2022-10-11 19:22:09
'''
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import torch
from torch import nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from contextlib import ExitStack, contextmanager
from skimage import measure

from utils import logger


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0, slice_num=5):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        contour = contour[::slice_num]
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def init_model(cfg_file, weights):
    cfg = LazyConfig.load(cfg_file)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    logger.info("Init model successfully.")
    return [cfg, model]

def do_inference(predictor, image):
    with ExitStack() as stack:
        if isinstance(predictor, nn.Module):
            stack.enter_context(inference_context(predictor))
        stack.enter_context(torch.no_grad())

        original_image = image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        outputs = predictor([inputs])[0]

        fields = outputs['instances'].get_fields()
        scores = fields['scores'].cpu().tolist()
        pred_boxes = fields['pred_boxes'].to('cpu')  # need to convert to mormal list
        pred_classes = fields['pred_classes'].cpu().tolist()
        assert len(scores) == len(pred_boxes) == len(pred_classes)
        logger.info("Do inference.")

        return scores, pred_classes, pred_boxes

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

if __name__ == "__main__":
    pass
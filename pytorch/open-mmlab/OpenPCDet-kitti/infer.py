'''
Author: Will Cheng Yong chengyong@pku.edu.cn
Date: 2022-10-02 16:18:53
LastEditors: Will Cheng Yong chengyong@pku.edu.cn
LastEditTime: 2022-10-08 16:28:25
FilePath: /lkdi-auto-annotation/detector/bbox/open-pcdet-kitti/infer.py
Description:

Copyright (c) 2022 by Will Cheng Yong chengyong@pku.edu.cn, All Rights Reserved.
'''
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from utils import logger

class InferDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, points=[]):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=None, logger=logger
        )
        self.points = points

    def __len__(self):
        return 1

    def __getitem__(self, index):
        input_dict = {
            'points': self.points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


@torch.no_grad()
def init_model(
    weights,
    cfg_file,
):
    cfg_from_yaml_file(cfg_file, cfg)
    demo_dataset = InferDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=weights, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    return cfg, model

@torch.no_grad()
def do_inference(
    infer_dataset,
    model
):
    try:
        for _, data_dict in enumerate(infer_dataset):
            data_dict = infer_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # boxes(xyzlwhr), scores, labels
            return [
                pred_dicts[0]['pred_boxes'].cpu().numpy().tolist(),
                pred_dicts[0]['pred_scores'].cpu().numpy().tolist(),
                pred_dicts[0]['pred_labels'].cpu().numpy().tolist(),
            ]

    except Exception as e:
        logger.error(e)
        raise ValueError(e)

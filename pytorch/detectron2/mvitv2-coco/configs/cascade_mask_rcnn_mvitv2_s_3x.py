'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-05 22:51:44
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-07-06 01:20:09
FilePath: /workspace/experiments/detectron2/projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from .cascade_mask_rcnn_mvitv2_t_3x import model, dataloader, optimizer, lr_multiplier, train


model.backbone.bottom_up.depth = 16
model.backbone.bottom_up.last_block_indexes = (0, 2, 13, 15)

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_S_in1k.pyth"

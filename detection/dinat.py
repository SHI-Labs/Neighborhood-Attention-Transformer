"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from nat import NAT


@BACKBONES.register_module()
class DiNAT(NAT):
    """
    DiNAT is NAT with dilations.
    It's that simple!
    """

    pass

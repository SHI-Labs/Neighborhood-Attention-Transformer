"""
Mask2Former training script + DiNAT as a backbone.

Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
sys.path.insert(0, './M2F')
_ds = os.getenv("DETECTRON2_DATASETS", "M2F/datasets")
os.environ["DETECTRON2_DATASETS"] = _ds

import detectron2.utils.comm as comm
from detectron2.engine import (
    default_argument_parser,
    launch,
    default_setup,
)
from detectron2.config import CfgNode, get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
import M2F.train_net as m2f_train_net
from dinat import *


def add_dinat_config(cfg):
    cfg.MODEL.DINAT = CfgNode()
    cfg.MODEL.DINAT.EMBED_DIM = 192
    cfg.MODEL.DINAT.DEPTHS = [3, 4, 18, 5]
    cfg.MODEL.DINAT.NUM_HEADS = [6, 12, 24, 48]
    cfg.MODEL.DINAT.KERNEL_SIZE = 7
    cfg.MODEL.DINAT.DILATIONS = None
    cfg.MODEL.DINAT.MLP_RATIO = 2.0
    cfg.MODEL.DINAT.QKV_BIAS = True
    cfg.MODEL.DINAT.QK_SCALE = None
    cfg.MODEL.DINAT.DROP_RATE = 0.0
    cfg.MODEL.DINAT.ATTN_DROP_RATE = 0.0
    cfg.MODEL.DINAT.DROP_PATH_RATE = 0.3
    cfg.MODEL.DINAT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


def m2f_dinat_setup(args):
    """
    Modified version of the original;
    Just adds DiNAT args.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    m2f_train_net.add_maskformer2_config(cfg)
    add_dinat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


m2f_train_net.setup = m2f_dinat_setup


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        m2f_train_net.main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


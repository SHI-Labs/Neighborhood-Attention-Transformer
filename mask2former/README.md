# Image Segmentation with Mask2Former

Our Mask2Former experiments in [DiNAT](https://arxiv.org/abs/2209.15001) were conducted with the [original Mask2Former repository](https://github.com/facebookresearch/Mask2Former).
We simply added DiNAT as a backbone, and trained it with the same settings as the original Swin-L experiments.
Refer to [Mask2Former's README](M2F/README.md) for more information.

Please make sure you clone this repository with the `--recursive` flag to include [Mask2Former's source](https://github.com/facebookresearch/Mask2Former).

## Setup
You can simply set up the environment by running:
```shell
pip install -r requirements-base.txt
pip install -r requirements.txt
```
The first command installs torch and torchvision, and the second installs the rest of the dependencies for those specific torch builds.
Recommended Python version is 3.8.

Make sure to refer to [Mask2Former's dataset instructions](M2F/datasets/README.md) to set up dataset paths before training/evaluation.

## Training
Just set `$CONFIG` to whichever you prefer, and run:
```
python train_m2f.py --num-gpus 8 --config-file $CONFIG
```

## Evaluation
Running evaluation is also through the same script, but with additional flags:
```
python train_m2f.py --config-file $CONFIG \
  --eval-only MODEL.WEIGHTS $PATH_TO_CHECKPOINT
```

## Model Zoo

### Instance Segmentation
| Backbone | Dataset | # of Params | FLOPs | AP | Config | Checkpoint |
|---|---|---|---|---|---|---|
| DiNAT-Large | MS-COCO | 220M | 522G | 50.7 | [YAML file](configs/coco/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_coco_instance.pth) |
| DiNAT-Large | ADE20K | 220M | 535G | 35.2 | [YAML file](configs/ade20k/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_instance.pth) |
| DiNAT-Large | Cityscapes | 220M | 522G | 44.5 | [YAML file](configs/cityscapes/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_instance.pth) |

### Semantic Segmentation
| Backbone | Dataset | # of Params | FLOPs | mIoU (multiscale) | Config | Checkpoint |
|---|---|---|---|---|---|---|
| DiNAT-Large | ADE20K | 220M | 518G | 58.2 | [YAML file](configs/ade20k/semantic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_semantic.pth) |
| DiNAT-Large | Cityscapes | 220M | 509G | 84.5 | [YAML file](configs/cityscapes/semantic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_semantic.pth) |


### Instance Segmentation
| Backbone | Dataset | # of Params | FLOPs | PQ | AP | mIoU | Config | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| DiNAT-Large | MS-COCO | 220M | 522G | 58.2 | 49.2 | 68.1 | [YAML file](configs/coco/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_coco_panoptic.pth) |
| DiNAT-Large | ADE20K | 220M | 535G | 48.5 | 34.4 | 56.2 | [YAML file](configs/ade20k/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_panoptic.pth) |
| DiNAT-Large | Cityscapes | 220M | 522G | 66.9 | 43.8 | 83.2 | [YAML file](configs/cityscapes/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_panoptic.pth) |
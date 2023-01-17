# DiNAT - Object Detection and Instance Segmentation

Make sure to set up your environment according to the [object detection README](README.md).

## Training and evaluation on COCO
Training and evaluation is identical to [NAT](NAT.md).

## Checkpoints
### DiNAT
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Pre-training | Checkpoint | Config file |
|---|---|---|---|---|---|---|---|---|
| DiNAT-Mini | Mask R-CNN | 40M | 225G | 47.2 | 42.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_mini.pth) | [config.py](configs/dinat/mask_rcnn_dinat_mini_3x_coco.py) |
| DiNAT-Tiny | Mask R-CNN | 48M | 258G | 48.6 | 43.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_tiny.pth) | [config.py](configs/dinat/mask_rcnn_dinat_tiny_3x_coco.py) |
| DiNAT-Small | Mask R-CNN | 70M | 330G | 49.3 | 44.0 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_small.pth) | [config.py](configs/dinat/mask_rcnn_dinat_small_3x_coco.py) |
| DiNAT-Mini | Cascade Mask R-CNN | 77M | 704G | 51.2 | 44.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_mini.pth) | [config.py](configs/dinat/cascade_mask_rcnn_dinat_mini_3x_coco.py) |
| DiNAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 52.2 | 45.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_tiny.pth) | [config.py](configs/dinat/cascade_mask_rcnn_dinat_tiny_3x_coco.py) |
| DiNAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.9 | 45.8 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_small.pth) | [config.py](configs/dinat/cascade_mask_rcnn_dinat_small_3x_coco.py) |
| DiNAT-Base | Cascade Mask R-CNN | 147M | 931G | 53.4 | 46.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_base.pth) | [config.py](configs/dinat/cascade_mask_rcnn_dinat_base_3x_coco.py) |
| DiNAT-Large | Cascade Mask R-CNN | 258M | 1276G | 55.3 | 47.8 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_large.pth) |

### DiNAT<sub>s</sub>
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Pre-training | Checkpoint | Config file |
|---|---|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | Mask R-CNN | 48M | 263G | 46.6 | 42.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_s_tiny.pth) | [config.py](configs/dinat_s/mask_rcnn_dinat_s_tiny_3x_coco.py) |
| DiNAT<sub>s</sub>-Small | Mask R-CNN | 69M | 350G | 48.6 | 43.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_s_small.pth) | [config.py](configs/dinat_s/mask_rcnn_dinat_s_small_3x_coco.py) |
| DiNAT<sub>s</sub>-Tiny | Cascade Mask R-CNN | 86M | 742G | 51.0 | 44.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_tiny.pth) | [config.py](configs/dinat_s/cascade_mask_rcnn_dinat_s_tiny_3x_coco.py) |
| DiNAT<sub>s</sub>-Small | Cascade Mask R-CNN | 107M | 829G | 52.3 | 45.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_small.pth) | [config.py](configs/dinat_s/cascade_mask_rcnn_dinat_s_small_3x_coco.py) |
| DiNAT<sub>s</sub>-Base | Cascade Mask R-CNN | 145M | 966G | 52.6 | 45.3 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_base.pth) | [config.py](configs/dinat_s/cascade_mask_rcnn_dinat_s_base_3x_coco.py) |
| DiNAT<sub>s</sub>-Large | Cascade Mask R-CNN | 253M | 1357G | 54.8 | 47.2 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_large.pth) | [config.py](configs/dinat_s/cascade_mask_rcnn_dinat_s_large_3x_coco.py) |

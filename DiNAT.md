# Dilated Neighborhood Attention Transformer

<a href="https://arxiv.org/abs/2209.15001"><img src="https://img.shields.io/badge/arXiv-2209.15001-orange" /></a>

Preprint Link: [Dilated Neighborhood Attention Transformer
](https://arxiv.org/abs/2209.15001)

By [Ali Hassani<sup>[1]</sup>](https://alihassanijr.com/),
and
[Humphrey Shi<sup>[1,2]</sup>](https://www.humphreyshi.com/)

In association with SHI Lab @ University of Oregon & UIUC<sup>[1]</sup> and Picsart AI Research (PAIR)<sup>[2]</sup>.


![DiNAT-Intro](assets/dinat/radar_dark.png#gh-dark-mode-only)
![DiNAT-Intro](assets/dinat/radar_light.png#gh-light-mode-only)


# Abstract
Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities, 
domains, and tasks. In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have also 
gained significant attention, thanks to their performance and easy integration into existing frameworks. These models 
typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA) or 
Swin Transformer's Shifted Window Self Attention. While effective at reducing self attention's quadratic complexity, 
local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling, 
and global receptive field. In this paper, we introduce Dilated Neighborhood Attention (DiNA), a natural, flexible and 
efficient extension to NA that can capture more global context and expand receptive fields exponentially at no 
additional cost. NA's local attention and DiNA's sparse global attention complement each other, and therefore we 
introduce Dilated Neighborhood Attention Transformer (DiNAT), a new hierarchical vision transformer built upon both. 
DiNAT variants enjoy significant improvements over strong baselines such as NAT, Swin, and ConvNeXt. Our large model is 
faster and ahead of its Swin counterpart by 1.5% box AP in COCO object detection, 1.3% mask AP in COCO instance 
segmentation, and 1.1% mIoU in ADE20K semantic segmentation. Paired with new frameworks, our large variant is the new 
state of the art panoptic segmentation model on COCO (58.2 PQ) and ADE20K (48.5 PQ), and instance segmentation model on 
Cityscapes (44.5 AP) and ADE20K (35.4 AP) (no extra data). It also matches the state of the art specialized semantic 
segmentation models on ADE20K (58.2 mIoU), and ranks second on Cityscapes (84.5 mIoU) (no extra data). 


# Results and checkpoints

## Image Classification
### DiNAT
DiNAT is identical to NAT in architecture, with every other layer replaced with Dilated NA.
These variants provide similar or better classification accuracy (except for Tiny), but yield significantly better downstream performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 |
|---|---|---|---|---|---|---|
| DiNAT-Mini | 224x224 | 7x7 | 20M | 2.7G | - | [81.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) |
| DiNAT-Tiny | 224x224 | 7x7 | 28M | 4.3G | - | [82.7%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) |
| DiNAT-Small | 224x224 | 7x7 | 51M | 7.8G | - | [83.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) |
| DiNAT-Base | 224x224 | 7x7 | 90M | 13.7G | - | [84.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) |
| DiNAT-Large | 224x224 | 7x7 | 200M | 30.6G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [86.6%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_224.pth) |
| DiNAT-Large | 384x384 | 7x7 | 200M | 89.7G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [87.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384.pth) |
| DiNAT-Large | 384x384 | 11x11 | 200M | 92.4G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224_11x11interp.pth) | [87.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth) |

### DiNAT<sub>s</sub>
DiNAT<sub>s</sub> variants are identical to Swin in terms of architecture, with WSA replaced with NA and SWSA replaced with DiNA.
These variants can provide better throughput on CUDA, at the expense of slightly higher memory footprint, and lower performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 |
|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | 224x224 | 7x7 | 28M | 4.5G | - | [81.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) |
| DiNAT<sub>s</sub>-Small | 224x224 | 7x7 | 50M | 8.7G | - | [83.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) |
| DiNAT<sub>s</sub>-Base | 224x224 | 7x7 | 88M | 15.4G | - | [83.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth) |
| DiNAT<sub>s</sub>-Large | 224x224 | 7x7 | 197M | 34.5G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [86.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_224.pth) |
| DiNAT<sub>s</sub>-Large | 384x384 | 7x7 | 197M | 101.5G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [87.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_384.pth) |

### Isotropic variants

| Model | # of Params | FLOPs | Top-1 |
|---|---|---|---|
| NAT-iso-Small | 22M | 4.3G | [80.0%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_small_in1k_224.pth) |
| DiNAT-iso-Small | 22M | 4.3G | [80.8%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_small_in1k_224.pth) |
| ViT-rpb-Small | 22M | 4.6G | [81.2%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_small_in1k_224.pth) |
| NAT-iso-Base | 86M | 16.9G | [81.6%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_base_in1k_224.pth) |
| DiNAT-iso-Base | 86M | 16.9G | [82.1%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_base_in1k_224.pth) |
| ViT-rpb-Base | 86M | 17.5G | [82.5%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_base_in1k_224.pth) |

Details on training and validation are provided in [classification](classification/DiNAT.md).

## Object Detection and Instance Segmentation
### DiNAT
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT-Mini | Mask R-CNN | 40M | 225G | 47.2 | 42.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_mini.pth) |
| DiNAT-Tiny | Mask R-CNN | 48M | 258G | 48.6 | 43.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_tiny.pth) |
| DiNAT-Small | Mask R-CNN | 70M | 330G | 49.3 | 44.0 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_small.pth) |
| DiNAT-Mini | Cascade Mask R-CNN | 77M | 704G | 51.2 | 44.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_mini.pth) |
| DiNAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 52.2 | 45.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_tiny.pth) |
| DiNAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.9 | 45.8 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_small.pth) |
| DiNAT-Base | Cascade Mask R-CNN | 147M | 931G | 53.4 | 46.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_base.pth) |
| DiNAT-Large | Cascade Mask R-CNN | 258M | 1276G | 55.3 | 47.8 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_large.pth) |

### DiNAT<sub>s</sub>
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | Mask R-CNN | 48M | 263G | 46.6 | 42.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_s_tiny.pth) |
| DiNAT<sub>s</sub>-Small | Mask R-CNN | 69M | 350G | 48.6 | 43.5 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/maskrcnn_dinat_s_small.pth) |
| DiNAT<sub>s</sub>-Tiny | Cascade Mask R-CNN | 86M | 742G | 51.0 | 44.1 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_tiny.pth) |
| DiNAT<sub>s</sub>-Small | Cascade Mask R-CNN | 107M | 829G | 52.3 | 45.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_small.pth) |
| DiNAT<sub>s</sub>-Base | Cascade Mask R-CNN | 145M | 966G | 52.6 | 45.3 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_base.pth) |
| DiNAT<sub>s</sub>-Large | Cascade Mask R-CNN | 253M | 1357G | 54.8 | 47.2 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/coco/cascadedmaskrcnn_dinat_s_large.pth) |


Details on training and validation are provided in [detection](detection/DiNAT.md).

## Semantic Segmentation
### DiNAT
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT-Mini | UPerNet | 50M | 900G | 45.8 | 47.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_mini.pth) |
| DiNAT-Tiny | UPerNet| 58M | 934G | 47.8 | 48.8 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_tiny.pth) |
| DiNAT-Small | UPerNet | 82M | 1010G | 48.9 | 49.9 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_small.pth) |
| DiNAT-Base | UPerNet | 123M | 1137G | 49.6 | 50.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_base.pth) |
| DiNAT-Large | UPerNet | 238M | 2335G | 54.0 | 54.9 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_large.pth) |

### DiNAT<sub>s</sub>
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | UPerNet| 60M | 941G | 46.0 | 47.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_tiny.pth) |
| DiNAT<sub>s</sub>-Small | UPerNet | 81M | 1030G | 48.6 | 49.9 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_small.pth) |
| DiNAT<sub>s</sub>-Base | UPerNet | 121M | 1173G | 49.4 | 50.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_base.pth) |
| DiNAT<sub>s</sub>-Large | UPerNet | 234M | 2466G | 53.4 | 54.6 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_large.pth) |


Details on training and validation are provided in [segmentation](segmentation/DiNAT.md).

## Image Segmentation with Mask2Former

### Instance Segmentation
| Backbone | Dataset | # of Params | FLOPs | AP | Config | Checkpoint |
|---|---|---|---|---|---|---|
| DiNAT-Large | MS-COCO | 220M | 522G | 50.8 | [YAML file](configs/coco/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_coco_instance.pth) |
| DiNAT-Large | ADE20K | 220M | 535G | 35.4 | [YAML file](configs/ade20k/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_instance.pth) |
| DiNAT-Large | Cityscapes | 220M | 522G | 45.1 | [YAML file](configs/cityscapes/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_instance.pth) |

### Semantic Segmentation
| Backbone | Dataset | # of Params | FLOPs | mIoU (multiscale) | Config | Checkpoint |
|---|---|---|---|---|---|---|
| DiNAT-Large | ADE20K | 220M | 518G | 58.1 | [YAML file](configs/ade20k/semantic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_semantic.pth) |
| DiNAT-Large | Cityscapes | 220M | 509G | 84.5 | [YAML file](configs/cityscapes/semantic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_semantic.pth) |


### Panoptic Segmentation
| Backbone | Dataset | # of Params | FLOPs | PQ | AP | mIoU | Config | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| DiNAT-Large | MS-COCO | 220M | 522G | 58.5 | 49.2 | 68.3 | [YAML file](configs/coco/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_coco_panoptic.pth) |
| DiNAT-Large | ADE20K | 220M | 535G | 49.4 | 35.0 | 56.3 | [YAML file](configs/ade20k/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_160k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_ade20k_panoptic.pth) |
| DiNAT-Large | Cityscapes | 220M | 522G | 67.2 | 44.5 | 83.4 | [YAML file](configs/cityscapes/panoptic-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_90k.yaml) | [Download](https://shi-labs.com/projects/dinat/checkpoints/m2f/mask2former_dinat_large_cityscapes_panoptic.pth) |

Details on training and validation are provided in [mask2former](mask2former/README.md).


# Citation
```bibtex
@article{hassani2022dilated,
	title        = {Dilated Neighborhood Attention Transformer},
	author       = {Ali Hassani and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2209.15001},
	eprint       = {2209.15001},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

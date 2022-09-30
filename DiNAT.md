# Dilated Neighborhood Attention Transformer
Preprint Link: [Dilated Neighborhood Attention Transformer
](https://arxiv.org/abs/2209.15001)

By [Ali Hassani](https://alihassanijr.com/),
and
[Humphrey Shi](https://www.humphreyshi.com/)

In association with SHI Lab @ University of Oregon & UIUC and Picsart AI Research (PAIR).


![DiNAT-Intro](assets/dinat/radar_dark.png#gh-dark-mode-only)
![DiNAT-Intro](assets/dinat/radar_light.png#gh-light-mode-only)


# Abstract
Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities, domains, and tasks.
In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have also gained significant attention, thanks to their performance and easy integration into existing frameworks.
These models typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA) or Swin Transformer's Shifted Window Self Attention.
While effective at reducing self attention's quadratic complexity, local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling, and global receptive field.
In this paper, we introduce Dilated Neighborhood Attention (DiNA), a natural, flexible and efficient extension to NA that can capture more global context and expand receptive fields exponentially at no additional cost. 
NA's local attention and DiNA's sparse global attention complement each other, and therefore we introduce Dilated Neighborhood Attention Transformer (DiNAT), a new hierarchical vision transformer built upon both.
DiNAT variants enjoy significant improvements over attention-based baselines such as NAT and Swin, as well as modern convolutional baseline ConvNeXt.
Our Large model is ahead of its Swin counterpart by 1.5% box AP in COCO object detection, 1.3% mask AP in COCO instance segmentation, and 1.1% mIoU in ADE20K semantic segmentation, and faster in throughput. 
We believe combinations of NA and DiNA have the potential to empower various tasks beyond those presented in this paper.


# Results and checkpoints

Checkpoints will be available soon. Stay tuned!

## Image Classification
### DiNAT
DiNAT is identical to NAT in architecture, with every other layer replaced with Dilated NA.
These variants provide similar or better classification accuracy (except for Tiny), but yield significantly better downstream performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 |
|---|---|---|---|---|---|---|
| DiNAT-Mini | 224x224 | 7x7 | 20M | 2.7G | - | [81.8%](#) |
| DiNAT-Tiny | 224x224 | 7x7 | 28M | 4.3G | - | [82.7%](#) |
| DiNAT-Small | 224x224 | 7x7 | 51M | 7.8G | - | [83.8%](#) |
| DiNAT-Base | 224x224 | 7x7 | 90M | 13.7G | - | [84.4%](#) |
| DiNAT-Large | 224x224 | 7x7 | 200M | 30.6G | [ImageNet-22K](#) | [86.5%](#) |
| DiNAT-Large | 384x384 | 7x7 | 200M | 89.7G | [ImageNet-22K](#) | [87.2%](#) |
| DiNAT-Large | 384x384 | 11x11 | 200M | 92.4G | [ImageNet-22K](#) | [87.3%](#) |

### DiNAT<sub>s</sub>
DiNAT<sub>s</sub> variants are identical to Swin in terms of architecture, with WSA replaced with NA and SWSA replaced with DiNA.
These variants can provide better throughput on CUDA, at the expense of slightly higher memory footprint, and lower performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 |
|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | 224x224 | 7x7 | 28M | 4.5G | - | [81.8%](#) |
| DiNAT<sub>s</sub>-Small | 224x224 | 7x7 | 50M | 8.7G | - | [83.5%](#) |
| DiNAT<sub>s</sub>-Base | 224x224 | 7x7 | 88M | 15.4G | - | [83.8%](#) |
| DiNAT<sub>s</sub>-Large | 224x224 | 7x7 | 197M | 34.5G | [ImageNet-22K](#) | [86.5%](#) |
| DiNAT<sub>s</sub>-Large | 384x384 | 7x7 | 197M | 101.5G | [ImageNet-22K](#) | [87.4%](#) |


Details on training and validation are provided in [classification](classification/DiNAT.md).

## Object Detection and Instance Segmentation
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT-Mini | Mask R-CNN | 40M | 225G | 47.2 | 42.5 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Tiny | Mask R-CNN | 48M | 258G | 48.6 | 43.5 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Small | Mask R-CNN | 70M | 330G | 49.3 | 44.0 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Mini | Cascade Mask R-CNN | 77M | 704G | 51.2 | 44.4 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 52.2 | 45.1 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.9 | 45.8 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Base | Cascade Mask R-CNN | 147M | 931G | 53.4 | 46.2 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Large | Cascade Mask R-CNN | 258M | 1276G | 55.2 | 47.7 | [ImageNet-22K](#) | [Download](#) |

Details on training and validation are provided in [detection](detection/DiNAT.md).

## Semantic Segmentation
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Pre-training | Checkpoint |
|---|---|---|---|---|---|---|---|
| DiNAT-Mini | UPerNet | 50M | 900G | 45.8 | 47.2 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Tiny | UPerNet| 58M | 934G | 47.8 | 48.8 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Small | UPerNet | 82M | 1010G | 48.9 | 49.9 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Base | UPerNet | 123M | 1137G | 49.6 | 50.4 | [ImageNet-1K](#) | [Download](#) |
| DiNAT-Large | UPerNet | 238M | 2335G | 53.5 | 54.6 | [ImageNet-22K](#) | [Download](#) |

Details on training and validation are provided in [segmentation](segmentation/DiNAT.md).


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
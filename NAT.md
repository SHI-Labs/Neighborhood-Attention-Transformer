# Neighborhood Attention Transformer

<a href="https://arxiv.org/abs/2204.07143"><img src="https://img.shields.io/badge/arXiv-2204.07143-orange" /></a>
<a href="https://www.youtube.com/watch?v=Ya4BfioxIHA"><img src="https://img.shields.io/badge/YouTube-Presentation-red" /></a>

[Neighborhood Attention Transformer](https://openaccess.thecvf.com/content/CVPR2023/html/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.html)
(CVPR 2023.)

By [Ali Hassani<sup>[1]</sup>](https://alihassanijr.com/),
[Steven Walton<sup>[1]</sup>](https://stevenwalton.github.io/),
[Jiachen Li<sup>[1]</sup>](https://chrisjuniorli.github.io/), 
[Shen Li<sup>[3]</sup>](https://mrshenli.github.io/), 
and
[Humphrey Shi<sup>[1,2]</sup>](https://www.humphreyshi.com/)

In association with SHI Lab @ University of Oregon & UIUC<sup>[1]</sup> and
Picsart AI Research (PAIR)<sup>[2]</sup>, and Meta/Facebook AI<sup>[3]</sup>


![NAT-Intro](assets/nat/intro_dark.png#gh-dark-mode-only)
![NAT-Intro](assets/nat/intro_light.png#gh-light-mode-only)


# Abstract
![NAT-Arch](assets/nat/model_dark.png#gh-dark-mode-only)
![NAT-Arch](assets/nat/model_light.png#gh-light-mode-only)
We present Neighborhood Attention (NA), the first efficient and scalable sliding window attention mechanism for vision. 
NA is a pixel-wise operation, localizing self attention (SA) to the nearest neighboring pixels, and therefore enjoys a linear 
time and space complexity compared to the quadratic complexity of SA. The sliding window pattern allows NA's receptive field to 
grow without needing extra pixel shifts, and preserves translational equivariance, unlike Swin Transformer's Window Self 
Attention (WSA). We develop [NATTEN (Neighborhood Attention Extension)](https://github.com/SHI-Labs/NATTEN/), a Python package 
with efficient C++ and CUDA kernels, which allows NA to run up to 40% faster than Swin's WSA while using up to 25% less memory. 
We further present Neighborhood Attention Transformer (NAT), a new hierarchical transformer design based on NA that boosts 
image classification and downstream vision performance. Experimental results on NAT are competitive; NAT-Tiny reaches 83.2% 
top-1 accuracy on ImageNet, 51.4% mAP on MS-COCO and 48.4% mIoU on ADE20K, which is 1.9% ImageNet accuracy, 1.0% COCO mAP, 
and 2.6% ADE20K mIoU improvement over a Swin model with similar size. 
To support more research based on sliding window attention, we open source our project and release our checkpoints.

## Implementation
Neighborhood Attention is implemented within our [Neighborhood Attention Extension (NATTEN)](https://github.com/SHI-Labs/NATTEN/). 
It's relatively fast, memory-efficient, supports half precision, and comes with both CPU and CUDA kernels.
There's still a lot of room for improvement, 
so feel free to open PRs and contribute to [NATTEN](https://github.com/SHI-Labs/NATTEN/)!

# Results and checkpoints

## Classification
| Model | # of Params | FLOPs | Top-1 |
|---|---|---|---|
| NAT-Mini | 20M | 2.7G | [81.8%](https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth) |
| NAT-Tiny | 28M | 4.3G | [83.2%](https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth) |
| NAT-Small | 51M | 7.8G | [83.7%](https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth) |
| NAT-Base | 90M | 13.7G | [84.3%](https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth) |


Details on training and validation are provided in [classification](classification/NAT.md).

## Object Detection
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Checkpoint |
|---|---|---|---|---|---|---|
| NAT-Mini | Mask R-CNN | 40M | 225G | 46.5 | 41.7 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_maskrcnn.pth) |
| NAT-Tiny | Mask R-CNN | 48M | 258G | 47.7 | 42.6 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_maskrcnn.pth) |
| NAT-Small | Mask R-CNN | 70M | 330G | 48.4 | 43.2 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_maskrcnn.pth) |
| NAT-Mini | Cascade Mask R-CNN | 77M | 704G | 50.3 | 43.6 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_cascademaskrcnn.pth) |
| NAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 51.4 | 44.5 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_cascademaskrcnn.pth) |
| NAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.0 | 44.9 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_cascademaskrcnn.pth) |
| NAT-Base | Cascade Mask R-CNN | 147M | 931G | 52.3 | 45.1 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_base_cascademaskrcnn.pth) |

Details on training and validation are provided in [detection](detection/NAT.md).

## Semantic Segmentation
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Checkpoint |
|---|---|---|---|---|---|---|
| NAT-Mini | UPerNet | 50M | 900G | 45.1 | 46.4 | [Download](https://shi-labs.com/projects/nat/checkpoints/SEG/nat_mini_upernet.pth) |
| NAT-Tiny | UPerNet| 58M | 934G | 47.1 | 48.4 | [Download](https://shi-labs.com/projects/nat/checkpoints/SEG/nat_tiny_upernet.pth) |
| NAT-Small | UPerNet | 82M | 1010G | 48.0 | 49.5 | [Download](https://shi-labs.com/projects/nat/checkpoints/SEG/nat_small_upernet.pth) |
| NAT-Base | UPerNet | 123M | 1137G | 48.5 | 49.7 | [Download](https://shi-labs.com/projects/nat/checkpoints/SEG/nat_base_upernet.pth) |

Details on training and validation are provided in [segmentation](segmentation/NAT.md).

# Citation
```bibtex
@inproceedings{hassani2023neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month        = {June},
	year         = {2023},
	pages        = {6185-6194}
}
```

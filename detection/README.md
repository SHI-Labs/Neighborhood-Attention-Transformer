# Object Detection and Instance Segmentation

## Requirements
Python 3.8 is strongly recommended.
PyTorch version `1.11` is strongly recommended.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements-base.txt # Installs torch
pip3 install -r requirements.txt # Installs NATTEN, MMCV, MMDET, and fvcore
```
This will install the recommended torch and torchvision, 
our PyTorch extension ([NATTEN](https://github.com/SHI-Labs/NATTEN)), 
MMCV, 
and all other dependencies.

Similar to and because of [classification](../classification/README.md), object detection also depends on `timm`, 
and `fvcore`. Object detection experiments were conducted with [mmdetection](https://github.com/open-mmlab/mmdetection).
Additionally, they depend on our extension, [NATTEN](https://github.com/SHI-Labs/NATTEN), which you can install 
[by referring to our website](https://www.shi-labs.com/natten/). 
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113

mmcv-full==1.4.8 # Wheels: https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

natten==0.14.2+torch111cu113 # Wheels: http://www.shi-labs.com/natten/wheels/cu113/torch1.11/index.html

mmdet==2.19.0
timm==0.5.0
fvcore==0.1.5.post20220305
```

## Models

* [Neighborhood Attention Transformer (NAT)](NAT.md)

* [Dilated Neighborhood Attention Transformer (DiNAT)](DiNAT.md)

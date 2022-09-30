# Semantic Segmentation

## Requirements
Python 3.8 is strongly recommended.
PyTorch version `1.11` is strongly recommended.
Please read the instructions on how to set up and compile [NATTEN](../NATTEN.md) carefully before starting.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements.txt
```
Similar to and because of [classification](../classification/README.md), semantic segmentation also depends on `timm`, `ninja`, 
and `fvcore`. Semantic Segmentation experiments were conducted with [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/).
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
timm==0.5.0
ninja==1.10.2.3
fvcore==0.1.5.post20220305

mmcv-full==1.4.8
mmsegmentation==0.20.2
```

## Models

* [Neighborhood Attention Transformer (NAT)](NAT.md)

* [Dilated Neighborhood Attention Transformer (DiNAT)](DiNAT.md)
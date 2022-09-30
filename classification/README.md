# Image Classification

## Requirements
Python 3.8 is strongly recommended.
PyTorch version `1.11` is strongly recommended.
Please read the instructions on how to set up and compile [NATTEN](../NATTEN.md) carefully before starting.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements.txt
```
Our models are based on PyTorch, and was trained on ImageNet-1k classification using the `timm` package. 
Additionally, they depend on [NATTEN](../NATTEN.md), which requires `ninja` to compile. 
The version of the `timm` training script available here also requires `fvcore` to count FLOPs.
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
timm==0.5.0
ninja==1.10.2.3
fvcore==0.1.5.post20220305
pyyaml==6.0
```

## Models

* [Neighborhood Attention Transformer (NAT)](NAT.md)

* [Dilated Neighborhood Attention Transformer (DiNAT)](DiNAT.md)
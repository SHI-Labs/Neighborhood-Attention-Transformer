# DiNAT - Classification

Make sure to set up your environment according to the [classification README](README.md).

## Training and on ImageNet-1K
Training and evaluation is identical to [NAT](NAT.md).

## Training on ImageNet-22K

Details will be released soon.

## Checkpoints
### DiNAT
DiNAT is identical to NAT in architecture, with every other layer replaced with Dilated NA.
These variants provide similar or better classification accuracy (except for Tiny), but yield significantly better downstream performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 | Config file |
|---|---|---|---|---|---|---|---|
| DiNAT-Mini | 224x224 | 7x7 | 20M | 2.7G | - | [81.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [dinat_mini.yml](configs/dinat_mini.yml) |
| DiNAT-Tiny | 224x224 | 7x7 | 28M | 4.3G | - | [82.7%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [dinat_tiny.yml](configs/dinat_tiny.yml) |
| DiNAT-Small | 224x224 | 7x7 | 51M | 7.8G | - | [83.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [dinat_small.yml](configs/dinat_small.yml) |
| DiNAT-Base | 224x224 | 7x7 | 90M | 13.7G | - | [84.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) | [dinat_base.yml](configs/dinat_base.yml) |
| DiNAT-Large | 224x224 | 7x7 | 200M | 30.6G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [86.6%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_224.pth) |
| DiNAT-Large | 384x384 | 7x7 | 200M | 89.7G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [87.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384.pth) |
| DiNAT-Large | 384x384 | 11x11 | 200M | 92.4G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224_11x11interp.pth) | [87.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth) |

### DiNAT<sub>s</sub>
DiNAT<sub>s</sub> variants are identical to Swin in terms of architecture, with WSA replaced with NA and SWSA replaced with DiNA.
These variants can provide better throughput on CUDA, at the expense of slightly higher memory footprint, and lower performance.

| Model | Resolution | Kernel size | # of Params | FLOPs | Pre-training | Top-1 | Config file |
|---|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | 224x224 | 7x7 | 28M | 4.5G | - | [81.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_1k_224.pth) | [dinat_s_tiny.yml](configs/dinat_s_tiny.yml) |
| DiNAT<sub>s</sub>-Small | 224x224 | 7x7 | 50M | 8.7G | - | [83.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_1k_224.pth) | [dinat_s_small.yml](configs/dinat_s_small.yml) |
| DiNAT<sub>s</sub>-Base | 224x224 | 7x7 | 88M | 15.4G | - | [83.8%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_1k_224.pth) | [dinat_s_base.yml](configs/dinat_s_base.yml) |
| DiNAT<sub>s</sub>-Large | 224x224 | 7x7 | 197M | 34.5G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [86.5%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_1k_224.pth) | [dinat_s_large.yml](configs/dinat_s_large.yml) |
| DiNAT<sub>s</sub>-Large | 384x384 | 7x7 | 197M | 101.5G | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [87.4%](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_1k_384.pth) | [dinat_s_large_384.yml](configs/dinat_s_large_384.yml) |

### Isotropic variants

| Model | # of Params | FLOPs | Top-1 | Config file |
|---|---|---|---|---|
| NAT-iso-Small | 22M | 4.3G | [80.0%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_small_in1k_224.pth) | [nat_isotropic_small.yml](configs/nat_isotropic_small.yml) |
| DiNAT-iso-Small | 22M | 4.3G | [80.8%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_small_in1k_224.pth) | [dinat_isotropic_small.yml](configs/dinat_isotropic_small.yml) |
| ViT-rpb-Small | 22M | 4.6G | [81.2%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_small_in1k_224.pth) | [vit_rpb_small.yml](configs/vit_rpb_small.yml) |
| NAT-iso-Base | 86M | 16.9G | [81.6%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_base_in1k_224.pth) | [nat_isotropic_base.yml](configs/nat_isotropic_base.yml) |
| DiNAT-iso-Base | 86M | 16.9G | [82.1%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_base_in1k_224.pth) | [dinat_isotropic_base.yml](configs/dinat_isotropic_base.yml) |
| ViT-rpb-Base | 86M | 17.5G | [82.5%](https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_base_in1k_224.pth) | [vit_rpb_base.yml](configs/vit_rpb_base.yml) |

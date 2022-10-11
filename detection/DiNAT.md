# DiNAT - Semantic Segmentation

Make sure to set up your environment according to the [semantic segmentation README](README.md).

## Training and evaluation on ADE20K
Training and evaluation is identical to [NAT](NAT.md).


## Checkpoints
### DiNAT
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Pre-training | Checkpoint | Config file |
|---|---|---|---|---|---|---|---|---|
| DiNAT-Mini | UPerNet | 50M | 900G | 45.8 | 47.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_mini.pth) | [config.py](configs/dinat/upernet_dinat_mini_512x512_160k_ade20k.py) |
| DiNAT-Tiny | UPerNet| 58M | 934G | 47.8 | 48.8 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_tiny.pth) | [config.py](configs/dinat/upernet_dinat_tiny_512x512_160k_ade20k.py) |
| DiNAT-Small | UPerNet | 82M | 1010G | 48.9 | 49.9 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_small.pth) | [config.py](configs/dinat/upernet_dinat_small_512x512_160k_ade20k.py) |
| DiNAT-Base | UPerNet | 123M | 1137G | 49.6 | 50.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_base.pth) | [config.py](configs/dinat/upernet_dinat_base_512x512_160k_ade20k.py) |
| DiNAT-Large | UPerNet | 238M | 2335G | 53.5 | 54.6 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_large.pth) | [config.py](configs/dinat/upernet_dinat_large_640x640_160k_ade20k.py) |

### DiNAT<sub>s</sub>
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Pre-training | Checkpoint | Config file |
|---|---|---|---|---|---|---|---|---|
| DiNAT<sub>s</sub>-Tiny | UPerNet| 60M | 941G | 46.0 | 47.4 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_tiny.pth) | [config.py](configs/dinat_s/upernet_dinat_s_tiny_512x512_160k_ade20k.py) |
| DiNAT<sub>s</sub>-Small | UPerNet | 81M | 1030G | 48.6 | 49.9 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_small.pth) | [config.py](configs/dinat_s/upernet_dinat_s_small_512x512_160k_ade20k.py) |
| DiNAT<sub>s</sub>-Base | UPerNet | 121M | 1173G | 49.4 | 50.2 | [ImageNet-1K](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_base.pth) | [config.py](configs/dinat_s/upernet_dinat_s_base_512x512_160k_ade20k.py) |
| DiNAT<sub>s</sub>-Large | UPerNet | 234M | 2466G | 53.4 | 54.6 | [ImageNet-22K](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth) | [Download](https://shi-labs.com/projects/dinat/checkpoints/ade20k/upernet_dinat_s_large.pth) | [config.py](configs/dinat_s/upernet_dinat_s_large_640x640_160k_ade20k.py) |


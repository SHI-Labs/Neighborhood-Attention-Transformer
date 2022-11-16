# Utils

## Salient map generator
You can use this script to generate salient maps for NAT models, as seen in 
[Neighborhood Attention Transformer (NAT)](https://arxiv.org/abs/2204.07143).

Simply specify an input image, output path, and model name to get started:

```
python utils/gen_salient_maps.py  --image eagle.png --image-out eagle_salient.png --model nat_base --use-cuda
```

We highly recommend using CUDA, as the default number of rounds is `100`.
If you're not using CUDA, you can try fewer rounds:

```
python utils/gen_salient_maps.py  --image eagle.png --image-out eagle_salient.png --model nat_base --rounds 10
```

# Salient maps

| Original | ViT-Base | Swin-Base | NAT-Base |
|---|---|---|---|
| ![img0](../assets/nat/salient/img0.png) | ![img0-vit-dark](../assets/nat/salient/img0_vit_dark.png#gh-dark-mode-only)![img0-vit-light](../assets/nat/salient/img0_vit_light.png#gh-light-mode-only)  | ![img0-swin-dark](../assets/nat/salient/img0_swin_dark.png#gh-dark-mode-only)![img0-swin-light](../assets/nat/salient/img0_swin_light.png#gh-light-mode-only) | ![img0-nat-dark](../assets/nat/salient/img0_nat_dark.png#gh-dark-mode-only)![img0-nat-light](../assets/nat/salient/img0_nat_light.png#gh-light-mode-only) |
| ![img1](../assets/nat/salient/img1.png) | ![img1-vit-dark](../assets/nat/salient/img1_vit_dark.png#gh-dark-mode-only)![img1-vit-light](../assets/nat/salient/img1_vit_light.png#gh-light-mode-only)  | ![img1-swin-dark](../assets/nat/salient/img1_swin_dark.png#gh-dark-mode-only)![img1-swin-light](../assets/nat/salient/img1_swin_light.png#gh-light-mode-only) | ![img1-nat-dark](../assets/nat/salient/img1_nat_dark.png#gh-dark-mode-only)![img1-nat-light](../assets/nat/salient/img1_nat_light.png#gh-light-mode-only) |
| ![img2](../assets/nat/salient/img2.png) | ![img2-vit-dark](../assets/nat/salient/img2_vit_dark.png#gh-dark-mode-only)![img2-vit-light](../assets/nat/salient/img2_vit_light.png#gh-light-mode-only)  | ![img2-swin-dark](../assets/nat/salient/img2_swin_dark.png#gh-dark-mode-only)![img2-swin-light](../assets/nat/salient/img2_swin_light.png#gh-light-mode-only) | ![img2-nat-dark](../assets/nat/salient/img2_nat_dark.png#gh-dark-mode-only)![img2-nat-light](../assets/nat/salient/img2_nat_light.png#gh-light-mode-only) |
| ![img3](../assets/nat/salient/img3.png) | ![img3-vit-dark](../assets/nat/salient/img3_vit_dark.png#gh-dark-mode-only)![img3-vit-light](../assets/nat/salient/img3_vit_light.png#gh-light-mode-only)  | ![img3-swin-dark](../assets/nat/salient/img3_swin_dark.png#gh-dark-mode-only)![img3-swin-light](../assets/nat/salient/img3_swin_light.png#gh-light-mode-only) | ![img3-nat-dark](../assets/nat/salient/img3_nat_dark.png#gh-dark-mode-only)![img3-nat-light](../assets/nat/salient/img3_nat_light.png#gh-light-mode-only) |


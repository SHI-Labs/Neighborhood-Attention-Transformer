# Neighborhood Attention Transformers

<a href="https://arxiv.org/abs/2209.15001"><img src="https://img.shields.io/badge/arXiv-Dilated%20Neighborhood%20Attention%20Trasnformer-%23C209C1" /></a>
<a href="https://arxiv.org/abs/2204.07143"><img src="https://img.shields.io/badge/arXiv-Neighborhood%20Attention%20Trasnformer-%2300B0F0" /></a>
[<img src="https://img.shields.io/badge/CUDA%20Extension-NATTEN-%23fc6562" />](https://github.com/SHI-Labs/NATTEN)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/instance-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/instance-segmentation-on-ade20k-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/panoptic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-ade20k-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/instance-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/instance-segmentation-on-cityscapes-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/panoptic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-cityscapes-val?p=dilated-neighborhood-attention-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dilated-neighborhood-attention-transformer/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=dilated-neighborhood-attention-transformer)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=stylenat-giving-each-head-a-new-perspective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-ffhq-1024-x-1024)](https://paperswithcode.com/sota/image-generation-on-ffhq-1024-x-1024?p=stylenat-giving-each-head-a-new-perspective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=stylenat-giving-each-head-a-new-perspective)

![NAT-Intro](assets/dinat/intro_dark.png#gh-dark-mode-only)
![NAT-Intro](assets/dinat/intro_light.png#gh-light-mode-only)

**Powerful hierarchical vision transformers based on sliding window attention.**

Neighborhood Attention (NA, local attention) was introduced in our original paper, 
[NAT](NAT.md), and runs efficiently with our extension to PyTorch, [NATTEN](https://github.com/SHI-Labs/NATTEN).

We recently introduced a new model, [DiNAT](DiNAT.md), 
which extends NA by dilating neighborhoods (DiNA, sparse global attention, a.k.a. dilated local attention).

Combinations of NA/DiNA are capable of preserving locality, maintaining
translational equivariance,
expanding the receptive field exponentially, 
and capturing longer-range inter-dependencies, 
leading to significant performance boosts in downstream vision tasks, such as
[StyleNAT](https://github.com/SHI-Labs/StyleNAT) for image generation.


# News

### November 18, 2022
* NAT and DiNAT are now available through HuggingFace's [transformers](https://github.com/huggingface/transformers).
  * NAT and DiNAT classification models are also available on the HuggingFace's Model Hub: [NAT](https://huggingface.co/models?filter=nat) | [DiNAT](https://huggingface.co/models?filter=dinat)

### November 11, 2022
* New preprint: [StyleNAT: Giving Each Head a New Perspective](https://github.com/SHI-Labs/StyleNAT).
  * Style-based GAN powered with Neighborhood Attention sets new SOTA on FFHQ-256 with a 2.05 FID.
  ![stylenat](assets/stylenat/stylenat.png)

### October 8, 2022
* [NATTEN](https://github.com/SHI-Labs/NATTEN) is now [available as a pip package](https://www.shi-labs.com/natten/)!
    * You can now install NATTEN with pre-compiled wheels, and start using it in seconds. 
    * NATTEN will be maintained and developed as a [separate project](https://github.com/SHI-Labs/NATTEN) to support broader usage of sliding window attention, even beyond computer vision.

### September 29, 2022
* New preprint: [Dilated Neighborhood Attention Transformer](DiNAT.md).


# Dilated Neighborhood Attention :fire:
![DiNAT-Abs](assets/dinat/radar_dark.png#gh-dark-mode-only)
![DiNAT-Abs](assets/dinat/radar_light.png#gh-light-mode-only)

A new hierarchical vision transformer based on Neighborhood Attention (local attention) and Dilated Neighborhood Attention (sparse global attention) that enjoys significant performance boost in downstream tasks.

Check out the [DiNAT README](DiNAT.md).


# Neighborhood Attention Transformer
![NAT-Abs](assets/nat/computeplot_dark.png#gh-dark-mode-only)
![NAT-Abs](assets/nat/computeplot_light.png#gh-light-mode-only)

Our original paper, [Neighborhood Attention Transformer (NAT)](NAT.md), the first efficient sliding-window local attention.

# How Neighborhood Attention works
Neighborhood Attention localizes the query token's (red) receptive field to its nearest neighboring tokens in the key-value pair (green). 
This is equivalent to dot-product self attention when the neighborhood size is identical to the image dimensions. 
Note that the edges are special (edge) cases.

![720p_fast_dm](assets/nat/720p_fast_dm.gif#gh-dark-mode-only)
![720p_fast_lm](assets/nat/720p_fast_lm.gif#gh-light-mode-only)



# Citation
```bibtex
@article{hassani2022neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2204.07143},
	eprint       = {2204.07143},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
@article{hassani2022dilated,
	title        = {Dilated Neighborhood Attention Transformer},
	author       = {Ali Hassani and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2209.15001},
	eprint       = {2209.15001},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
@article{walton2022stylenat,
	title        = {StyleNAT: Giving Each Head a New Perspective},
	author       = {Steven Walton and Ali Hassani and Xingqian Xu and Zhangyang Wang and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2211.05770},
	eprint       = {2211.05770},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

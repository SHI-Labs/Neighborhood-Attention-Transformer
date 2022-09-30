# Neighborhood Attention Transformers

<a href="https://arxiv.org/abs/2209.15001"><img src="https://img.shields.io/badge/arXiv-Dilated%20Neighborhood%20Attention%20Trasnformer-%23C209C1" /></a>
<a href="https://arxiv.org/abs/2204.07143"><img src="https://img.shields.io/badge/arXiv-Neighborhood%20Attention%20Trasnformer-%2300B0F0" /></a>
[<img src="https://img.shields.io/badge/Extension-Neighborhood%20Attention%20CUDA%20Extension%20for%20PyTorch-%23fc6562" />](NATTEN.md)

![NAT-Intro](assets/dinat/intro_dark.png#gh-dark-mode-only)
![NAT-Intro](assets/dinat/intro_light.png#gh-light-mode-only)

**Powerful hierarchical vision transformers based on sliding window attention.**

Neighborhood Attention (NA, local attention) was introduced in our original paper, 
[NAT](NAT.md), and runs efficiently with our CUDA extension to PyTorch, [NATTEN](NATTEN.md).

We recently introduced a new model, [DiNAT](DiNAT.md), which extends NA by dilating neighborhoods (DiNA, sparse global attention).

Combinations of NA/DiNA are capable of preserving locality, 
expanding the receptive field exponentially, 
and capturing longer-range inter-dependencies, 
leading to significant performance boosts in downstream vision tasks.


# Dilated Neighborhood Attention :fire:
![DiNAT-Intro](assets/dinat/radar_dark.png#gh-dark-mode-only)
![DiNAT-Intro](assets/dinat/radar_light.png#gh-light-mode-only)

<a href="https://arxiv.org/abs/2209.15001"><img src="https://img.shields.io/badge/arXiv-2209.15001-orange" /></a>

A new hierarchical vision transformer based on Neighborhood Attention (local attention) and Dilated Neighborhood Attention (sparse global attention).

Check out the [DiNAT README](DiNAT.md).


# Neighborhood Attention Transformer

<a href="https://arxiv.org/abs/2204.07143"><img src="https://img.shields.io/badge/arXiv-2204.07143-orange" /></a>

Our original paper, [Neighborhood Attention Transformer (NAT)](NAT.md), the first efficient sliding-window local attention.

# How Neighborhood Attention works
Neighborhood Attention localizes the query token's (red) receptive field to its nearest neighboring tokens in the key-value pair (green). 
This is equivalent to dot-product self attention when the neighborhood size is identical to the image dimensions. 
Note that the edges are special (edge) cases.

![720p_fast_dm](assets/nat/720p_fast_dm.gif#gh-dark-mode-only)
![720p_fast_lm](assets/nat/720p_fast_lm.gif#gh-light-mode-only)


# News

### September 29, 2022
* New preprint: [Dilated Neighborhood Attention Transformer](DiNAT.md).
* [NA CUDA extension v0.13](NATTEN.md) released with dilation support!
  * See [changelog](CHANGELOG.md).

### July 9, 2022
* [NA CUDA extension v0.12](NATTEN.md) released.
  * NA runs much more efficiently now, up to 40% faster and uses up to 25% less memory compared to Swin Transformer’s Shifted Window Self Attention.
  * Improved FP16 throughput.
  * Improved training speed and stability.
  * See [changelog](CHANGELOG.md).
  
### May 12, 2022
* [1-D Neighborhood Attention](NATTEN.md) support added!
* Moved the kernel to `natten/` now, since there's a single version for all three tasks, and we're adding more features to the extension.

### April 30, 2022
* [NA CUDA extension v0.11](NATTEN.md) released.
  * It's faster in both training and inference, 
  * with a single version for all three tasks (no downstream-specific version)
* [PyTorch implementation](NATTEN.md) released
  * Works both with and without CUDA, but not very efficient. Try to use the CUDA extension when possible.
  * See [changelog](CHANGELOG.md).


# Catalog
- [x] Neighborhood Attention 1D (CUDA)
- [x] Neighborhood Attention 2D (CUDA)
- [ ] Neighborhood Attention 1D (PyTorch)
- [x] Neighborhood Attention 2D (PyTorch)
- [x] Dilation support
- [ ] BFloat16 support (coming soon)
- [ ] Zeros/Valid padding support (coming soon)
- [ ] HuggingFace Demo




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
```

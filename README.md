# Neighborhood Attention Transformers

<a href="https://arxiv.org/abs/2209.15001"><img src="https://img.shields.io/badge/arXiv-Dilated%20Neighborhood%20Attention%20Trasnformer-%23C209C1" /></a>
<a href="https://arxiv.org/abs/2204.07143"><img src="https://img.shields.io/badge/arXiv-Neighborhood%20Attention%20Trasnformer-%2300B0F0" /></a>
[<img src="https://img.shields.io/badge/CUDA%20Extension-NATTEN-%23fc6562" />](https://github.com/SHI-Labs/NATTEN)

![NAT-Intro](assets/dinat/intro_dark.png#gh-dark-mode-only)
![NAT-Intro](assets/dinat/intro_light.png#gh-light-mode-only)

**Powerful hierarchical vision transformers based on sliding window attention.**

Neighborhood Attention (NA, local attention) was introduced in our original paper, 
[NAT](NAT.md), and runs efficiently with our CUDA extension to PyTorch, [NATTEN](https://github.com/SHI-Labs/NATTEN).

We recently introduced a new model, [DiNAT](DiNAT.md), 
which extends NA by dilating neighborhoods (DiNA, sparse global attention, a.k.a. dilated local attention).

Combinations of NA/DiNA are capable of preserving locality, 
expanding the receptive field exponentially, 
and capturing longer-range inter-dependencies, 
leading to significant performance boosts in downstream vision tasks.


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


# News

### October 8, 2022
* [NATTEN](https://github.com/SHI-Labs/NATTEN) is now [available as a pip package](https://www.shi-labs.com/natten/)!
    * You can now install NATTEN with pre-compiled wheels, and start using it in seconds. 

### September 29, 2022
* New preprint: [Dilated Neighborhood Attention Transformer](DiNAT.md).
* [NA CUDA extension v0.13](https://github.com/SHI-Labs/NATTEN) released with dilation support!

### July 9, 2022
* [NA CUDA extension v0.12](https://github.com/SHI-Labs/NATTEN) released.
  * NA runs much more efficiently now, up to 40% faster and uses up to 25% less memory compared to Swin Transformer’s Shifted Window Self Attention.
  * Improved FP16 throughput.
  * Improved training speed and stability.
  
### May 12, 2022
* [1-D Neighborhood Attention](https://github.com/SHI-Labs/NATTEN) support added!
* Moved the kernel to `natten/` now, since there's a single version for all three tasks, and we're adding more features to the extension.

### April 30, 2022
* [NA CUDA extension v0.11](https://github.com/SHI-Labs/NATTEN) released.
  * It's faster in both training and inference, 
  * with a single version for all three tasks (no downstream-specific version)
* [PyTorch implementation](https://github.com/SHI-Labs/NATTEN) released
  * Works both with and without CUDA, but not very efficient. Try to use the CUDA extension when possible.



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

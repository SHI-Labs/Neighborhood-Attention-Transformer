# Neighborhood Attention Transformers



![NAT-Intro](assets/dinat/intro_dark.png#gh-dark-mode-only)
![NAT-Intro](assets/dinat/intro_light.png#gh-light-mode-only)


# Dilated Neighborhood Attention :fire:
Check out our new model, [Dilated Neighborhood Attention Transformer (DiNAT)](DiNAT.md).


# Neighborhood Attention Transformer
Check our original paper, [Neighborhood Attention Transformer (NAT)](NAT.md).

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

![V012](assets/natten/v012dark.png#gh-dark-mode-only) ![V012](assets/natten/v012light.png#gh-light-mode-only)
![V012](assets/natten/kernelmemory_dark.png#gh-dark-mode-only) ![V012](assets/natten/kernelmemory_light.png#gh-light-mode-only)


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
- [ ] Zeros/Valid padding support
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

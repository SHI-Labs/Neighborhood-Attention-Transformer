# Neighborhood Attention CUDA Kernel

![computeplot_dark](../../assets/kernelplot_dark.png#gh-dark-mode-only)
![computeplot_light](../../assets/kernelplot_light.png#gh-light-mode-only)

## Requirements
NATTEN is a PyTorch CUDA extension, therefore requires PyTorch. 
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
ninja==1.10.2.3
```
PyTorch version `1.11` is strongly recommended as the CUDA extension runs faster due to an updated 
version of the atomic add operator we use for backpropagation.

## Setup
### With Ninja
The recommended way of installing it is through `ninja`. 
By having `ninja` installed, you don't need to install anything manually. 
The extension will compile when it's called first.
To compile and ensure that the extension is functioning correctly, please run:
```
python3 gradcheck.py
```

### Without Ninja
If you want to build without JIT, simply build with setup:
```shell
python setup.py install
```
After it builds, please run the following to ensure that the extension is functioning correctly:
```
python3 gradcheck.py
```


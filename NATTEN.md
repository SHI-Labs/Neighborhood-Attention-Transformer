# Neighborhood Attention

## PyTorch Implementation (new)
This implementation is based on `F.unfold` and `F.pad`, therefore doesn't require the extension, and can run on a CPU.
However, it is very inefficient, and uses up a lot of memory (see the figure below).
### Usage
Simply import `LegacyNeighborhoodAttention` from `natten`:
```python
from natten import LegacyNeighborhoodAttention
```

## CUDA Extension

Training time improvement vs CUDA extension version | Throughput vs Accuracy
:-------------------------:|:-------------------------:
![computeplot_dark](assets/kernelplot_dark.png#gh-dark-mode-only) ![computeplot_light](assets/kernelplot_light.png#gh-light-mode-only) | ![NAT-Intro](assets/throughputplot_dark.png#gh-dark-mode-only) ![NAT-Intro](assets/throughputplot_light.png#gh-light-mode-only)


Compute vs Accuracy |  Memory usage vs Accuracy
:-------------------------:|:-------------------------:
![computeplot_dark](assets/computeplot_dark.png#gh-dark-mode-only) ![computeplot_light](assets/computeplot_light.png#gh-light-mode-only) | ![NAT-Intro](assets/memoryusage_dark.png#gh-dark-mode-only) ![NAT-Intro](assets/memoryusage_light.png#gh-light-mode-only) 


### Requirements
NATTEN is a PyTorch CUDA extension, therefore requires PyTorch. 
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
ninja==1.10.2.3
```
PyTorch version `1.11` is strongly recommended as the CUDA extension runs faster due to an updated 
version of the atomic add operator we use for backpropagation.

### Setup
#### With Ninja
The recommended way of installing it is through `ninja`. 
By having `ninja` installed, you don't need to install anything manually. 
The extension will compile when it's called first.
To compile and ensure that the extension is functioning correctly, please run:
```
python3 natten/gradcheck.py
python3 natten/gradcheck1d.py # 1D NA
```

#### Without Ninja
If you want to build without JIT, simply build with setup:
```shell
cd natten/src
python setup.py install
```
After it builds, please run the following to ensure that the extension is functioning correctly:
```
python3 natten/gradcheck.py
python3 natten/gradcheck1d.py # 1D NA
```
### Usage
Simply import `NeighborhoodAttention` from `natten`:
```python
from natten import NeighborhoodAttention
```
To use the 1D version of NA, simply import `NeighborhoodAttention1d` from `natten`:
```python
from natten import NeighborhoodAttention1d
```

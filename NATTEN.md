# Neighborhood Attention CUDA Extension (NATTEN)
NATTEN is a CUDA extension to PyTorch.

![V012](assets/natten/v012dark.png#gh-dark-mode-only) ![V012](assets/natten/v012light.png#gh-light-mode-only)
![V012](assets/natten/kernelmemory_dark.png#gh-dark-mode-only) ![V012](assets/natten/kernelmemory_light.png#gh-light-mode-only)

### Requirements
The following are the recommended versions of these libraries and are strongly encouraged for speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
ninja==1.10.2.3
```
PyTorch version `1.11` is strongly recommended as the CUDA extension runs faster due to an updated 
version of the atomic add operator we use for backpropagation.
PyTorch version `1.12` support is experimental, therefore use at your own risk.

### Setup
#### With Ninja
The recommended way of installing it is through `ninja`. 
By having `ninja` installed, you don't need to install anything manually. 
The extension will compile when it's called first.
To compile and ensure that the extension is functioning correctly, please run:
```
python3 natten/gradcheck1d.py # 1D NA
python3 natten/gradcheck2d.py # 2D NA
```

#### Without Ninja
If you want to build without JIT, simply build with setup:
```shell
cd natten/src
python setup.py install
```
After it builds, please run the following to ensure that the extension is functioning correctly:
```
python3 natten/gradcheck1d.py # 1D NA
python3 natten/gradcheck2d.py # 2D NA
```
### Usage
Simply import `NeighborhoodAttention2D` from `natten`:
```python
from natten import NeighborhoodAttention2D
```
To use the 1D version of NA, simply import `NeighborhoodAttention1D` from `natten`:
```python
from natten import NeighborhoodAttention1D
```


## Pure PyTorch Implementation
This implementation is based on `F.unfold` and `F.pad`, therefore doesn't require the extension, and can run on a CPU.
However, it is very inefficient, and uses up a lot of memory (see the figure below).
### Usage
Simply import `LegacyNeighborhoodAttention2D` from `natten`:
```python
from natten import LegacyNeighborhoodAttention2D
```
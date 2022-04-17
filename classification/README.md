# NAT - Classification

## Requirements
Python 3.8 is strongly encouraged.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements.txt
```
NAT is based on PyTorch, and was trained on ImageNet-1k classification using the `timm` package. 
Additionally, NAT depends upon two PyTorch CUDA extensions, which require the package `ninja` to compile. 
The version of the `timm` training script available here also requires `fvcore` to count FLOPs.
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
timm==0.5.0
ninja==1.10.2.3
fvcore==0.1.5.post20220305
pyyaml==6.0
```
PyTorch version `1.11` is strongly recommended as the CUDA extension runs faster due to an updated 
version of the atomic add operator we use for backpropagation.

## Setup
Once you have the requirements, please run the following and allow a few minutes for Ninja to compile the extension:
```shell
python3 cuda/gradcheck.py
```
Ninja is not verbose while compiling, so it is normal not to see any outputs unless there is an error. 
After a few minutes you should see the message "Verifying backward pass...".
You should see two tests pass afterwards to verify that the extension is functioning normally. 
Once it is done, you can start training normally.

It is recommended to repeat this step upon every update to the repository.

You can find more details about the extension [here](cuda/README.md).

## Training on ImageNet-1k
<details>
<summary>
<b>NAT-Mini</b>
</summary>

```shell
./dist_train.sh $NUM_GPUS -c configs/nat_mini.yml /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Tiny</b>
</summary>

```shell
./dist_train.sh $NUM_GPUS -c configs/nat_tiny.yml /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Small</b>
</summary>

```shell
./dist_train.sh $NUM_GPUS -c configs/nat_small.yml /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Base</b>
</summary>

```shell
./dist_train.sh $NUM_GPUS -c configs/nat_base.yml /path/to/ImageNet1k
```
</details>

## Validation
<details>
<summary>
<b>NAT-Mini</b>
</summary>

```shell
python3 validate.py --model nat_mini --pretrained /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Tiny</b>
</summary>

```shell
python3 validate.py --model nat_tiny --pretrained /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Small</b>
</summary>

```shell
python3 validate.py --model nat_small --pretrained /path/to/ImageNet1k
```
</details>
<details>
<summary>
<b>NAT-Base</b>
</summary>

```shell
python3 validate.py --model nat_base --pretrained /path/to/ImageNet1k
```
</details>

## Checkpoints
| Model | # of Params | FLOPs | Top-1 | Config file |
|---|---|---|---|---|
| NAT-Mini | 20M | 2.7G | [81.8%](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_mini.pth) | [nat_mini.yml](configs/nat_mini.yml) |
| NAT-Tiny | 28M | 4.3G | [83.2%](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_tiny.pth) | [nat_tiny.yml](configs/nat_tiny.yml) |
| NAT-Small | 51M | 7.8G | [83.7%](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_small.pth) | [nat_small.yml](configs/nat_small.yml) |
| NAT-Base | 90M | 13.7G | [84.3%](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_base.pth) | [nat_base.yml](configs/nat_base.yml) |


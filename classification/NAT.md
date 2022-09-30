# NAT - Classification

Make sure to set up your environment according to the [classification README](README.md).

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


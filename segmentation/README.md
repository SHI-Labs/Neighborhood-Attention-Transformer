# NAT - Semantic Segmentation

## Requirements
Python 3.7 is strongly encouraged.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements.txt
```
Similar to and because of [classification](../classification/README.md), object detection also depends on `timm`, `ninja`, 
and `fvcore`. Semantic Segmentation experiments were conducted with [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/).
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
timm==0.5.0
ninja==1.10.2.3
fvcore==0.1.5.post20220305

mmcv-full==1.4.8
mmsegmentation==0.20.2
```

## Setup
The kernel is identical to the one in classification, it is just configured differently for downstream tasks because 
they are more resolution-heavy, as opposed to the former which is batch-heavy.

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

## Training on ADE20K
<details>
<summary>
<b>NAT-Mini + UPerNet</b>
</summary>

```shell
./dist_train.sh configs/nat/upernet_nat_mini_512x512_160k_ade20k.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Tiny + UPerNet</b>
</summary>

```shell
./dist_train.sh configs/nat/upernet_nat_tiny_512x512_160k_ade20k.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Small + UPerNet</b>
</summary>

```shell
./dist_train.sh configs/nat/upernet_nat_small_512x512_160k_ade20k.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Base + UPerNet</b>
</summary>

```shell
./dist_train.sh configs/nat/upernet_nat_base_512x512_160k_ade20k.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>

## Validation
<details>
<summary>
<b>NAT-Mini + UPerNet</b>
</summary>

Single scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_mini_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_mini_upernet.pth \
    $NUM_GPUS \
    --eval mIoU
```

Multi scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_mini_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_mini_upernet.pth \
    $NUM_GPUS \
    --eval mIoU --aug-test
```
</details>
<details>
<summary>
<b>NAT-Tiny + UPerNet</b>
</summary>

Single scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_tiny_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_tiny_upernet.pth \
    $NUM_GPUS \
    --eval mIoU
```

Multi scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_tiny_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_tiny_upernet.pth \
    $NUM_GPUS \
    --eval mIoU --aug-test
```
</details>
<details>
<summary>
<b>NAT-Small + UPerNet</b>
</summary>

Single scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_small_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_small_upernet.pth \
    $NUM_GPUS \
    --eval mIoU
```

Multi scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_small_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_small_upernet.pth \
    $NUM_GPUS \
    --eval mIoU --aug-test
```
</details>
<details>
<summary>
<b>NAT-Base + UPerNet</b>
</summary>

Single scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_base_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_base_upernet.pth \
    $NUM_GPUS \
    --eval mIoU
```

Multi scale:
```shell
./dist_test.sh \
    configs/nat/upernet_nat_base_512x512_160k_ade20k.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_base_upernet.pth \
    $NUM_GPUS \
    --eval mIoU --aug-test
```
</details>

## Checkpoints
| Backbone | Network | # of Params | FLOPs | mIoU | mIoU (multi-scale) | Checkpoint | Config |
|---|---|---|---|---|---|---|---|
| NAT-Mini | UPerNet | 50M | 900G | 45.1 | 46.4 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_mini_upernet.pth) | [config.py](configs/nat/upernet_nat_mini_512x512_160k_ade20k.py) |
| NAT-Tiny | UPerNet | 58M | 934G | 47.1 | 48.4 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_tiny_upernet.pth) | [config.py](configs/nat/upernet_nat_tiny_512x512_160k_ade20k.py) |
| NAT-Small | UPerNet | 82M | 1010G | 48.0 | 49.5 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_small_upernet.pth) | [config.py](configs/nat/upernet_nat_small_512x512_160k_ade20k.py) |
| NAT-Base | UPerNet | 123M | 1137G | 48.5 | 49.7 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/SEG/nat_base_upernet.pth) | [config.py](configs/nat/upernet_nat_base_512x512_160k_ade20k.py) |

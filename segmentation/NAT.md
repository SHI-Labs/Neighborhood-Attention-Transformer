# NAT - Semantic Segmentation

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

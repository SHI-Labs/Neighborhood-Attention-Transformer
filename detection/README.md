# NAT - Object Detection

## Requirements
Python 3.7 is strongly encouraged.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements.txt
```
Similar to and because of [classification](../classification/README.md), object detection also depends on `timm`, `ninja`, 
and `fvcore`. Object detection experiments were conducted with [mmdetection](https://github.com/open-mmlab/mmdetection).
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113
timm==0.5.0
ninja==1.10.2.3
fvcore==0.1.5.post20220305

mmcv-full==1.4.8
mmdet==2.19.0
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

## Training on COCO

### Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_mini_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_tiny_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_small_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>

### Cascade Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Cascade R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Base + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>

## Validation
### Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_mini_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_mini_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_tiny_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_tiny_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_small_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_small_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>

### Cascade Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Cascade R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_mini_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Tiny + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_tiny_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Small + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_small_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Base + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py \
    http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_base_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>

## Checkpoints
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Checkpoint | Config |
|---|---|---|---|---|---|---|---|
| NAT-Mini | Mask R-CNN | 40M | 225G | 46.5 | 41.7 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_mini_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_mini_3x_coco.py) |
| NAT-Tiny | Mask R-CNN | 48M | 258G | 47.7 | 42.6 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_tiny_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_tiny_3x_coco.py) |
| NAT-Small | Mask R-CNN | 70M | 330G | 48.4 | 43.2 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_small_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_small_3x_coco.py) |
| NAT-Mini | Cascade Mask R-CNN | 77M | 704G | 50.3 | 43.6 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_mini_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py) |
| NAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 51.4 | 44.5 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_tiny_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py) |
| NAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.0 | 44.9 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_small_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py) |
| NAT-Base | Cascade Mask R-CNN | 147M | 931G | 52.3 | 45.1 | [Download](http://ix.cs.uoregon.edu/~alih/nat/checkpoints/DET/nat_base_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py) |



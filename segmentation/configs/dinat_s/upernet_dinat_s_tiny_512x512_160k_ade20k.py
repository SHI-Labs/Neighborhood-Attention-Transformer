_base_ = [
    '../_base_/models/upernet_dinats.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.3,
        patch_norm=True,
        kernel_size=7,
        dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4], [1, 2]],
        pretrained='https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth',
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    ))

# AdamW optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'rpb': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                 }),)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# Mixed precision
fp16 = None
optimizer_config = dict(
    type="Fp16OptimizerHook",
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
)

"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from timm.models.registry import register_model
from nat import NAT

model_urls = {
    # ImageNet-1K
    "dinat_mini_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth",
    "dinat_tiny_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth",
    "dinat_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth",
    "dinat_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth",
    "dinat_large_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_224.pth",
    "dinat_large_1k_384": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384.pth",
    "dinat_large_1k_384_11x11": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth",
    # ImageNet-22K
    "dinat_large_21k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth",
    "dinat_large_21k_11x11": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224_11x11interp.pth",
    # 11x11 contains the same weights as the original, except for RPB which is interpolated using a bicubic interpolation.
    # Swin uses the same interpolation when changing window sizes.
}


class DiNAT(NAT):
    """
    DiNAT is NAT with dilations.
    It's that simple!
    """

    pass


@register_model
def dinat_mini(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_mini_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_tiny(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_small(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_base(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_large(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        embed_dim=192,
        mlp_ratio=2,
        drop_path_rate=0.35,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_large_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_large_384(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        embed_dim=192,
        mlp_ratio=2,
        drop_path_rate=0.35,
        kernel_size=7,
        dilations=[
            [1, 13, 1],
            [1, 6, 1, 6],
            [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_large_1k_384"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_large_384_11x11(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        embed_dim=192,
        mlp_ratio=2,
        drop_path_rate=0.35,
        kernel_size=11,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_large_1k_384_11x11"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_large_21k(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        embed_dim=192,
        mlp_ratio=2,
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_large_21k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_large_21k_11x11(pretrained=False, **kwargs):
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        embed_dim=192,
        mlp_ratio=2,
        drop_path_rate=0.2,
        kernel_size=11,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_large_21k_11x11"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

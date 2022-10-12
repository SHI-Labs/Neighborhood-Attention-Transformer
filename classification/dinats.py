"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

DiNAT_s -- our alternative model.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from nat import Mlp

model_urls = {
    # ImageNet-1K
    "dinat_s_tiny_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth",
    "dinat_s_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth",
    "dinat_s_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth",
    "dinat_s_large_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_224.pth",
    "dinat_s_large_1k_384": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_384.pth",

    # ImageNet-22K
    "dinat_s_large_21k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth",
}


class NATransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7, dilation=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = pad(x, (0, 0, 0, W % 2, 0, H % 2))
            _, H, W, _ = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilations=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            NATransformerLayer(dim=dim,
                               num_heads=num_heads,
                               kernel_size=kernel_size,
                               dilation=1 if dilations is None else dilations[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """
    From Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DiNAT_s(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 kernel_size=7,
                 dilations=None,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               kernel_size=kernel_size,
                               dilations=None if dilations is None else dilations[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def dinat_s_tiny(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], embed_dim=96, mlp_ratio=4,
                    drop_path_rate=0.2,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_s_small(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], embed_dim=96, mlp_ratio=4,
                    drop_path_rate=0.3,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_s_base(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=128, mlp_ratio=4,
                    drop_path_rate=0.5,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_s_large(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=192, mlp_ratio=4,
                    drop_path_rate=0.35,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_s_large_384(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=192, mlp_ratio=4,
                    drop_path_rate=0.35,
                    kernel_size=7,
                    dilations=[
                        [1, 13],
                        [1, 6],
                        [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_large_1k_384']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def dinat_s_large_21k(pretrained=False, **kwargs):
    model = DiNAT_s(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=192, mlp_ratio=4,
                    drop_path_rate=0.2,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                    **kwargs)
    if pretrained:
        url = model_urls['dinat_s_large_21k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

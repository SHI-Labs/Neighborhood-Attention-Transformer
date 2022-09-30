"""
Neighborhood Attention PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from .nattencuda2d import NeighborhoodAttention2D


class NeighborhoodAttention(NeighborhoodAttention2D):
    """
    Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__(dim=dim, kernel_size=kernel_size, num_heads=num_heads,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
                         dilation=dilation)
        warnings.warn('Using NeighborhoodAttention has been deprecated since natten v0.13. ' +
                      'Please consider using NeighborhoodAttention2D instead.', DeprecationWarning, stacklevel=2)

"""
Neighborhood Attention PyTorch Module (Based on existing torch modules)
This version does not require the torch extension and is implemented using unfold + pad.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from .nattentorch2d import LegacyNeighborhoodAttention2D


class LegacyNeighborhoodAttention(LegacyNeighborhoodAttention2D):
    """
    Legacy Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__(dim=dim, kernel_size=kernel_size, num_heads=num_heads,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
                         mode=mode)
        warnings.warn('Using LegacyNeighborhoodAttention has been deprecated since natten v0.13. ' +
                      'Please consider using LegacyNeighborhoodAttention2D instead.', DeprecationWarning, stacklevel=2)

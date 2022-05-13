"""
Neighborhood Attention 1D PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import os

try:
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.realpath(__file__))
    natten1dav_cuda = load(
        'natten1dav_cuda', [f'{this_dir}/src/natten1dav_cuda.cpp', f'{this_dir}/src/natten1dav_cuda_kernel.cu'], verbose=False)
    natten1dqkrpb_cuda = load(
        'natten1dqkrpb_cuda', [f'{this_dir}/src/natten1dqkrpb_cuda.cpp', f'{this_dir}/src/natten1dqkrpb_cuda_kernel.cu'], verbose=False)
except:
    try:
        import natten1dav_cuda
        import natten1dqkrpb_cuda
    except:
        raise RuntimeError("Could not load NATTEN1D CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN1D correctly.")


class NATTEN1DAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = natten1dav_cuda.forward(
                attn, 
                value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = natten1dav_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_attn, d_value = outputs
        return d_attn, d_value


class NATTEN1DQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = natten1dqkrpb_cuda.forward(
                query,
                key,
                rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = natten1dqkrpb_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb


class NeighborhoodAttention1d(nn.Module):
    """
    Neighborhood Attention 1D Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, and 11; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        Lo = L
        pad_r = 0
        if L <= self.kernel_size:
            pad_r = self.kernel_size - L
            x = pad(x, (0, 0, 0, pad_r))
            B, L, C = x.shape
            assert L == self.kernel_size, f"Something went wrong. {L} should equal {self.kernel_size}!"
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = NATTEN1DQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTEN1DAVFunction.apply(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, L, C)
        if pad_r:
            x = x[:, :Lo, :]
        return self.proj_drop(self.proj(x))


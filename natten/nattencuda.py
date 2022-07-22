"""
Neighborhood Attention PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.cpp_extension import load, is_ninja_available
import warnings
import os


if is_ninja_available():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    nattenav_cuda = load(
        'nattenav_cuda', [f'{this_dir}/src/nattenav_cuda.cpp', f'{this_dir}/src/nattenav_cuda_kernel.cu'], verbose=False)
    nattenqkrpb_cuda = load(
        'nattenqkrpb_cuda', [f'{this_dir}/src/nattenqkrpb_cuda.cpp', f'{this_dir}/src/nattenqkrpb_cuda_kernel.cu'], verbose=False)
else:
    warnings.warn("Ninja is not installed, looking up extensions manually.")
    try:
        import nattenav_cuda
        import nattenqkrpb_cuda
    except:
        raise RuntimeError("Could not load NATTEN CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction(Function):
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
        out = nattenav_cuda.forward(
                attn,
                value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_attn, d_value = outputs
        return d_attn, d_value, None


class NATTENQKRPBFunction(Function):
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
        attn = nattenqkrpb_cuda.forward(
                query,
                key,
                rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None


class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

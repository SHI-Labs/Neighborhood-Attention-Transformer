"""
Neighborhood Attention 1D Gradcheck
Checks gradients computed by the extension's backwards kernels against finite differences
to ensure correctness.
WARNING: This script uses A LOT OF MEMORY, so choose sizes wisely.
The defaults provided should run on standard GPUs.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from natten1dcuda import NATTEN1DAVFunction, NATTEN1DQKRPBFunction
import torch
from torch.autograd import gradcheck
import argparse

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=2)
parser.add_argument('-n', '--heads', type=int, default=4)
parser.add_argument('-x', '--length', type=int, default=32)
parser.add_argument('-d', '--dim', type=int, default=32)
parser.add_argument('-k', '--kernel-size', type=int, default=7)
args = parser.parse_args()
kernel_size = args.kernel_size

kwargs = {'dtype': torch.float64,
          'device': 'cuda:0',
          'requires_grad': True}
query = torch.randn((args.batch_size, args.heads, args.length, args.dim), **kwargs)
key = torch.randn((args.batch_size, args.heads, args.length, args.dim), **kwargs)
value = torch.randn((args.batch_size, args.heads, args.length, args.dim), **kwargs)


print("Verifying backward pass...")

rpb = torch.randn((args.heads, 2 * kernel_size - 1), **kwargs)
variables = [query, key, rpb]

if gradcheck(NATTEN1DQKRPBFunction.apply, variables, nondet_tol=1e-8):
    print('1D QK+RPB Gradients Ok')

attn = torch.randn((args.batch_size, args.heads, args.length, kernel_size), **kwargs)
variables = [attn, value]

if gradcheck(NATTEN1DAVFunction.apply, variables, nondet_tol=1e-8):
    print('1D AV Gradients Ok')

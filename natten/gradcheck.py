"""
Neighborhood Attention Gradcheck
Checks gradients computed by the extension's backwards kernels against finite differences
to ensure correctness.
WARNING: This script uses A LOT OF MEMORY, so choose sizes wisely.
The defaults provided should run on standard GPUs.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from nattencuda import NATTENAVFunction, NATTENQKRPBFunction
import torch
from torch.autograd import gradcheck
import argparse
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-n', '--heads', type=int, default=2)
parser.add_argument('-x', '--height', type=int, default=9)
parser.add_argument('-y', '--width', type=int, default=9)
parser.add_argument('-d', '--dim', type=int, default=32)
parser.add_argument('-k', '--kernel-size', type=int, default=7)
parser.add_argument('--slow', action='store_true', default=False)
args = parser.parse_args()
kernel_size = args.kernel_size
assert kernel_size > 1 and kernel_size % 2 == 1, \
    f"Kernel size must be an odd number greater than 1, got {kernel_size}."

for (dt, dtn, eps, atol, rtol, ndtol, fm) in [
    (torch.float64, 'DOUBLE PRECISION', 1e-06, 1e-05, 0.001, 1e-8, not args.slow),
    # (torch.float32, 'FULL PRECISION',   1e-05, 1e-04, 0.001, 1e-6, not args.slow),
    (torch.float16, 'HALF PRECISION',   5e-01, 1e-02, 0.001, 5e-1, not args.slow)
]:
    kwargs = {'dtype': dt,
              'device': 'cuda:0',
              'requires_grad': True}
    query = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)
    key = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)
    value = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)

    print(f"Verifying backward pass in {dtn}...")

    rpb = torch.randn((args.heads, 2 * kernel_size - 1, 2 * kernel_size - 1), **kwargs)
    variables = [query, key, rpb]

    if gradcheck(NATTENQKRPBFunction.apply, variables, eps=eps, atol=atol, rtol=rtol, nondet_tol=ndtol, fast_mode=fm):
        print('QK+RPB Gradients Ok')

    attn = torch.randn((args.batch_size, args.heads, args.height, args.width, kernel_size * kernel_size), **kwargs)
    variables = [attn, value]

    if gradcheck(NATTENAVFunction.apply, variables, eps=eps, atol=atol, rtol=rtol, nondet_tol=0, fast_mode=fm):
        print('AV Gradients Ok')

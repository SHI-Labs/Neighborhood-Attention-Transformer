from natten import NATTENAVFunction, NATTENQKRPBFunction
import torch
from torch.autograd import gradcheck
import argparse

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-n', '--heads', type=int, default=2)
parser.add_argument('-x', '--height', type=int, default=9)
parser.add_argument('-y', '--width', type=int, default=9)
parser.add_argument('-d', '--dim', type=int, default=32)
parser.add_argument('-k', '--kernel-size', type=int, default=7)
args = parser.parse_args()
kernel_size = args.kernel_size

kwargs = {'dtype': torch.float64,
          'device': 'cuda:0',
          'requires_grad': True}
query = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)
key = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)
value = torch.randn((args.batch_size, args.heads, args.height, args.width, args.dim), **kwargs)


print("Verifying backward pass...")

attn = torch.randn((args.batch_size, args.heads, args.height, args.width, kernel_size * kernel_size), **kwargs)
variables = [attn, value]

if gradcheck(NATTENAVFunction.apply, variables, nondet_tol=1e-8):
    print('AV Gradients Ok')


rpb = torch.randn((args.heads, 2 * kernel_size - 1, 2 * kernel_size - 1), **kwargs)
variables = [query, key, rpb]

if gradcheck(NATTENQKRPBFunction.apply, variables, nondet_tol=1e-8):
    print('QK+RPB Gradients Ok')

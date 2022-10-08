import torch
from fvcore.nn import FlopCountAnalysis
from natten.flops import add_natten_handle


def get_gflops(model, input, disable_warnings=False):
    flop_ctr = FlopCountAnalysis(model, input)
    flop_ctr = add_natten_handle(flop_ctr)
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total() / 1e9


def get_mparams(model, **kwargs):
    return sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6

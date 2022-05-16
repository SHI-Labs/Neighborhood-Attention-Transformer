import torch
from fvcore.nn import FlopCountAnalysis
from natten.flops import qk_rpb_flop, av_flop, qk_1d_rpb_flop, av_1d_flop


def get_gflops(model, input, disable_warnings=False):
    flop_ctr = FlopCountAnalysis(model, input)
    flop_ctr = flop_ctr.set_op_handle(
        **{
            "prim::PythonOp.NATTENQKRPBFunction": qk_rpb_flop,
            "prim::PythonOp.NATTENAVFunction": av_flop,
            "prim::PythonOp.NATTEN1DQKRPBFunction": qk_1d_rpb_flop,
            "prim::PythonOp.NATTEN1DAVFunction": av_1d_flop,
        })
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total() / 1e9


def get_imagenet_gflops(model, img_size=224, disable_warnings=False, device='cpu'):
    flop_ctr = FlopCountAnalysis(model, torch.randn(1, 3, img_size, img_size).to(device))
    flop_ctr = flop_ctr.set_op_handle(
        **{
            "prim::PythonOp.NATTENQKRPBFunction": qk_rpb_flop,
            "prim::PythonOp.NATTENAVFunction": av_flop
        })
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total() / 1e9


def get_mparams(model, **kwargs):
    return sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6


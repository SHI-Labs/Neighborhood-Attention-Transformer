import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from fvcore.nn.jit_handles import get_shape


def qk_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the QKRPB kernel.
    """
    assert len(inputs) == 3, f"Expected 3 inputs (query, key, rpb), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert len(input_shapes[0]) == 5, f"Query must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert len(input_shapes[1]) == 5, f"Key must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert len(input_shapes[2]) == 3, f"RelPosBias must be a 3-dim tensor, got {len(input_shapes[2])}"
    assert len(output_shapes[0]) == 5, f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert input_shapes[0] == input_shapes[1], f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, height, width, dim = input_shapes[0]
    batch_size, heads, height, width, kernel_size_sq = output_shapes[0]

    flops = batch_size * heads * height * width * dim * kernel_size_sq
    flops += batch_size * heads * height * width * kernel_size_sq
    return flops


def av_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the AV kernel.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert len(input_shapes[0]) == 5, f"Attn must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert len(input_shapes[1]) == 5, f"Value must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert len(output_shapes[0]) == 5, f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], f"Out and Value shapes did not match! O: {output_shapes[0]}, V: {input_shapes[1]}"
    batch_size, heads, height, width, kernel_size_sq = input_shapes[0]
    batch_size, heads, height, width, dim = output_shapes[0]
    flops = batch_size * heads * height * width * dim * kernel_size_sq
    return flops

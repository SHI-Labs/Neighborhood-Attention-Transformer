/*
NATTEN-AV TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor nattenav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

torch::Tensor nattenav_cuda_forward_fp16(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

// CUDA backward declarations
std::vector<torch::Tensor> nattenav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

std::vector<torch::Tensor> nattenav_cuda_backward_fp16(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

std::vector<torch::Tensor> nattenav_cuda_backward_tiled_32(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

std::vector<torch::Tensor> nattenav_cuda_backward_fp16_tiled_32(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor nattenav_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    CHECK_INPUT(attn);
    CHECK_INPUT(value);
    bool half = ::detail::scalar_type(value.scalar_type()) == at::ScalarType::Half;
    if (half)
        return nattenav_cuda_forward_fp16(attn, value, dilation);
    return nattenav_cuda_forward(attn, value, dilation);
}

std::vector<torch::Tensor> nattenav_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    CHECK_INPUT(d_out);
    CHECK_INPUT(attn);
    CHECK_INPUT(value);
    int dim = value.size(4);
    int kernel_size = sqrt(attn.size(4));
    bool half = ::detail::scalar_type(value.scalar_type()) == at::ScalarType::Half;
    if ((kernel_size == 7 || kernel_size == 5 || kernel_size == 9 || kernel_size == 11 || kernel_size == 13) && dim == 32){
        if (half)
            return nattenav_cuda_backward_fp16_tiled_32(d_out, attn, value, dilation);
        return nattenav_cuda_backward_tiled_32(d_out, attn, value, dilation);
    }
    if (half)
        return nattenav_cuda_backward_fp16(d_out, attn, value, dilation);
    return nattenav_cuda_backward(d_out, attn, value, dilation);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nattenav_forward, "NATTENAV forward (CUDA)");
  m.def("backward", &nattenav_backward, "NATTENAV backward (CUDA)");
}

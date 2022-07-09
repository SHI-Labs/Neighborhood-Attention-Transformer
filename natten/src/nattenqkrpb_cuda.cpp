/*
NATTEN-QKRPB TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor nattenqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

torch::Tensor nattenqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

torch::Tensor nattenqkrpb_cuda_forward_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

torch::Tensor nattenqkrpb_cuda_forward_fp16_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

// CUDA backward declarations
std::vector<torch::Tensor> nattenqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key);

std::vector<torch::Tensor> nattenqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor nattenqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    CHECK_INPUT(rpb);
    int dim = query.size(4);
    int kernel_size = (rpb.size(1) + 1) / 2;
    bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
    if ((kernel_size == 7 || kernel_size == 5 || kernel_size == 9 || kernel_size == 11 || kernel_size == 13) && dim == 32){
        if (half)
            return nattenqkrpb_cuda_forward_fp16_tiled_32(query, key, rpb);
        return nattenqkrpb_cuda_forward_tiled_32(query, key, rpb);
    }
    if (half)
        return nattenqkrpb_cuda_forward_fp16(query, key, rpb);
    return nattenqkrpb_cuda_forward(query, key, rpb);
}

std::vector<torch::Tensor> nattenqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
    CHECK_INPUT(d_attn);
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
    if (half)
        return nattenqkrpb_cuda_backward_fp16(d_attn, query, key);
    return nattenqkrpb_cuda_backward(d_attn, query, key);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nattenqkrpb_forward, "NATTENQK+RPB forward (CUDA)");
  m.def("backward", &nattenqkrpb_backward, "NATTENQK+RPB backward (CUDA)");
}

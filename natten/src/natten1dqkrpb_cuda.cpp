/*
NATTEN1D-QKRPB TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor natten1dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

// CUDA backward declarations
std::vector<torch::Tensor> natten1dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor natten1dqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(rpb);
  return natten1dqkrpb_cuda_forward(query, key, rpb);
}

std::vector<torch::Tensor> natten1dqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
  CHECK_INPUT(d_attn);
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  return natten1dqkrpb_cuda_backward(d_attn, query, key);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &natten1dqkrpb_forward, "NATTEN1DQK+RPB forward (CUDA)");
  m.def("backward", &natten1dqkrpb_backward, "NATTEN1DQK+RPB backward (CUDA)");
}

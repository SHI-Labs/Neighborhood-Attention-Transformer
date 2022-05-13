/*
NATTEN1D-AV TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor natten1dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value);

// CUDA backward declaration
std::vector<torch::Tensor> natten1dav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor natten1dav_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
  CHECK_INPUT(attn);
  CHECK_INPUT(value);
  return natten1dav_cuda_forward(attn, value);
}

std::vector<torch::Tensor> natten1dav_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
  CHECK_INPUT(d_out);
  CHECK_INPUT(attn);
  CHECK_INPUT(value);
  return natten1dav_cuda_backward(d_out, attn, value);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &natten1dav_forward, "NATTEN1DAV forward (CUDA)");
  m.def("backward", &natten1dav_backward, "NATTEN1DAV backward (CUDA)");
}

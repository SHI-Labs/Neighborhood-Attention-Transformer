/*
NATTEN1D-AV TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}


template <typename scalar_t, typename accscalar_t>
__global__ void natten1dav_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> out,
    const int length,
    const int KERNEL_SIZE,
    const int heads,
    const int dim,
    const int totalElements) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/length;
        const int i = indtmp1 - indtmp2 * length;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int NEIGHBORHOOD_SIZE = KERNEL_SIZE / 2;
        int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= length) * (length - i - NEIGHBORHOOD_SIZE - 1);
        accscalar_t updt = accscalar_t(0);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        const int valueOffset = b * value.stride(0) + h * value.stride(1) + d * value.stride(3);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int valueIndex = valueOffset + xi * value.stride(2);
            updt += static_cast<accscalar_t>(attn.data()[attnOffset]) * static_cast<accscalar_t>(value.data()[valueIndex]);
            ++attnOffset;
        }
        out.data()[linearIndex] = static_cast<scalar_t>(updt);
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void natten1da_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
    const int length,
    const int KERNEL_SIZE,
    const int batch_size,
    const int heads,
    const int dim) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int NEIGHBORHOOD_SIZE = KERNEL_SIZE / 2;
                int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= length) * (length - i - NEIGHBORHOOD_SIZE - 1);
                accscalar_t updt = accscalar_t(0);
                const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
                const int outOffset = batchHeadOffset + i * d_out.stride(2);
                const int valueOffset = batchHeadOffset + (ki+ni) * value.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += static_cast<accscalar_t>(d_out.data()[outOffset+dimOffset]) * static_cast<accscalar_t>(value.data()[valueOffset+dimOffset]);
                const int index = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + ki * d_attn.stride(3);
                d_attn.data()[index] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <typename scalar_t>
__global__ void natten1dv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length,
    const int KERNEL_SIZE,
    const int heads,
    const int dim,
    const int d_value_numel) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_value_numel){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/length;
        const int i = indtmp1 - indtmp2 * length;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int NEIGHBORHOOD_SIZE = KERNEL_SIZE / 2;
        int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= length) * (length - i - NEIGHBORHOOD_SIZE - 1);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        const int valueOffset = b * d_out.stride(0) + h * d_out.stride(1) + d * d_out.stride(3); 
        const int outOffset = valueOffset + i * d_out.stride(2);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int valueIndex = xi * d_value.stride(2) + valueOffset;
            at::native::fastAtomicAdd(d_value.data(), valueIndex, d_value_numel, attn.data()[attnOffset] * d_out.data()[outOffset], true);
            ++attnOffset;
        }
    }
}

torch::Tensor natten1dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t length = value.size(2);
    int64_t dim = value.size(3);
    int64_t KERNEL_SIZE = attn.size(3);

    auto out = torch::zeros({batch_size, heads, length, dim}, value.options());

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, value.scalar_type(), "natten1dav_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        natten1dav_cuda_forward_kernel<scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, length, KERNEL_SIZE, heads, dim, n);
    }));
    return out;
}

std::vector<torch::Tensor> natten1dav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t length = value.size(2);
    int64_t dim = value.size(3);
    int64_t KERNEL_SIZE = attn.size(3);
    int zsize = batch_size * heads;
    int SEQUENCETHREADS = 8;
    int BATCHTHREADS = 32;
    while (zsize < (BATCHTHREADS >> 1))
    {
        BATCHTHREADS = BATCHTHREADS >> 1;
    }
    int KERNELTHREADS = 1024 / (BATCHTHREADS * SEQUENCETHREADS);

    auto d_attn = torch::zeros(
            {batch_size, heads, length, KERNEL_SIZE}, attn.options());
    auto d_value = torch::zeros(
            {batch_size, heads, length, dim}, value.options());

    const dim3 attn_blocks(
            (length + SEQUENCETHREADS - 1) / SEQUENCETHREADS,
            (KERNEL_SIZE + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(SEQUENCETHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_attn.scalar_type(), "natten1dav_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        natten1da_cuda_backward_kernel<scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, length, KERNEL_SIZE,
                batch_size, heads, dim);
        natten1dv_cuda_backward_kernel<scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, length, KERNEL_SIZE,
                heads, dim, n_value);
    }));
    return {d_attn, d_value};
}

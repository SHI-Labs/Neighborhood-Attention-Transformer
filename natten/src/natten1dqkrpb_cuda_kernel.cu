/*
NATTEN1D-QKRPB TORCH EXTENSION (CUDA)

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
__global__ void natten1dqkrpb_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
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
                int pi = NEIGHBORHOOD_SIZE;
                int ni = i - NEIGHBORHOOD_SIZE;
                if (ni < 0)
                {
                    ni = 0;
                    pi = KERNEL_SIZE - 1 - i;
                }
                else if (i + NEIGHBORHOOD_SIZE >= length)
                {
                    ni = length - KERNEL_SIZE;
                    pi = length - i - 1;
                }
                accscalar_t updt = accscalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2);
                const int keyOffset = batchHeadOffset + (ki+ni) * key.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += static_cast<accscalar_t>(query.data()[queryOffset+dimOffset]) * static_cast<accscalar_t>(key.data()[keyOffset+dimOffset]);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki * attn.stride(3);
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1);
                updt += rpb.data()[rpbIndex];
                attn.data()[index] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void natten1dq_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
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
        accscalar_t d_query_update = accscalar_t(0);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2);
        const int keyOffset = b * key.stride(0) + h * key.stride(1) + d * key.stride(3);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int keyIndex = keyOffset + xi * key.stride(2);
            d_query_update += static_cast<accscalar_t>(d_attn.data()[attnOffset]) * static_cast<accscalar_t>(key.data()[keyIndex]);
            ++attnOffset;
        }
        d_query.data()[linearIndex] = static_cast<scalar_t>(d_query_update);
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void natten1drpb_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const int length,
    const int KERNEL_SIZE,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int ki = linearIndex - indtmp1 * KERNEL_SIZE;
        const int h = indtmp1/length;
        const int i = indtmp1 - h * length;
        const int NEIGHBORHOOD_SIZE = KERNEL_SIZE / 2;
        const int pi = NEIGHBORHOOD_SIZE + (i < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - i) + (i + NEIGHBORHOOD_SIZE >= length) * (length - i - 1 - NEIGHBORHOOD_SIZE);
        accscalar_t d_rpb_update = accscalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + ki * d_attn.stride(3);
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += static_cast<accscalar_t>(d_attn.data()[attnOffset]);
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <typename scalar_t>
__global__ void natten1dk_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const int length,
    const int KERNEL_SIZE,
    const int heads,
    const int dim,
    const int d_key_numel) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
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
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2);
        const int keyOffset = b * d_key.stride(0) + h * d_key.stride(1) + d * d_key.stride(3);
        const int queryOffset = keyOffset + i * query.stride(2);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int keyIndex = keyOffset + xi * d_key.stride(2);
            at::native::fastAtomicAdd(d_key.data(), keyIndex, d_key_numel, query.data()[queryOffset] * d_attn.data()[attnOffset], true);
            ++attnOffset;
        }
    }
}

torch::Tensor natten1dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t RPB_MAX = rpb.size(1);
    int64_t KERNEL_SIZE = (RPB_MAX + 1) / 2;
    int zsize = batch_size * heads;
    int SEQUENCETHREADS = 8;
    int BATCHTHREADS = 32;
    while (zsize < (BATCHTHREADS >> 1))
    {
        BATCHTHREADS = BATCHTHREADS >> 1;
    }
    int KERNELTHREADS = 1024 / (BATCHTHREADS * SEQUENCETHREADS);


    auto attn = torch::zeros(
            {batch_size, heads, length, KERNEL_SIZE}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (length + SEQUENCETHREADS - 1) / SEQUENCETHREADS,
            (KERNEL_SIZE + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(SEQUENCETHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(), "natten1dqk_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        natten1dqkrpb_cuda_forward_kernel<scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, length, KERNEL_SIZE, batch_size, heads, dim);
    }));
    return attn;
}

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t KERNEL_SIZE = d_attn.size(3);
    int64_t RPB_MAX = KERNEL_SIZE * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros(
            {heads, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * length * KERNEL_SIZE;
    int blocks_rpb = GET_BLOCKS(n_rpb);
    dim3 grid_rpb(blocks_rpb);
    int32_t n_query = d_query.numel();
    int blocks_query = GET_BLOCKS(n_query);
    dim3 grid_query(blocks_query);
    int32_t n_key = d_key.numel();
    int blocks_key = GET_BLOCKS(n_key);
    dim3 grid_key(blocks_key);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_rpb.scalar_type(), "natten1dqkrpb_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        natten1drpb_cuda_backward_kernel<scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, length, KERNEL_SIZE, batch_size, d_rpb.numel(), n_rpb);
        natten1dq_cuda_backward_kernel<scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, length, KERNEL_SIZE, heads, dim, n_query);
        natten1dk_cuda_backward_kernel<scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, length, KERNEL_SIZE, heads, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

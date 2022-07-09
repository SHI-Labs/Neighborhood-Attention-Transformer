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
#include <cuda_fp16.h>
#include "natten_commons.cuh"

#define CUDA_NUM_THREADS_Q 512
#define CUDA_NUM_THREADS_K 512
#define CUDA_NUM_THREADS_RPB 64
#define CUDA_NUM_THREADS_Q16 512
#define CUDA_NUM_THREADS_K16 256
#define CUDA_NUM_THREADS_RPB16 64


template <int KS, int NS, typename scalar_t>
__global__ void natten1dqkrpb_cuda_forward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dimhalf) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
                const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
                __half2 updt = __float2half2_rn(0.f);
                const int batchHeadOffset = b * (dimhalf*length*heads) + h * (dimhalf*length);
                const int queryOffset = batchHeadOffset + i * dimhalf;
                const int keyOffset = batchHeadOffset + (ki+ni) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki;
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
            }
        }
    }
}


template <int KS, int NS, typename scalar_t>
__global__ void natten1dqkrpb_cuda_forward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dim) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
                const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2);
                const int keyOffset = batchHeadOffset + (ki+ni) * key.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki;
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1);
                updt += rpb.data()[rpbIndex];
                attn.data()[index] = updt;
            }
        }
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dq_cuda_backward_kernel_fp32(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dim,
    const int totalElements) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
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
        const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        scalar_t d_query_update = scalar_t(0);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2);
        const int keyOffset = b * key.stride(0) + h * key.stride(1) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int keyIndex = keyOffset + xi * key.stride(2);
            d_query_update += d_attn.data()[attnOffset] * key.data()[keyIndex];
            ++attnOffset;
        }
        d_query.data()[linearIndex] = d_query_update;
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dq_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dimhalf,
    const int totalElements) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        __half2* d_query2 = reinterpret_cast<__half2*>(d_query.data());
        __half2* key2 = reinterpret_cast<__half2*>(key.data());
        int indtmp1 = linearIndex/dimhalf;
        const int d = linearIndex - indtmp1 * dimhalf;
        int indtmp2 = indtmp1/length;
        const int i = indtmp1 - indtmp2 * length;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        __half2 d_query_update = __float2half2_rn(0.f);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2);
        const int keyOffset = b * (dimhalf * length * heads) + h * (dimhalf * length) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int keyIndex = keyOffset + xi * dimhalf;
            scalar_t a = d_attn.data()[attnOffset];
            d_query_update = __hfma2(__halves2half2(a, a), key2[keyIndex], d_query_update);
            ++attnOffset;
        }
        d_query2[linearIndex] = d_query_update;
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1drpb_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const int length,
    const int kernel_size_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int ki = linearIndex - indtmp1 * KERNEL_SIZE;
        const int h = indtmp1/length;
        const int i = indtmp1 - h * length;
        const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        float d_rpb_update = scalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += static_cast<float>(d_attn.data()[attnOffset]);
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1drpb_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const int length,
    const int kernel_size_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int ki = linearIndex - indtmp1 * KERNEL_SIZE;
        const int h = indtmp1/length;
        const int i = indtmp1 - h * length;
        const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        scalar_t d_rpb_update = scalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += static_cast<float>(d_attn.data()[attnOffset]);
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dk_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dimhalf,
    const int d_key_numel) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
        __half2* d_key2 = reinterpret_cast<__half2*>(d_key.data());
        __half2* query2 = reinterpret_cast<__half2*>(query.data());
        int indtmp1 = linearIndex/dimhalf;
        const int d = linearIndex - indtmp1 * dimhalf;
        int indtmp2 = indtmp1/length;
        const int i = indtmp1 - indtmp2 * length;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int ei = get_backward_window_end(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
        const int queryOffset = b * (dimhalf * length * heads) + h * (dimhalf * length) + d;
        __half2 d_key_update = __float2half2_rn(0.f);
        #pragma unroll
        for (int xi=ni; xi < ei; ++xi){
            const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
            const int queryIndex = queryOffset + xi * dimhalf;
            const int attnIndex = attnOffset + xi * d_attn.stride(2) + (i-oni);
            scalar_t a = d_attn.data()[attnIndex];
            d_key_update = __hfma2(query2[queryIndex], __halves2half2(a, a), d_key_update);
        }
        d_key2[linearIndex] = d_key_update;
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dk_cuda_backward_kernel_fp32(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dim,
    const int d_key_numel) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
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
        const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int ei = get_backward_window_end(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
        const int queryOffset = b * query.stride(0) + h * query.stride(1) + d;
        scalar_t d_key_update = scalar_t(0);
        #pragma unroll
        for (int xi=ni; xi < ei; ++xi){
            const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
            const int queryIndex = queryOffset + xi * query.stride(2);
            const int attnIndex = attnOffset + xi * d_attn.stride(2) + (i-oni);
            d_key_update += query.data()[queryIndex] * d_attn.data()[attnIndex];
        }
        d_key.data()[linearIndex] = d_key_update;
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
    int kernel_size = (RPB_MAX + 1) / 2;
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten1dqkrpb_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dqkrpb_cuda_forward_kernel_fp32,
                blocks, threads, 0, stream, 
                query_a, key_a, rpb_a, attn_a, length, batch_size, heads, kernel_size, dim);
    }));
    return attn;
}

torch::Tensor natten1dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dimhalf = query.size(3) / 2;
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size);
    TORCH_CHECK(dimhalf*2 == query.size(3), "Dims per head must be an even number in FP16.");
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_HALF_TYPES(at::kHalf, query.scalar_type(), "natten1dqkrpb_cuda_forward_fp16", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dqkrpb_cuda_forward_kernel_fp16,
                blocks, threads, 0, stream, 
                query_a, key_a, rpb_a, attn_a, length, batch_size, heads, kernel_size, dimhalf);
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
    int kernel_size = d_attn.size(3);
    CHECK_SEQUENCE(length, kernel_size);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros({heads, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * length * kernel_size;
    int blocks_rpb = GET_BLOCKS(n_rpb, CUDA_NUM_THREADS_RPB);
    dim3 grid_rpb(blocks_rpb);
    dim3 blockr(CUDA_NUM_THREADS_RPB);
    int32_t n_query = d_query.numel();
    int blocks_query = GET_BLOCKS(n_query, CUDA_NUM_THREADS_Q);
    dim3 grid_query(blocks_query);
    dim3 blockq(CUDA_NUM_THREADS_Q);
    int32_t n_key = d_key.numel();
    int blocks_key = GET_BLOCKS(n_key, CUDA_NUM_THREADS_K);
    dim3 grid_key(blocks_key);
    dim3 blockk(CUDA_NUM_THREADS_K);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten1dqkrpb_backward_cuda", ([&] {
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1drpb_cuda_backward_kernel, grid_rpb, blockr, 0, stream,
                d_rpb_a, d_attn_a, length, kernel_size, batch_size, d_rpb.numel(), n_rpb);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dq_cuda_backward_kernel_fp32, grid_query, blockq, 0, stream,
                d_query_a, d_attn_a, key_a, length, heads, kernel_size, dim, n_query);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dk_cuda_backward_kernel_fp32, grid_key, blockk, 0, stream,
                d_key_a, d_attn_a, query_a, length, heads, kernel_size, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dimhalf = query.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == query.size(3), "Dims per head must be an even number in FP16.");
    int64_t kernel_size = d_attn.size(3);
    CHECK_SEQUENCE(length, kernel_size);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros({heads, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * length * kernel_size;
    int blocks_rpb = GET_BLOCKS(n_rpb, CUDA_NUM_THREADS_RPB16);
    dim3 grid_rpb(blocks_rpb);
    dim3 blockr(CUDA_NUM_THREADS_RPB16);
    int32_t nhalf_query = d_query.numel() / 2;
    int blocks_query = GET_BLOCKS(nhalf_query, CUDA_NUM_THREADS_Q16);
    dim3 grid_query(blocks_query);
    dim3 blockq(CUDA_NUM_THREADS_Q16);
    int32_t nhalf_key = d_key.numel() / 2;
    int blocks_key = GET_BLOCKS(nhalf_key, CUDA_NUM_THREADS_K16);
    dim3 grid_key(blocks_key);
    dim3 blockk(CUDA_NUM_THREADS_K16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_query.scalar_type(), "natten1dqkrpb_backward_cuda_fp16", ([&] {
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1drpb_cuda_backward_kernel_fp16, grid_rpb, blockr, 0, stream,
                d_rpb_a, d_attn_a, length, kernel_size, batch_size, d_rpb.numel(), n_rpb);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dq_cuda_backward_kernel_fp16, grid_query, blockq, 0, stream,
                d_query_a, d_attn_a, key_a, length, heads, kernel_size, dimhalf, nhalf_query);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dk_cuda_backward_kernel_fp16, grid_key, blockk, 0, stream,
                d_key_a, d_attn_a, query_a, length, heads, kernel_size, dimhalf, nhalf_key);
    }));
    return {d_query, d_key, d_rpb};
}

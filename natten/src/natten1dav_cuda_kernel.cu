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
#include <cuda_fp16.h>
#include "natten_commons.cuh"

#define CUDA_NUM_THREADS_F 512
#define CUDA_NUM_THREADS_FP16 512
#define CUDA_NUM_THREADS_V 512
#define CUDA_NUM_THREADS_V16 256


template <int KS, int NS, typename scalar_t>
__global__ void natten1dav_cuda_forward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> out,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dimhalf,
    const int totalElements) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        __half2* value2 = reinterpret_cast<__half2*>(value.data());
        __half2* out2 = reinterpret_cast<__half2*>(out.data());
        int indtmp1 = linearIndex/dimhalf;
        const int d = linearIndex - indtmp1 * dimhalf;
        int indtmp2 = indtmp1/length;
        const int i = indtmp1 - indtmp2 * length;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;

        const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        __half2 updt = __float2half2_rn(0.f);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        const int valueOffset = b * (dimhalf * length * heads) + h * (dimhalf * length) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int valueIndex = valueOffset + xi * dimhalf;
            scalar_t a = attn.data()[attnOffset];
            updt = __hfma2(__halves2half2(a, a), value2[valueIndex], updt);
            ++attnOffset;
        }
        out2[linearIndex] = updt;
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dav_cuda_forward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> out,
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
        scalar_t updt = scalar_t(0);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        const int valueOffset = b * value.stride(0) + h * value.stride(1) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi){
            const int valueIndex = valueOffset + xi * value.stride(2);
            updt += attn.data()[attnOffset] * value.data()[valueIndex];
            ++attnOffset;
        }
        out.data()[linearIndex] = updt;
    }
}


template <int KS, int NS, typename scalar_t>
__global__ void natten1da_cuda_backward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
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
                __half2* d_out2 = reinterpret_cast<__half2*>(d_out.data());
                __half2* value2 = reinterpret_cast<__half2*>(value.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
                __half2 updt = __float2half2_rn(0.f);
                const int batchHeadOffset = b * (dimhalf*length*heads) + h * (dimhalf*length);
                const int d_outOffset = batchHeadOffset + i * dimhalf;
                const int valueOffset = batchHeadOffset + (ki+ni) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(d_out2[d_outOffset+dimOffset], value2[valueOffset+dimOffset], updt);
                const int index = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
                d_attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
            }
        }
    }
}


template <int KS, int NS, typename scalar_t>
__global__ void natten1da_cuda_backward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> value,
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
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
                const int d_outOffset = batchHeadOffset + i * d_out.stride(2);
                const int valueOffset = batchHeadOffset + (ki+ni) * value.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += d_out.data()[d_outOffset+dimOffset] * value.data()[valueOffset+dimOffset];
                const int index = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
                d_attn.data()[index] = updt;
            }
        }
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dv_cuda_backward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dim,
    const int d_value_numel) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
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
        const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int ei = get_backward_window_end(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
        const int attnOffset = b * attn.stride(0) + h * attn.stride(1);
        const int outOffset = b * d_out.stride(0) + h * d_out.stride(1) + d;
        scalar_t d_value_update = scalar_t(0);
        #pragma unroll
        for (int xi=ni; xi < ei; ++xi){
            const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
            const int outIndex = outOffset + xi * d_out.stride(2);
            const int attnIndex = attnOffset + xi * attn.stride(2) + (i-oni);
            d_value_update += d_out.data()[outIndex] * attn.data()[attnIndex];
        }
        d_value.data()[linearIndex] = d_value_update;
    }
}

template <int KS, int NS, typename scalar_t>
__global__ void natten1dv_cuda_backward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dimhalf,
    const int d_value_numel) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_value_numel){
        __half2* d_out2 = reinterpret_cast<__half2*>(d_out.data());
        __half2* d_value2 = reinterpret_cast<__half2*>(d_value.data());
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
        const int attnOffset = b * attn.stride(0) + h * attn.stride(1);
        const int outOffset = b * (dimhalf * length * heads) + h * (dimhalf * length) + d;
        __half2 d_value_update = __float2half2_rn(0.f);
        #pragma unroll
        for (int xi=ni; xi < ei; ++xi){
            const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
            const int outIndex = outOffset + xi * dimhalf;
            const int attnIndex = attnOffset + xi * attn.stride(2) + (i-oni);
            scalar_t a = attn.data()[attnIndex];
            d_value_update = __hfma2(d_out2[outIndex], __halves2half2(a, a), d_value_update);
        }
        d_value2[linearIndex] = d_value_update;
    }
}

torch::Tensor natten1dav_cuda_forward_fp16(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dimhalf = value.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(3), "Dims per head must be an even number in FP16.");
    int kernel_size = attn.size(3);
    CHECK_SEQUENCE(length, kernel_size);

    auto out = torch::zeros_like(value);

    int32_t nhalf = out.numel() / 2;
    int blocks = GET_BLOCKS(nhalf, CUDA_NUM_THREADS_FP16);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_FP16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, value.scalar_type(), "natten1dav_forward_cuda_fp16", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dav_cuda_forward_kernel_fp16, grid, block, 0, stream,
                attn_a, value_a, out_a, length, heads, kernel_size, dimhalf, nhalf);
    }));
    return out;
}

torch::Tensor natten1dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dim = value.size(3);
    int kernel_size = attn.size(3);
    CHECK_SEQUENCE(length, kernel_size);

    auto out = torch::zeros_like(value);

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n, CUDA_NUM_THREADS_F);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_F);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten1dav_forward_cuda", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dav_cuda_forward_kernel_fp32, grid, block, 0, stream,
                attn_a, value_a, out_a, length, heads, kernel_size, dim, n);
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
    int kernel_size = attn.size(3);
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten1dav_backward_cuda", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1da_cuda_backward_kernel_fp32, attn_blocks, attn_threads, 0, stream,
                d_out_a, d_attn_a, value_a, length, batch_size, heads, kernel_size, dim);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dv_cuda_backward_kernel_fp32, grid_value, block, 0, stream,
                d_out_a, d_value_a, attn_a, length, heads, kernel_size, dim, n_value);
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten1dav_cuda_backward_fp16(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t length = value.size(2);
    int64_t dimhalf = value.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(3), "Dims per head must be an even number in FP16.");
    int kernel_size = attn.size(3);
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t nhalf_value = d_value.numel() / 2;
    int blocks_value = GET_BLOCKS(nhalf_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_attn.scalar_type(), "natten1dav_backward_cuda_fp16", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, natten1da_cuda_backward_kernel_fp16, attn_blocks, attn_threads, 0, stream,
                d_out_a, d_attn_a, value_a, length, batch_size, heads, kernel_size, dimhalf);
        LAUNCH_DNA_KNS_1D(kernel_size, natten1dv_cuda_backward_kernel_fp16, grid_value, block, 0, stream,
                d_out_a, d_value_a, attn_a, length, heads, kernel_size, dimhalf, nhalf_value);
    }));
    return {d_attn, d_value};
}

/*
NATTEN-AV TORCH EXTENSION (CUDA)

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


template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattenav_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> out,
    const int height,
    const int width,
    const int heads,
    const int dim,
    const int totalElements) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;

        const int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= height) * (height - i - NEIGHBORHOOD_SIZE - 1);
        const int nj = max(j - NEIGHBORHOOD_SIZE, 0) + (j + NEIGHBORHOOD_SIZE >= width) * (width - j - NEIGHBORHOOD_SIZE - 1);
        accscalar_t updt = accscalar_t(0);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3);
        const int valueOffset = b * value.stride(0) + h * value.stride(1) + d * value.stride(4);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE; ++xj){
                const int valueIndex = valueOffset + xi * value.stride(2) + xj * value.stride(3);
                updt += static_cast<accscalar_t>(attn.data()[attnOffset]) * static_cast<accscalar_t>(value.data()[valueIndex]);
                ++attnOffset;
            }
        out.data()[linearIndex] = static_cast<scalar_t>(updt);
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattena_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> value,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dim) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ki = y / KERNEL_SIZE;
                const int kj = y - ki * KERNEL_SIZE;
                const int i = x / width;
                const int j = x - i * width;
                int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= height) * (height - i - NEIGHBORHOOD_SIZE - 1);
                int nj = max(j - NEIGHBORHOOD_SIZE, 0) + (j + NEIGHBORHOOD_SIZE >= width) * (width - j - NEIGHBORHOOD_SIZE - 1);
                accscalar_t updt = accscalar_t(0);
                const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
                const int outOffset = batchHeadOffset + i * d_out.stride(2) + j * d_out.stride(3);
                const int valueOffset = batchHeadOffset + (ki+ni) * value.stride(2) + (kj+nj) * value.stride(3);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += static_cast<accscalar_t>(d_out.data()[outOffset+dimOffset]) * static_cast<accscalar_t>(value.data()[valueOffset+dimOffset]);
                const int index = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + y * d_attn.stride(4);
                d_attn.data()[index] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
    const int height,
    const int width,
    const int heads,
    const int dim,
    const int d_value_numel) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_value_numel){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;

        int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= height) * (height - i - NEIGHBORHOOD_SIZE - 1);
        int nj = max(j - NEIGHBORHOOD_SIZE, 0) + (j + NEIGHBORHOOD_SIZE >= width) * (width - j - NEIGHBORHOOD_SIZE - 1);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3);
        const int valueOffset = b * d_out.stride(0) + h * d_out.stride(1) + d * d_out.stride(4); 
        const int outOffset = valueOffset + i * d_out.stride(2) + j * d_out.stride(3);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE; ++xj){
                const int valueIndex = xi * d_value.stride(2) + xj * d_value.stride(3) + valueOffset;
                at::native::fastAtomicAdd(d_value.data(), valueIndex, d_value_numel, attn.data()[attnOffset] * d_out.data()[outOffset], true);
                ++attnOffset;
            }
    }
}

torch::Tensor nattenav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dim = value.size(4);
    int64_t KERNEL_SIZE_SQ = attn.size(4);

    auto out = torch::zeros({batch_size, heads, height, width, dim}, value.options());

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, value.scalar_type(), "nattenav_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenav_cuda_forward_kernel<7, 3, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    heads, dim, n);
        else if (KERNEL_SIZE_SQ == 25)
            nattenav_cuda_forward_kernel<5, 2, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    heads, dim, n);
        else if (KERNEL_SIZE_SQ == 9)
            nattenav_cuda_forward_kernel<3, 1, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    heads, dim, n);
        else if (KERNEL_SIZE_SQ == 81)
            nattenav_cuda_forward_kernel<9, 4, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    heads, dim, n);
        else if (KERNEL_SIZE_SQ == 121)
            nattenav_cuda_forward_kernel<11, 5, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    heads, dim, n);
    }));
    return out;
}

std::vector<torch::Tensor> nattenav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dim = value.size(4);
    int64_t KERNEL_SIZE_SQ = attn.size(4);
    int zsize = batch_size * heads;
    int xsize = height * width;
    int PIXELTHREADS = 4;
    int BATCHTHREADS = 32;
    while (zsize < (BATCHTHREADS >> 1))
    {
        BATCHTHREADS = BATCHTHREADS >> 1;
    }
    int KERNELTHREADS = 1024 / (BATCHTHREADS * PIXELTHREADS);

    auto d_attn = torch::zeros(
            {batch_size, heads, height, width, KERNEL_SIZE_SQ}, attn.options());
    auto d_value = torch::zeros(
            {batch_size, heads, height, width, dim}, value.options());

    const dim3 attn_blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (KERNEL_SIZE_SQ + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_attn.scalar_type(), "nattenav_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49) {
            nattena_cuda_backward_kernel<7, 3, scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads, dim);
            nattenv_cuda_backward_kernel<7, 3, scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    heads, dim, n_value);
        }
        else if (KERNEL_SIZE_SQ == 25) {
            nattena_cuda_backward_kernel<5, 2, scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads, dim);
            nattenv_cuda_backward_kernel<5, 2, scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    heads, dim, n_value);
        }
        else if (KERNEL_SIZE_SQ == 9) {
            nattena_cuda_backward_kernel<3, 1, scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads, dim);
            nattenv_cuda_backward_kernel<3, 1, scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    heads, dim, n_value);
        }
        else if (KERNEL_SIZE_SQ == 81) {
            nattena_cuda_backward_kernel<9, 4, scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads, dim);
            nattenv_cuda_backward_kernel<9, 4, scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    heads, dim, n_value);
        }
        else if (KERNEL_SIZE_SQ == 121) {
            nattena_cuda_backward_kernel<11, 5, scalar_t, accscalar_t><<<attn_blocks, attn_threads, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads, dim);
            nattenv_cuda_backward_kernel<11, 5, scalar_t><<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    heads, dim, n_value);
        }
    }));
    return {d_attn, d_value};
}

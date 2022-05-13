/*
NATTEN-QKRPB TORCH EXTENSION (CUDA)

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
__global__ void nattenqkrpb_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
                int pi = NEIGHBORHOOD_SIZE, pj = NEIGHBORHOOD_SIZE;
                int ni = i - NEIGHBORHOOD_SIZE;
                int nj = j - NEIGHBORHOOD_SIZE;
                if (ni < 0)
                {
                    ni = 0;
                    pi = KERNEL_SIZE - 1 - i;
                }
                else if (i + NEIGHBORHOOD_SIZE >= height)
                {
                    ni = height - KERNEL_SIZE;
                    pi = height - i - 1;
                }
                if (nj < 0)
                {
                    nj = 0;
                    pj = KERNEL_SIZE - 1 - j;
                }
                else if (j + NEIGHBORHOOD_SIZE >= width)
                {
                    nj = width - KERNEL_SIZE;
                    pj = width - j - 1;
                }
                accscalar_t updt = accscalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
                const int keyOffset = batchHeadOffset + (ki+ni) * key.stride(2) + (kj+nj) * key.stride(3);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += static_cast<accscalar_t>(query.data()[queryOffset+dimOffset]) * static_cast<accscalar_t>(key.data()[keyOffset+dimOffset]);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + y * attn.stride(4);
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
                updt += rpb.data()[rpbIndex];
                attn.data()[index] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattenq_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
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
        int ni = max(i - NEIGHBORHOOD_SIZE, 0) + (i + NEIGHBORHOOD_SIZE >= height) * (height - i - NEIGHBORHOOD_SIZE - 1);
        int nj = max(j - NEIGHBORHOOD_SIZE, 0) + (j + NEIGHBORHOOD_SIZE >= width) * (width - j - NEIGHBORHOOD_SIZE - 1);
        accscalar_t d_query_update = accscalar_t(0);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3);
        const int keyOffset = b * key.stride(0) + h * key.stride(1) + d * key.stride(4);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE; ++xj){
                const int keyIndex = keyOffset + xi * key.stride(2) + xj * key.stride(3);
                d_query_update += static_cast<accscalar_t>(d_attn.data()[attnOffset]) * static_cast<accscalar_t>(key.data()[keyIndex]);
                ++attnOffset;
            }
        d_query.data()[linearIndex] = static_cast<scalar_t>(d_query_update);
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattenrpb_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int kj = linearIndex - indtmp1 * KERNEL_SIZE;
        int indtmp2 = indtmp1/KERNEL_SIZE;
        const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        const int h = indtmp2;
        const int pi = NEIGHBORHOOD_SIZE + (i < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - i) + (i + NEIGHBORHOOD_SIZE >= height) * (height - i - 1 - NEIGHBORHOOD_SIZE);
        const int pj = NEIGHBORHOOD_SIZE + (j < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - j) + (j + NEIGHBORHOOD_SIZE >= width) * (width - j - 1 - NEIGHBORHOOD_SIZE);
        accscalar_t d_rpb_update = accscalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + (ki*KERNEL_SIZE+kj) * d_attn.stride(4);
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += static_cast<accscalar_t>(d_attn.data()[attnOffset]);
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1) + (pj+kj) * d_rpb.stride(2);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenk_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const int height,
    const int width,
    const int heads,
    const int dim,
    const int d_key_numel) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
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
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3);
        const int keyOffset = b * d_key.stride(0) + h * d_key.stride(1) + d * d_key.stride(4);
        const int queryOffset = keyOffset + i * query.stride(2) + j * query.stride(3);
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE; ++xi)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE; ++xj){
                const int keyIndex = keyOffset + xi * d_key.stride(2) + xj * d_key.stride(3);
                at::native::fastAtomicAdd(d_key.data(), keyIndex, d_key_numel, query.data()[queryOffset] * d_attn.data()[attnOffset], true);
                ++attnOffset;
            }
    }
}

torch::Tensor nattenqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int64_t RPB_MAX = rpb.size(1);
    int64_t KERNEL_SIZE_SQ = pow((RPB_MAX + 1) / 2, 2);
    int zsize = batch_size * heads;
    int xsize = height * width;
    int PIXELTHREADS = 4;
    int BATCHTHREADS = 32;
    while (zsize < (BATCHTHREADS >> 1))
    {
        BATCHTHREADS = BATCHTHREADS >> 1;
    }
    int KERNELTHREADS = 1024 / (BATCHTHREADS * PIXELTHREADS);


    auto attn = torch::zeros(
            {batch_size, heads, height, width, KERNEL_SIZE_SQ}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (KERNEL_SIZE_SQ + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(), "nattenqk_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenqkrpb_cuda_forward_kernel<7, 3, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dim);
        else if (KERNEL_SIZE_SQ == 25)
            nattenqkrpb_cuda_forward_kernel<5, 2, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dim);
        else if (KERNEL_SIZE_SQ == 9)
            nattenqkrpb_cuda_forward_kernel<3, 1, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dim);
        else if (KERNEL_SIZE_SQ == 81)
            nattenqkrpb_cuda_forward_kernel<9, 4, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dim);
        else if (KERNEL_SIZE_SQ == 121)
            nattenqkrpb_cuda_forward_kernel<11, 5, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dim);
    }));
    return attn;
}

std::vector<torch::Tensor> nattenqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int64_t KERNEL_SIZE_SQ = d_attn.size(4);
    int64_t RPB_MAX = sqrt(KERNEL_SIZE_SQ) * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros(
            {heads, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * height * width * KERNEL_SIZE_SQ;
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
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_rpb.scalar_type(), "nattenqkrpb_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49) {
            nattenrpb_cuda_backward_kernel<7, 3, scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, d_rpb.numel(), n_rpb);
            nattenq_cuda_backward_kernel<7, 3, scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, heads, dim, n_query);
            nattenk_cuda_backward_kernel<7, 3, scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, heads, dim, n_key);
            }
        else if (KERNEL_SIZE_SQ == 25) {
            nattenrpb_cuda_backward_kernel<5, 2, scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, d_rpb.numel(), n_rpb);
            nattenq_cuda_backward_kernel<5, 2, scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, heads, dim, n_query);
            nattenk_cuda_backward_kernel<5, 2, scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, heads, dim, n_key);
            }
        else if (KERNEL_SIZE_SQ == 9) {
            nattenrpb_cuda_backward_kernel<3, 1, scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, d_rpb.numel(), n_rpb);
            nattenq_cuda_backward_kernel<3, 1, scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, heads, dim, n_query);
            nattenk_cuda_backward_kernel<3, 1, scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, heads, dim, n_key);
            }
        else if (KERNEL_SIZE_SQ == 81) {
            nattenrpb_cuda_backward_kernel<9, 4, scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, d_rpb.numel(), n_rpb);
            nattenq_cuda_backward_kernel<9, 4, scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, heads, dim, n_query);
            nattenk_cuda_backward_kernel<9, 4, scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, heads, dim, n_key);
            }
        else if (KERNEL_SIZE_SQ == 121) {
            nattenrpb_cuda_backward_kernel<11, 5, scalar_t, accscalar_t><<<grid_rpb, block, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, d_rpb.numel(), n_rpb);
            nattenq_cuda_backward_kernel<11, 5, scalar_t, accscalar_t><<<grid_query, block, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, heads, dim, n_query);
            nattenk_cuda_backward_kernel<11, 5, scalar_t><<<grid_key, block, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, heads, dim, n_key);
            }
    }));
    return {d_query, d_key, d_rpb};
}

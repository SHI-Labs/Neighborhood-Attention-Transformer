#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define BATCHTHREADS 32
#define DIMTHREADS 4
#define PIXELTHREADS 8

// Backward Q threads
#define BATCHTHREADS_BACKWARD_Q 16
#define DIMTHREADS_BACKWARD_Q 8
#define PIXELTHREADS_BACKWARD_Q 4

// Backward K threads
#define BATCHTHREADS_BACKWARD_K 32
#define DIMTHREADS_BACKWARD_K 16
#define PIXELTHREADS_BACKWARD_K 2

#define RPB_PIXELTHREADS 32
#define RPB_DIMTHREADS 8
#define RPB_KERNELTHREADS 4

// Unrolls
#define BATCHUNROLL 2

#define DIMUNROLL_BACKWARD_Q 2
#define DIM_DIV_DIMUNROLL_BACKWARD_Q 16

#define DIMUNROLL_BACKWARD_K 2
#define DIM_DIV_DIMUNROLL_BACKWARD_K 16

#define DIM 32


template <int KERNEL_SIZE, int KERNEL_SIZE_SQ, int WIN_SIZE, int MID_CELL, typename scalar_t, typename accscalar_t>
__global__ void nattenqkrpb_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
    const int height,
    const int width,
    const int batch_size,
    const int heads) {
    // Batch index
    const int b = BATCHUNROLL * (blockIdx.z * blockDim.z + threadIdx.z);

    // Embedding index
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Head index
    const int h = y / KERNEL_SIZE_SQ;
    // Kernel index
    const int ki = (y / KERNEL_SIZE) % KERNEL_SIZE;
    const int kj = y % KERNEL_SIZE;

    // Pixel index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Feature map pixel index (x axis)
    const int i = x / width;
    // Feature map pixel index (y axis)
    const int j = x % width;

    // Neighbor window starting point
    int pi = WIN_SIZE, pj = WIN_SIZE;
    int ni = i - WIN_SIZE;
    int nj = j - WIN_SIZE;
    if (ni < 0)
    {
        ni = 0;
        pi = MID_CELL - i;
    }
    else if (i + WIN_SIZE >= height)
    {
        ni = height - KERNEL_SIZE;
        pi = height - i - 1;
    }
    if (nj < 0)
    {
        nj = 0;
        pj = MID_CELL - j;
    }
    else if (j + WIN_SIZE >= width)
    {
        nj = width - KERNEL_SIZE;
        pj = width - j - 1;
    }
    if (i < height && j < width && h < heads && ki < KERNEL_SIZE && kj < KERNEL_SIZE)
        for (int bo=0; bo < BATCHUNROLL && b + bo < batch_size; ++bo){
            accscalar_t updt = accscalar_t(0);
            for (int d=0; d < DIM; ++d)
                updt += query[b+bo][h][i][j][d] * key[b+bo][h][ki+ni][kj+nj][d];
            attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj] = updt + rpb[h][pi+ki][pj+kj];
        }
}

template <int KERNEL_SIZE, int WIN_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattenq_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const int height,
    const int width,
    const int batch_size,
    const int heads) {
    // Batch index
    const int b = BATCHUNROLL * (blockIdx.z * blockDim.z + threadIdx.z);

    // Embedding index
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Head index
    const int h = y / DIM_DIV_DIMUNROLL_BACKWARD_Q;
    // Dim index
    const int d = DIMUNROLL_BACKWARD_Q * (y % DIM_DIV_DIMUNROLL_BACKWARD_Q);

    // Pixel index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Feature map pixel index (x axis)
    const int i = x / width;
    // Feature map pixel index (y axis)
    const int j = x % width;

    // Neighbor window starting point
    int ni = max(i - WIN_SIZE, 0) + (i + WIN_SIZE >= height) * (height - i - WIN_SIZE - 1);
    int nj = max(j - WIN_SIZE, 0) + (j + WIN_SIZE >= width) * (width - j - WIN_SIZE - 1);
    if (i < height && j < width && h < heads)
        for (int bo=0; bo < BATCHUNROLL && b + bo < batch_size; ++bo)
            for (int doff=0; doff < DIMUNROLL_BACKWARD_Q && d + doff < DIM; ++doff)
            {
                accscalar_t d_query_update = accscalar_t(0);
                for (int ki=0, xi=ni; ki < KERNEL_SIZE; ++ki, ++xi)
                    for (int kj=0, xj=nj; kj < KERNEL_SIZE; ++kj, ++xj)
                        d_query_update += d_attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj] * key[b+bo][h][xi][xj][d+doff];
                d_query[b+bo][h][i][j][d+doff] = d_query_update;
            }
}

template <int KERNEL_SIZE, int WIN_SIZE, int MID_CELL, typename scalar_t, typename accscalar_t>
__global__ void nattenrpb_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int d_rpb_numel) {
    // Neighborhood index
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int ki = z / KERNEL_SIZE;
    const int kj = z % KERNEL_SIZE;

    // Head index
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    // Pixel index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Feature map pixel index (x axis)
    const int i = x / width;
    // Feature map pixel index (y axis)
    const int j = x % width;

    // Neighbor window starting point
    int pi = WIN_SIZE, pj = WIN_SIZE;
    if (i < WIN_SIZE)
        pi = MID_CELL - i;
    else if (i + WIN_SIZE >= height)
        pi = height - i - 1;
    if (j < WIN_SIZE)
        pj = MID_CELL - j;
    else if (j + WIN_SIZE >= width)
        pj = width - j - 1;
    if (i < height && j < width && h < heads && ki < KERNEL_SIZE && kj < KERNEL_SIZE) {
        accscalar_t d_rpb_update = accscalar_t(0);
        for (int b=0; b < batch_size && b < batch_size; ++b)
            d_rpb_update += d_attn[b][h][i][j][ki*KERNEL_SIZE+kj];
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1) + (pj+kj) * d_rpb.stride(2);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KERNEL_SIZE, int WIN_SIZE, typename scalar_t>
__global__ void nattenk_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int d_key_numel) {
    // Batch index
    const int b = BATCHUNROLL * (blockIdx.z * blockDim.z + threadIdx.z);

    // Embedding index
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Head index
    const int h = y / DIM_DIV_DIMUNROLL_BACKWARD_K;
    // Dim index
    const int d = DIMUNROLL_BACKWARD_K * (y % DIM_DIV_DIMUNROLL_BACKWARD_K);

    // Pixel index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Feature map pixel index (x axis)
    const int i = x / width;
    // Feature map pixel index (y axis)
    const int j = x % width;

    // Neighbor window starting point
    int ni = max(i - WIN_SIZE, 0) + (i + WIN_SIZE >= height) * (height - i - WIN_SIZE - 1);
    int nj = max(j - WIN_SIZE, 0) + (j + WIN_SIZE >= width) * (width - j - WIN_SIZE - 1);
    if (i < height && j < width && h < heads)
        for (int bo=0; bo < BATCHUNROLL && b + bo < batch_size; ++bo)
            for (int ki=0, xi=ni; ki < KERNEL_SIZE; ++ki, ++xi)
                for (int kj=0, xj=nj; kj < KERNEL_SIZE; ++kj, ++xj)
                    for (int doff=0; doff < DIMUNROLL_BACKWARD_K && d + doff < DIM; ++doff) { 
                        const int index = (b+bo) * d_key.stride(0) + h * d_key.stride(1) + xi * d_key.stride(2) + xj * d_key.stride(3) + (d+doff) * d_key.stride(4);
                        at::native::fastAtomicAdd(d_key.data(), index, d_key_numel, query[b+bo][h][i][j][d+doff] * d_attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj], true);
                    }
}

std::vector<torch::Tensor> nattenqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t RPB_MAX = rpb.size(1);
    int64_t KERNEL_SIZE_SQ = pow((RPB_MAX + 1) / 2, 2);

    auto attn = torch::zeros(
            {batch_size, heads, height, width, KERNEL_SIZE_SQ}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (height * width + PIXELTHREADS - 1) / PIXELTHREADS,
            (KERNEL_SIZE_SQ * heads + DIMTHREADS - 1) / DIMTHREADS,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, DIMTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(), "nattenqk_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenqkrpb_cuda_forward_kernel<7, 49, 3, 6, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 25)
            nattenqkrpb_cuda_forward_kernel<5, 25, 2, 4, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 9)
            nattenqkrpb_cuda_forward_kernel<3, 9, 1, 2, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 81)
            nattenqkrpb_cuda_forward_kernel<9, 81, 4, 8, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 121)
            nattenqkrpb_cuda_forward_kernel<11, 121, 5, 10, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads);
    }));
    return {attn};
}

std::vector<torch::Tensor> nattenqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t KERNEL_SIZE_SQ = d_attn.size(4);
    int64_t RPB_MAX = sqrt(KERNEL_SIZE_SQ) * 2 - 1;
   
    // Compute gradient for queries and keys
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    auto d_rpb = torch::zeros(
            {heads, RPB_MAX, RPB_MAX}, d_attn.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocksrpb(
            (height * width + RPB_PIXELTHREADS - 1) / RPB_PIXELTHREADS,
            (heads + RPB_DIMTHREADS - 1) / RPB_DIMTHREADS,
            (KERNEL_SIZE_SQ + RPB_KERNELTHREADS - 1) / RPB_KERNELTHREADS);
    const dim3 blocksq(
            (height * width + PIXELTHREADS_BACKWARD_Q - 1) / PIXELTHREADS_BACKWARD_Q,
            (DIM_DIV_DIMUNROLL_BACKWARD_Q * heads + DIMTHREADS_BACKWARD_Q - 1) / DIMTHREADS_BACKWARD_Q,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS_BACKWARD_Q - 1) / BATCHTHREADS_BACKWARD_Q);
    const dim3 blocksk(
            (height * width + PIXELTHREADS_BACKWARD_K - 1) / PIXELTHREADS_BACKWARD_K,
            (DIM_DIV_DIMUNROLL_BACKWARD_K * heads + DIMTHREADS_BACKWARD_K - 1) / DIMTHREADS_BACKWARD_K,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS_BACKWARD_K - 1) / BATCHTHREADS_BACKWARD_K);
    const dim3 threadsrpb(RPB_PIXELTHREADS, RPB_DIMTHREADS, RPB_KERNELTHREADS);
    const dim3 threadsq(PIXELTHREADS_BACKWARD_Q, DIMTHREADS_BACKWARD_Q, BATCHTHREADS_BACKWARD_Q);
    const dim3 threadsk(PIXELTHREADS_BACKWARD_K, DIMTHREADS_BACKWARD_K, BATCHTHREADS_BACKWARD_K);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_rpb.scalar_type(), "nattenrpb_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenrpb_cuda_backward_kernel<7, 3, 6, scalar_t, accscalar_t><<<blocksrpb, threadsrpb, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, heads, d_rpb.numel());
        else if (KERNEL_SIZE_SQ == 25)
            nattenrpb_cuda_backward_kernel<5, 2, 4, scalar_t, accscalar_t><<<blocksrpb, threadsrpb, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, heads, d_rpb.numel());
        else if (KERNEL_SIZE_SQ == 9)
            nattenrpb_cuda_backward_kernel<3, 1, 2, scalar_t, accscalar_t><<<blocksrpb, threadsrpb, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, heads, d_rpb.numel());
        else if (KERNEL_SIZE_SQ == 81)
            nattenrpb_cuda_backward_kernel<9, 4, 8, scalar_t, accscalar_t><<<blocksrpb, threadsrpb, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, heads, d_rpb.numel());
        else if (KERNEL_SIZE_SQ == 121)
            nattenrpb_cuda_backward_kernel<11, 5, 10, scalar_t, accscalar_t><<<blocksrpb, threadsrpb, 0, stream>>>(d_rpb_a, d_attn_a, height, width, batch_size, heads, d_rpb.numel());
    }));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_query.scalar_type(), "nattenq_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_query_a = d_query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenq_cuda_backward_kernel<7, 3, scalar_t, accscalar_t><<<blocksq, threadsq, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 25)
            nattenq_cuda_backward_kernel<5, 2, scalar_t, accscalar_t><<<blocksq, threadsq, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 9)
            nattenq_cuda_backward_kernel<3, 1, scalar_t, accscalar_t><<<blocksq, threadsq, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 81)
            nattenq_cuda_backward_kernel<9, 4, scalar_t, accscalar_t><<<blocksq, threadsq, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, batch_size, heads);
        else if (KERNEL_SIZE_SQ == 121)
            nattenq_cuda_backward_kernel<11, 5, scalar_t, accscalar_t><<<blocksq, threadsq, 0, stream>>>(d_query_a, d_attn_a, key_a, height, width, batch_size, heads);
    }));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_key.scalar_type(), "nattenk_backward_cuda", ([&] {
        auto d_key_a = d_key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenk_cuda_backward_kernel<7, 3, scalar_t><<<blocksk, threadsk, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, batch_size, heads, d_key.numel());
        else if (KERNEL_SIZE_SQ == 25)
            nattenk_cuda_backward_kernel<5, 2, scalar_t><<<blocksk, threadsk, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, batch_size, heads, d_key.numel());
        else if (KERNEL_SIZE_SQ == 9)
            nattenk_cuda_backward_kernel<3, 1, scalar_t><<<blocksk, threadsk, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, batch_size, heads, d_key.numel());
        else if (KERNEL_SIZE_SQ == 81)
            nattenk_cuda_backward_kernel<9, 4, scalar_t><<<blocksk, threadsk, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, batch_size, heads, d_key.numel());
        else if (KERNEL_SIZE_SQ == 121)
            nattenk_cuda_backward_kernel<11, 5, scalar_t><<<blocksk, threadsk, 0, stream>>>(d_key_a, d_attn_a, query_a, height, width, batch_size, heads, d_key.numel());
    }));
    return {d_query, d_key, d_rpb};
}

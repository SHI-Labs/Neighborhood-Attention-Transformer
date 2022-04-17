#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define BATCHTHREADS 2
#define DIMTHREADS 32
#define PIXELTHREADS 16

// Backward A threads
#define BATCHTHREADS_BACKWARD_A 2
#define DIMTHREADS_BACKWARD_A 32
#define PIXELTHREADS_BACKWARD_A 16

// Backward V threads
#define BATCHTHREADS_BACKWARD_V 2
#define DIMTHREADS_BACKWARD_V 32
#define PIXELTHREADS_BACKWARD_V 16

// Unrolls
#define BATCHUNROLL 1

#define DIMUNROLL_FORWARD 1
#define DIM_DIV_DIMUNROLL_FORWARD 32

#define DIMUNROLL_BACKWARD_A 1
#define DIM_DIV_DIMUNROLL_BACKWARD_A 32

#define DIMUNROLL_BACKWARD_V 1
#define DIM_DIV_DIMUNROLL_BACKWARD_V 32

#define DIM 32

template <int KERNEL_SIZE, int WIN_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattenav_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> out,
    const int height,
    const int width,
    const int batch_size,
    const int heads) {
    const int b = BATCHUNROLL * (blockIdx.z * blockDim.z + threadIdx.z);

    // Embedding index
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Head index
    const int h = y / DIM_DIV_DIMUNROLL_FORWARD;
    // Dim index
    const int d = DIMUNROLL_FORWARD * (y % DIM_DIV_DIMUNROLL_FORWARD);

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
        {
            for (int doff=0; doff < DIMUNROLL_FORWARD && d + doff < DIM; ++doff)
            {
                accscalar_t updt = accscalar_t(0);
                for (int ki=0, xi=ni; ki < KERNEL_SIZE; ++ki, ++xi)
                    for (int kj=0, xj=nj; kj < KERNEL_SIZE; ++kj, ++xj)
                        updt += attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj] * value[b+bo][h][xi][xj][d+doff];
                out[b+bo][h][i][j][d+doff] = updt;
            }
        }
}

template <int KERNEL_SIZE, int KERNEL_SIZE_SQ, int WIN_SIZE, typename scalar_t, typename accscalar_t>
__global__ void nattena_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> value,
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
    int ni = max(i - WIN_SIZE, 0) + (i + WIN_SIZE >= height) * (height - i - WIN_SIZE - 1);
    int nj = max(j - WIN_SIZE, 0) + (j + WIN_SIZE >= width) * (width - j - WIN_SIZE - 1);
    if (i < height && j < width && h < heads && ki < KERNEL_SIZE && kj < KERNEL_SIZE)
        for (int bo=0; bo < BATCHUNROLL && b + bo < batch_size; ++bo)
        {
            accscalar_t updt = accscalar_t(0);
            for (int d=0; d < DIM; ++d)
                updt += d_out[b+bo][h][i][j][d] * value[b+bo][h][ni+ki][nj+kj][d];
            d_attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj] = updt;
        }
}

template <int KERNEL_SIZE, int WIN_SIZE, typename scalar_t>
__global__ void nattenv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_out,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int d_value_numel) {
    // Batch index
    const int b = BATCHUNROLL * (blockIdx.z * blockDim.z + threadIdx.z);

    // Embedding index
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Head index
    const int h = y / DIM_DIV_DIMUNROLL_BACKWARD_V;
    // Dim index
    const int d = DIMUNROLL_BACKWARD_V * (y % DIM_DIV_DIMUNROLL_BACKWARD_V);

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
        for (int ki=0, xi=ni; ki < KERNEL_SIZE; ++ki, ++xi)
            for (int kj=0, xj=nj; kj < KERNEL_SIZE; ++kj, ++xj)
                for (int bo=0; bo < BATCHUNROLL && b + bo < batch_size; ++bo)
                    for (int doff=0; doff < DIMUNROLL_BACKWARD_V && d + doff < DIM; ++doff){
                        const int index = (b+bo) * d_value.stride(0) + h * d_value.stride(1) + xi * d_value.stride(2) + xj * d_value.stride(3) + (d+doff) * d_value.stride(4);
                        at::native::fastAtomicAdd(d_value.data(), index, d_value_numel, attn[b+bo][h][i][j][ki*KERNEL_SIZE+kj] * d_out[b+bo][h][i][j][d+doff], true);
                    }
}

std::vector<torch::Tensor> nattenav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t KERNEL_SIZE_SQ = attn.size(4);

    auto out = torch::zeros({batch_size, heads, height, width, DIM}, value.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (height * width + PIXELTHREADS - 1) / PIXELTHREADS,
            (DIM_DIV_DIMUNROLL_FORWARD * heads + DIMTHREADS - 1) / DIMTHREADS,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, DIMTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, value.scalar_type(), "nattenav_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenav_cuda_forward_kernel<7, 3, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 25)
            nattenav_cuda_forward_kernel<5, 2, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 9)
            nattenav_cuda_forward_kernel<3, 1, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 81)
            nattenav_cuda_forward_kernel<9, 4, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 121)
            nattenav_cuda_forward_kernel<11, 5, scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(attn_a, value_a, out_a, height, width,
                    batch_size, heads);
    }));
    return {out};
}

std::vector<torch::Tensor> nattenav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t KERNEL_SIZE_SQ = attn.size(4);

    // Compute gradient for queries and keys
    auto d_attn = torch::zeros(
            {batch_size, heads, height, width, KERNEL_SIZE_SQ}, attn.options());
    auto d_value = torch::zeros(
            {batch_size, heads, height, width, DIM}, value.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocksa(
            (height * width + PIXELTHREADS_BACKWARD_A - 1) / PIXELTHREADS_BACKWARD_A,
            (KERNEL_SIZE_SQ * heads + DIMTHREADS_BACKWARD_A - 1) / DIMTHREADS_BACKWARD_A,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS_BACKWARD_A - 1) / BATCHTHREADS_BACKWARD_A);
    const dim3 blocksv(
            (height * width + PIXELTHREADS_BACKWARD_V - 1) / PIXELTHREADS_BACKWARD_V,
            (DIM_DIV_DIMUNROLL_BACKWARD_V * heads + DIMTHREADS_BACKWARD_V - 1) / DIMTHREADS_BACKWARD_V,
            (max(int(batch_size / BATCHUNROLL), 1) + BATCHTHREADS_BACKWARD_V - 1) / BATCHTHREADS_BACKWARD_V);
    const dim3 threadsa(PIXELTHREADS_BACKWARD_A, DIMTHREADS_BACKWARD_A, BATCHTHREADS_BACKWARD_A);
    const dim3 threadsv(PIXELTHREADS_BACKWARD_V, DIMTHREADS_BACKWARD_V, BATCHTHREADS_BACKWARD_V);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_attn.scalar_type(), "nattenav_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattena_cuda_backward_kernel<7, 49, 3, scalar_t, accscalar_t><<<blocksa, threadsa, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 25)
            nattena_cuda_backward_kernel<5, 25, 2, scalar_t, accscalar_t><<<blocksa, threadsa, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 9)
            nattena_cuda_backward_kernel<3, 9, 1, scalar_t, accscalar_t><<<blocksa, threadsa, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 81)
            nattena_cuda_backward_kernel<9, 81, 4, scalar_t, accscalar_t><<<blocksa, threadsa, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads);
        else if (KERNEL_SIZE_SQ == 121)
            nattena_cuda_backward_kernel<11, 121, 5, scalar_t, accscalar_t><<<blocksa, threadsa, 0, stream>>>(d_out_a, d_attn_a, value_a, height, width,
                    batch_size, heads);
    }));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, d_attn.scalar_type(), "nattena_backward_cuda", ([&] {
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (KERNEL_SIZE_SQ == 49)
            nattenv_cuda_backward_kernel<7, 3, scalar_t><<<blocksv, threadsv, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    batch_size, heads, d_value.numel());
        else if (KERNEL_SIZE_SQ == 25)
            nattenv_cuda_backward_kernel<5, 2, scalar_t><<<blocksv, threadsv, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    batch_size, heads, d_value.numel());
        else if (KERNEL_SIZE_SQ == 9)
            nattenv_cuda_backward_kernel<3, 1, scalar_t><<<blocksv, threadsv, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    batch_size, heads, d_value.numel());
        else if (KERNEL_SIZE_SQ == 81)
            nattenv_cuda_backward_kernel<9, 4, scalar_t><<<blocksv, threadsv, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    batch_size, heads, d_value.numel());
        else if (KERNEL_SIZE_SQ == 121)
            nattenv_cuda_backward_kernel<11, 5, scalar_t><<<blocksv, threadsv, 0, stream>>>(d_out_a, d_value_a, attn_a, height, width,
                    batch_size, heads, d_value.numel());
    }));
    return {d_attn, d_value};
}

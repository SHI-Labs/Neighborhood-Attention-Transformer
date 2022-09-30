/*
NATTEN-COMMON FUNCTIONS (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#define AT_DISPATCH_HALF_TYPES(SCALARTYPE1, TYPE, NAME, ...)                         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        break;                                                              \
    }                                                                       \
  }()


#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}


inline __host__ __device__ int get_backward_window_start(const int index, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    return (index < KERNEL_SIZE * dilation) ? (index % dilation) : index - NEIGHBORHOOD_SIZE * dilation;
}


inline __host__ __device__ int get_backward_window_end(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    return (index >= length - KERNEL_SIZE * dilation) ? (length) : (index + (NEIGHBORHOOD_SIZE + 1) * dilation);
}


inline __host__ __device__ int get_window_start(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    if (dilation <= 1)
        return  max(index - NEIGHBORHOOD_SIZE, 0) + (index + NEIGHBORHOOD_SIZE >= length) * (length - index - NEIGHBORHOOD_SIZE - 1);
    int ni = index - NEIGHBORHOOD_SIZE * dilation;
    if (ni < 0)
        return index % dilation;
    if (index + NEIGHBORHOOD_SIZE * dilation >= length){
        const int imodd = index % dilation;
        const int a = int(length / dilation) * dilation;
        const int b = length - a;
        if (imodd < b)
            return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
        return a + imodd - KERNEL_SIZE * dilation;
    }
    return ni;
}


inline __host__ __device__ int get_pb_start(const int index, const int length, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE, const int dilation)
{
    if (dilation <= 1)
        return NEIGHBORHOOD_SIZE + (index < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - index) + (index + NEIGHBORHOOD_SIZE >= length) * (length - index - 1 - NEIGHBORHOOD_SIZE);
    if (index - NEIGHBORHOOD_SIZE * dilation < 0)
        return KERNEL_SIZE - 1 - (index / dilation);
    if (index + NEIGHBORHOOD_SIZE * dilation >= length)
        return (length - index - 1) / dilation;
    return NEIGHBORHOOD_SIZE;
}

#define CHECK_SEQUENCE(length, kernel_size, dilation) TORCH_CHECK(length >= kernel_size*dilation, "Input sequence length must be greater than or equal to kernel size x dilation.")
#define CHECK_FEATMAP(height, width, kernel_size, dilation) TORCH_CHECK(height >= kernel_size*dilation && width >= kernel_size*dilation, "Input resolution must be greater than or equal to kernel size x dilation.")
#define CHECK_3DFEATMAP(depth, height, width, kernel_size_d, kernel_size, dilation_d, dilation) TORCH_CHECK(depth >= kernel_size_d*dilation_d && height >= kernel_size*dilation && width >= kernel_size*dilation, "Input resolution must be greater than or equal to kernel size x dilation.")

// 2D Neighborhood Attention

// THE FOLLOWING CAN BE MODIFIED TO SUPPORT ADDITIONAL KERNEL SIZES
// MAKE SURE TO EDIT BOTH CHECK_KERNELSIZE AND LAUNCH_NA_KNS

#define CHECK_KERNELSIZE(NAME, kernel_size) TORCH_CHECK( \
        kernel_size == 3 || kernel_size == 5 || kernel_size == 7 || \
        kernel_size == 9 || kernel_size == 11 || kernel_size == 13, \
        NAME, " does not support kernel size ", kernel_size)

// First number is the kernel size itself, second is floor(kernel_size / 2) aka neighborhood radius.
#define LAUNCH_DNA_KNS(kernel_size, dilation, NAME, BLK, TPB, SMEM, CSTREAM, ...)                    \
({                                                                                                   \
    switch (kernel_size) {                                                                           \
        case 3:                                                                                      \
            _IN_LAUNCH_DNA_KNS(3, 1, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 5:                                                                                      \
            _IN_LAUNCH_DNA_KNS(5, 2, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 7:                                                                                      \
            _IN_LAUNCH_DNA_KNS(7, 3, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 9:                                                                                      \
            _IN_LAUNCH_DNA_KNS(9, 4, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 11:                                                                                     \
            _IN_LAUNCH_DNA_KNS(11, 5, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);         \
            break;                                                                                   \
        case 13:                                                                                     \
            _IN_LAUNCH_DNA_KNS(13, 6, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);         \
            break;                                                                                   \
        default:                                                                                     \
            TORCH_INTERNAL_ASSERT(false);                                                            \
            break;                                                                                   \
    }                                                                                                \
})

#define _IN_LAUNCH_DNA_KNS(KS, NS, dilation, NAME, BLK, TPB, SMEM, CSTREAM, ...)                     \
({                                                                                                   \
    switch (dilation) {                                                                              \
        case 1:                                                                                      \
            NAME<KS, NS, 1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 2:                                                                                      \
            NAME<KS, NS, 2, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 3:                                                                                      \
            NAME<KS, NS, 3, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 4:                                                                                      \
            NAME<KS, NS, 4, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 5:                                                                                      \
            NAME<KS, NS, 5, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 6:                                                                                      \
            NAME<KS, NS, 6, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 7:                                                                                      \
            NAME<KS, NS, 7, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 8:                                                                                      \
            NAME<KS, NS, 8, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 9:                                                                                      \
            NAME<KS, NS, 9, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 10:                                                                                     \
            NAME<KS, NS, 10, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 11:                                                                                     \
            NAME<KS, NS, 11, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 12:                                                                                     \
            NAME<KS, NS, 12, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 13:                                                                                     \
            NAME<KS, NS, 13, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 14:                                                                                     \
            NAME<KS, NS, 14, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 15:                                                                                     \
            NAME<KS, NS, 15, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 16:                                                                                     \
            NAME<KS, NS, 16, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        default:                                                                                     \
            NAME<KS, NS, -1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
    }                                                                                                \
})

#define LAUNCH_DNA_KNS_TILED79(TILE, KTILE, KS, NS, dilation, NAME, BLK, TPB, SMEM, CSTREAM, ...)    \
({                                                                                                   \
    switch (dilation) {                                                                              \
        case 1:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 2:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 2, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 3:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 3, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 4:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 4, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 5:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 5, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 6:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 6, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 7:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 7, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 8:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 8, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 9:                                                                                      \
            NAME<TILE, KTILE, KS, NS, 9, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                   \
        case 10:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 10, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 11:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 11, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 12:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 12, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 13:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 13, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 14:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 14, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 15:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 15, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        case 16:                                                                                     \
            NAME<TILE, KTILE, KS, NS, 16, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
        default:                                                                                     \
            NAME<TILE, KTILE, KS, NS, -1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                   \
    }                                                                                                \
})

#define LAUNCH_DNA_KNS_TILED1113(TX, TY, KX, KY, KS, NS, dilation, TMP, NAME, BLK, TPB, SMEM, CSTREAM, ...)  \
({                                                                                                           \
    switch (dilation) {                                                                                      \
        case 1:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 1, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 2:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 2, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 3:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 3, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 4:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 4, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 5:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 5, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 6:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 6, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 7:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 7, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 8:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 8, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 9:                                                                                              \
            NAME<TX, TY, KX, KY, KS, NS, 9, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                           \
        case 10:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 10, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 11:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 11, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 12:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 12, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 13:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 13, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 14:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 14, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 15:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 15, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        case 16:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, 16, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
        default:                                                                                             \
            NAME<TX, TY, KX, KY, KS, NS, -1, scalar_t, TMP><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                           \
    }                                                                                                        \
})

#define LAUNCH_DNA_DS(dilation, NAME, BLK, TPB, SMEM, CSTREAM, ...)                     \
({                                                                                                   \
    switch (dilation) {                                                                              \
        case 1:                                                                                      \
            NAME<1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 2:                                                                                      \
            NAME<2, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 3:                                                                                      \
            NAME<3, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 4:                                                                                      \
            NAME<4, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 5:                                                                                      \
            NAME<5, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 6:                                                                                      \
            NAME<6, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 7:                                                                                      \
            NAME<7, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 8:                                                                                      \
            NAME<8, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 9:                                                                                      \
            NAME<9, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                     \
            break;                                                                                   \
        case 10:                                                                                     \
            NAME<10, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 11:                                                                                     \
            NAME<11, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 12:                                                                                     \
            NAME<12, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 13:                                                                                     \
            NAME<13, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 14:                                                                                     \
            NAME<14, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 15:                                                                                     \
            NAME<15, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        case 16:                                                                                     \
            NAME<16, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
        default:                                                                                     \
            NAME<-1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);                    \
            break;                                                                                   \
    }                                                                                                \
})


// 1D KERNEL LAUNCHER
// First number is the kernel size itself, second is floor(kernel_size / 2) aka neighborhood radius.
#define LAUNCH_DNA_KNS_1D(kernel_size, dilation, NAME, BLK, TPB, SMEM, CSTREAM, ...)                 \
({                                                                                                   \
    switch (kernel_size) {                                                                           \
        case 3:                                                                                      \
            _IN_LAUNCH_DNA_KNS(3, 1, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 5:                                                                                      \
            _IN_LAUNCH_DNA_KNS(5, 2, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 7:                                                                                      \
            _IN_LAUNCH_DNA_KNS(7, 3, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 9:                                                                                      \
            _IN_LAUNCH_DNA_KNS(9, 4, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);          \
            break;                                                                                   \
        case 11:                                                                                     \
            _IN_LAUNCH_DNA_KNS(11, 5, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);         \
            break;                                                                                   \
        case 13:                                                                                     \
            _IN_LAUNCH_DNA_KNS(13, 6, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);         \
            break;                                                                                   \
        default:                                                                                     \
            _IN_LAUNCH_DNA_KNS(-1, -1, dilation, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);        \
            break;                                                                                   \
    }                                                                                                \
})

// 3D KERNEL LAUNCHER
#define LAUNCH_NA_KDNDS_INN(kernel_size, KERNEL_SIZE_DPTH, NEIGH_SIZE_DPTH, NAME, BLK, TPB, SMEM, CSTREAM, ...)       \
({                                                                                                                    \
    switch (kernel_size) {                                                                                            \
        case 3:                                                                                                       \
            NAME<3, KERNEL_SIZE_DPTH, 1, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                                    \
        case 5:                                                                                                       \
            NAME<5, KERNEL_SIZE_DPTH, 2, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                                    \
        case 7:                                                                                                       \
            NAME<7, KERNEL_SIZE_DPTH, 3, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                                    \
        case 9:                                                                                                       \
            NAME<9, KERNEL_SIZE_DPTH, 4, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);        \
            break;                                                                                                    \
        case 11:                                                                                                      \
            NAME<11, KERNEL_SIZE_DPTH, 5, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                                    \
        case 13:                                                                                                      \
            NAME<13, KERNEL_SIZE_DPTH, 6, NEIGH_SIZE_DPTH, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);       \
            break;                                                                                                    \
        default:                                                                                                      \
            TORCH_INTERNAL_ASSERT(false);                                                                             \
            break;                                                                                                    \
    }                                                                                                                 \
})

#define LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, NAME, BLK, TPB, SMEM, CSTREAM, ...)              \
({                                                                                                   \
    switch (kernel_size_d) {                                                                         \
        case 3:                                                                                      \
            LAUNCH_NA_KDNDS_INN(kernel_size, 3, 1, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);      \
            break;                                                                                   \
        case 5:                                                                                      \
            LAUNCH_NA_KDNDS_INN(kernel_size, 5, 2, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);      \
            break;                                                                                   \
        case 7:                                                                                      \
            LAUNCH_NA_KDNDS_INN(kernel_size, 7, 3, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);      \
            break;                                                                                   \
        case 9:                                                                                      \
            LAUNCH_NA_KDNDS_INN(kernel_size, 9, 4, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);      \
            break;                                                                                   \
        case 11:                                                                                     \
            LAUNCH_NA_KDNDS_INN(kernel_size, 11, 5, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);     \
            break;                                                                                   \
        case 13:                                                                                     \
            LAUNCH_NA_KDNDS_INN(kernel_size, 13, 6, NAME, BLK, TPB, SMEM, CSTREAM, __VA_ARGS__);     \
            break;                                                                                   \
        default:                                                                                     \
            TORCH_INTERNAL_ASSERT(false);                                                            \
            break;                                                                                   \
    }                                                                                                \
})

"""
Neighborhood Attention Torch Extension (CUDA only) Setup

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='natcuda',
    version='0.11',
    author='Ali Hassani',
    author_email='alih@uoregon.edu',
    description='Neighborhood Attention CUDA Kernel',
    ext_modules=[
        CUDAExtension('nattenav_cuda', [
            'nattenav_cuda.cpp',
            'nattenav_cuda_kernel.cu',
        ]),
        CUDAExtension('nattenqkrpb_cuda', [
            'nattenqkrpb_cuda.cpp',
            'nattenqkrpb_cuda_kernel.cu',
        ]),
        CUDAExtension('natten1dav_cuda', [
            'natten1dav_cuda.cpp',
            'natten1dav_cuda_kernel.cu',
        ]),
        CUDAExtension('natten1dqkrpb_cuda', [
            'natten1dqkrpb_cuda.cpp',
            'natten1dqkrpb_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='natcuda',
    version='0.1',
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
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

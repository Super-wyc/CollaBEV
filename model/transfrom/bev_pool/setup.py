import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

def make_cuda_ext(name, sources, sources_cuda=[]):
    define_macros = []
    extra_link_args = ['-L/usr/lib/x86_64-linux-gnu']
    extra_compile_args = {'cxx': ['-g', '-O3', '-fopenmp', '-pthread']}

    if os.path.exists('/usr/local/cuda'):
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-O3'
        ]
        sources += sources_cuda
    else:
        print('CUDA is not found, CPU version will be built.')
        extension = CppExtension
        
    return extension(
        name=name,
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

setup(
    name='bev_pool',
    ext_modules=[
        make_cuda_ext(
            name='bev_pool_ext',
            sources=[
                'src/bev_pool.cpp',
            ],
            sources_cuda=[
                'src/bev_pool_cuda.cu',
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

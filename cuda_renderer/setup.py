from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA home
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

setup(
    name='nlos_gaussian_renderer',
    ext_modules=[
        CUDAExtension(
            name='nlos_gaussian_renderer._C',
            sources=[
                'src/bindings.cpp',
                'src/ray_aabb.cu',
                'src/volume_renderer.cu',
                'src/volume_renderer_analytic.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--extended-lambda',
                    '--expt-relaxed-constexpr',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                ]
            },
            include_dirs=[
                os.path.join(os.path.dirname(__file__), 'include'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch'],
)


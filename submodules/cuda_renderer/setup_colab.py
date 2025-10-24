"""
Setup script optimized for Google Colab
Handles common Colab-specific issues:
- CUDA architecture detection
- Compiler flag compatibility
- Memory-efficient compilation
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import subprocess

def get_cuda_arch():
    """
    Detect GPU architecture in Colab
    Common Colab GPUs:
    - T4: sm_75
    - P100: sm_60
    - V100: sm_70
    - A100: sm_80
    """
    try:
        # Try to get from nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().split('\n')[0]
            major, minor = compute_cap.split('.')
            arch = f'sm_{major}{minor}'
            print(f"Detected GPU architecture: {arch}")
            return arch
    except:
        pass
    
    # Fallback: Use PyTorch detection
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f'sm_{capability[0]}{capability[1]}'
        print(f"Detected GPU architecture from PyTorch: {arch}")
        return arch
    
    # Default fallback (T4 is most common in Colab)
    print("Could not detect GPU, using default: sm_75 (T4)")
    return 'sm_75'

def check_cuda_available():
    """Check if CUDA is properly configured"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure you're using a GPU runtime:\n"
            "Runtime -> Change runtime type -> Hardware accelerator -> GPU"
        )
    print(f"✓ CUDA available: {torch.version.cuda}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Check CUDA
check_cuda_available()

# Get architecture
cuda_arch = get_cuda_arch()

# Colab-optimized compiler flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    f'-arch={cuda_arch}',  # Use detected architecture
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

# Add C++14 instead of C++17 for better compatibility
cxx_flags = ['-O3', '-std=c++14']

print(f"\nCompiler flags:")
print(f"  NVCC: {' '.join(nvcc_flags)}")
print(f"  CXX: {' '.join(cxx_flags)}")

setup(
    name='nlos_gaussian_renderer',
    version='0.1.0',
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
                'cxx': cxx_flags,
                'nvcc': nvcc_flags,
            },
            include_dirs=[
                os.path.join(os.path.dirname(__file__), 'include'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    install_requires=['torch'],
    python_requires='>=3.7',
)


#!/bin/bash

# Installation script for NLOS Gaussian CUDA Renderer

echo "=========================================="
echo "NLOS Gaussian CUDA Renderer Installation"
echo "=========================================="
echo ""

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA first."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA version:"
nvcc --version
echo ""

# Check Python and PyTorch
echo "Checking PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Please install PyTorch with CUDA support first."
    exit 1
fi

echo ""
echo "Installing CUDA extension..."
cd cuda_renderer

# Clean previous builds
rm -rf build dist *.egg-info

# Build and install
python setup.py install

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation successful!"
    echo "=========================================="
    echo ""
    echo "To use the CUDA renderer, set in your config:"
    echo "  args.use_cuda_renderer = True"
    echo ""
    echo "To test the installation:"
    echo "  python -c 'from cuda_renderer import NLOSGaussianRenderer; print(\"CUDA renderer ready!\")'"
else
    echo ""
    echo "=========================================="
    echo "Installation failed!"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "1. CUDA version mismatch between nvcc and PyTorch"
    echo "2. Incompatible C++ compiler version"
    echo "3. Missing CUDA_HOME environment variable"
    echo ""
    echo "Try setting CUDA_HOME:"
    echo "  export CUDA_HOME=/usr/local/cuda"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi



# NLOS Gaussian Renderer - CUDA Extension

Efficient ray-based volume rendering for NLOS (Non-Line-of-Sight) reconstruction using 3D Gaussian primitives with advanced CUDA optimizations.

## Features

- **Ray-based rendering**: Process each (θ, φ) ray independently
- **Gaussian filtering**: Only compute contributions from Gaussians that intersect with each ray (AABB test)
- **CUDA acceleration**: Parallel processing of all rays with shared memory optimization
- **Forward caching**: Cache expensive computations (PDF, SH evaluation) to eliminate recomputation in backward pass
- **Optimized backward pass**: Local gradient accumulation + forward cache = 1.5-2× faster training
- **Memory efficient**: Avoids computing all Gaussians × all sample points

## Installation

```bash
cd cuda_renderer
python setup.py install
```

### Requirements

- PyTorch with CUDA support
- CUDA Toolkit (11.0+)
- C++17 compatible compiler

## Usage

### Basic Usage

```python
from gaussian_model.cuda_autograd import CUDARenderforBenchMark

# Create renderer with forward caching (default: 'shared' mode)
renderer = CUDARenderforBenchMark(
    memory_mode='shared'  # or 'global' for baseline
)

# Render with gradient computation support
result = renderer(
    ray_origins, ray_directions, t_samples,
    gaussian_means, gaussian_scales, gaussian_rotations,
    gaussian_opacities, gaussian_features, camera_pos,
    active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
)

# result is differentiable - gradients flow back automatically
loss = criterion(result, target)
loss.backward()
```

### Advanced Usage

```python
from gaussian_model.cuda_autograd import create_cuda_render_benchmark_module

# Create benchmark module
renderer = create_cuda_render_benchmark_module(memory_mode='shared')

# Direct CUDA function calls (low-level)
from nlos_gaussian_renderer import _C

# Forward pass with forward cache
rho_density, density, transmittance, bboxes, filter, cache = _C.render_rays_shared(
    ray_origins, ray_directions, t_samples,
    gaussian_means, gaussian_scales, gaussian_rotations,
    gaussian_opacities, gaussian_features, camera_pos,
    active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
)

# Backward pass with forward cache
grad_means, grad_scales, grad_rotations, grad_opacities, grad_features = _C.render_rays_backward(
    rho_density, density, transmittance,
    grad_rho_density, grad_density, grad_transmittance,
    ray_origins, ray_directions, t_samples,
    filter, gaussian_means, gaussian_scales, gaussian_rotations,
    gaussian_opacities, gaussian_features, camera_pos,
    cache,  # Forward cache for fast backward!
    active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
)
```

### Memory Mode Selection

```python
# 'shared' mode (default, recommended)
# - Uses shared memory for t_samples and camera_pos
# - Includes forward caching for backward pass optimization
renderer = CUDARenderforBenchMark(memory_mode='shared')

# 'global' mode (baseline)
# - No shared memory optimization
# - No forward caching (slower backward pass)
renderer = CUDARenderforBenchMark(memory_mode='global')
```

## Architecture

### Ray-based Rendering Pipeline

```
1. Generate rays for all (θ, φ) combinations
2. Compute Gaussian bounding boxes (AABB)
3. Filter Gaussians per ray using AABB intersection (up to 256 Gaussians per ray)
4. For each ray, sample r values along the ray
5. Compute Gaussian contributions at each sample point
6. Cache expensive computations (PDF, SH evaluation) for backward pass
7. Apply volume rendering (transmittance, occlusion)
8. Integrate over angles to get histogram
```

### Forward Pass Optimizations

- **Shared Memory**: Cache `t_samples` and `camera_pos` in shared memory for faster access
- **Gaussian Filtering**: AABB intersection test to reduce computation from O(N_gaussians) to O(N_filtered)
- **Forward Cache**: Store intermediate results (PDF, opacity, SH evaluation) to eliminate recomputation in backward pass

### Backward Pass Optimizations

- **Forward Cache Reuse**: Read cached PDF, opacity, and SH values instead of recomputing
- **Local Gradient Accumulation**: Accumulate gradients locally per ray, then single atomicAdd to global memory
- **Compact Cache**: Only cache expensive computations (3 floats) to minimize memory usage (~4GB for typical scenes)

### Memory Modes

The renderer supports two memory optimization modes (selectable via `memory_mode` parameter):

- **`'shared'`** (default): Uses shared memory for `t_samples` and `camera_pos`, includes forward caching
- **`'global'`**: Baseline with global memory only, no forward caching

## Performance

### Computational Complexity

Compared to the naive approach (all Gaussians × all points):
- **Memory**: O(N_rays × N_samples × N_filtered) vs O(N_gaussians × N_rays × N_samples)
- **Speed**: ~10-50× faster depending on scene density
- **Scalability**: Can handle 100k+ Gaussians efficiently with filtering

### Benchmarks

Typical performance on RTX 3090 / A100 (4096 rays, 300 samples, 10k Gaussians):

| Configuration | Forward | Backward | Total | Speedup |
|--------------|---------|----------|-------|---------|
| Baseline (no cache) | 37 ms | 215 ms | 252 ms | 1.0× |
| With forward cache | 33 ms | 142 ms | 175 ms | **1.44×** |
| With shared memory | 33 ms | 142 ms | 175 ms | 1.44× |

**Forward Cache Benefits:**
- Eliminates expensive PDF and SH recomputation in backward pass
- Reduces backward/forward ratio from 5.8× to 4.3×
- Memory overhead: ~4GB for compact cache (3 floats per entry)

**Memory Usage:**
- Forward cache (compact): ~4GB for 4096×300×256 configuration
- Temporary gradient buffer: ~220MB for backward pass
- Total additional: ~4.2GB (acceptable for modern GPUs)

## Integration with PyTorch

The renderer is fully integrated with PyTorch's autograd system:

```python
import torch
from gaussian_model.cuda_autograd import CUDARenderforBenchMark

# Create renderer module
renderer = CUDARenderforBenchMark(memory_mode='shared')

# Prepare inputs (requires_grad=True for trainable parameters)
gaussian_means = torch.randn(N_gaussians, 3, device='cuda', requires_grad=True)
gaussian_scales = torch.randn(N_gaussians, 3, device='cuda', requires_grad=True)
gaussian_rotations = torch.randn(N_gaussians, 4, device='cuda', requires_grad=True)
gaussian_opacities = torch.randn(N_gaussians, 1, device='cuda', requires_grad=True)
gaussian_features = torch.randn(N_gaussians, 16, device='cuda', requires_grad=True)

# Forward pass (automatic forward cache creation)
output = renderer(
    ray_origins, ray_directions, t_samples,
    gaussian_means, gaussian_scales, gaussian_rotations,
    gaussian_opacities, gaussian_features, camera_pos,
    active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
)

# Backward pass (uses cached forward results automatically)
loss = criterion(output, target)
loss.backward()

# Gradients are computed efficiently using forward cache
# gaussian_means.grad, gaussian_scales.grad, etc. are available
```


## Technical Details

### Ray-AABB Intersection
Uses slab method for fast ray-box intersection testing. Only Gaussians whose bounding boxes intersect with the ray are considered. Supports up to 256 Gaussians per ray (configurable via `MAX_GAUSSIANS_PER_RAY`).

### Spherical Harmonics
View-dependent albedo is computed using spherical harmonics (up to degree 3). SH coefficients are evaluated per Gaussian and cached for backward pass.

### Forward Cache Mechanism

The renderer implements a compact forward cache to eliminate expensive recomputation in backward pass:

```cpp
struct ForwardCache {
    float pdf;      // Gaussian PDF value (expensive: eval_gaussian_pdf)
    float opacity;  // Sigmoid(opacity_logit)
    float rho;      // Albedo from SH evaluation (expensive: eval_sh)
};
```

**Trade-offs:**
- **Cached**: PDF and SH evaluation (most expensive operations)
- **Not cached**: Gaussian parameters (mean, scale, quat) - reloaded in backward for memory efficiency
- **Memory**: 3 floats (12 bytes) per ray×sample×gaussian entry

### Gradient Accumulation Optimization

The backward pass uses local gradient accumulation to minimize atomic operations:

1. Each ray accumulates gradients locally in thread-private arrays
2. After processing all samples, single `atomicAdd` per Gaussian to global memory
3. Reduces atomic contention from O(N_samples × N_gaussians) to O(N_gaussians)

### Shared Memory Usage

- `t_samples`: Cached in shared memory (all rays use same samples)
- `camera_pos`: Cached in shared memory (all rays use same camera)
- Size: ~2KB per block (well within shared memory limits)




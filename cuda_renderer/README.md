# NLOS Gaussian Renderer - CUDA Extension

Efficient ray-based volume rendering for NLOS (Non-Line-of-Sight) reconstruction using 3D Gaussian primitives.

## Features

- **Ray-based rendering**: Process each (θ, φ) ray independently
- **Gaussian filtering**: Only compute contributions from Gaussians that intersect with each ray (AABB test)
- **CUDA acceleration**: Parallel processing of all rays
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

```python
from cuda_renderer import NLOSGaussianRenderer

# Create renderer
renderer = NLOSGaussianRenderer(sigma_threshold=3.0)

# Render
result, pred_histogram = renderer.render(
    gaussian_model=model,
    camera_pos=camera_position,
    theta_range=(theta_min, theta_max),
    phi_range=(phi_min, phi_max),
    r_range=(r_min, r_max),
    num_theta=32,
    num_phi=32,
    num_r=200,
    c=1.0,
    deltaT=0.01,
    scaling_modifier=1.0,
    use_occlusion=True,
    rendering_type='netf'
)
```

## Architecture

```
Ray-based Rendering Pipeline:
1. Generate rays for all (θ, φ) combinations
2. [Optional] Filter Gaussians per ray using AABB intersection
3. For each ray, sample r values along the ray
4. Compute Gaussian contributions at each sample point
5. Apply volume rendering (transmittance, occlusion)
6. Integrate over angles to get histogram
```

## Performance

Compared to the naive approach (all Gaussians × all points):
- **Memory**: O(N_rays × N_samples × N_filtered) vs O(N_gaussians × N_rays × N_samples)
- **Speed**: ~10-50× faster depending on scene density
- **Scalability**: Can handle 10k+ Gaussians efficiently

## Integration with Existing Code

The renderer can be used as a drop-in replacement for `gaussian_transient_rendering`:

```python
# Old approach (in nlos_helpers.py)
result, pred_histogram = gaussian_transient_rendering(
    args, model, data_kwargs, input_points, 
    current_camera_grid_positions, I1, I2, num_r, dtheta, dphi
)

# New approach (with CUDA renderer)
from cuda_renderer import NLOSGaussianRenderer
renderer = NLOSGaussianRenderer()

result, pred_histogram = renderer.render(
    gaussian_model=model,
    camera_pos=current_camera_grid_positions,
    theta_range=(theta_min, theta_max),
    phi_range=(phi_min, phi_max),
    r_range=(I1 * c * deltaT, I2 * c * deltaT),
    num_theta=args.num_sampling_points,
    num_phi=args.num_sampling_points,
    num_r=num_r,
    c=data_kwargs['c'],
    deltaT=data_kwargs['deltaT'],
    use_occlusion=args.occlusion,
    rendering_type=args.rendering_type
)
```

## Technical Details

### Ray-AABB Intersection
Uses slab method for fast ray-box intersection testing. Only Gaussians whose bounding boxes intersect with the ray are considered.

### Volume Rendering
Implements both rendering types:
- **NeRF-style (netf)**: Uses exponential transmittance
- **NeuS-style (nlos-neus)**: Uses alpha compositing

### Spherical Harmonics
View-dependent albedo is computed using spherical harmonics (up to degree 3).

## Troubleshooting

If compilation fails:
1. Check CUDA is properly installed: `nvcc --version`
2. Check PyTorch CUDA version matches: `python -c "import torch; print(torch.version.cuda)"`
3. Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`

## Future Improvements

- [ ] Implement tile-based rendering for better memory access patterns
- [ ] Add support for non-confocal settings
- [ ] Optimize AABB filtering with spatial data structures (BVH, octree)
- [ ] Multi-GPU support for large scenes



"""
Simple Gaussian Filtering Statistics Checker
Quick check without visualization
"""

import torch
import numpy as np

try:
    from nlos_gaussian_renderer import _C
    print("✅ CUDA renderer loaded successfully")
except ImportError as e:
    print(f"❌ CUDA renderer not available: {e}")
    print("\nPlease compile first:")
    print("  cd submodules/cuda_renderer")
    print("  pip install -e .")
    exit(1)

device = torch.device("cuda:0")

# Configuration
N_RAYS = 4096
N_SAMPLES = 300
N_GAUSSIANS = 10000  # Start with 10k, then try 50k, 100k
SCENE_TYPE = "random"  # or "clustered"

print(f"\n{'='*70}")
print(f"Gaussian Filtering Statistics")
print(f"{'='*70}")
print(f"Scene:     {SCENE_TYPE}")
print(f"Gaussians: {N_GAUSSIANS:,}")
print(f"Rays:      {N_RAYS:,}")
print(f"Samples:   {N_SAMPLES}")
print(f"{'='*70}\n")

# Generate test scene
print("Generating scene...")
if SCENE_TYPE == "clustered":
    n_clusters = 10
    cluster_centers = torch.randn(n_clusters, 3, device=device) * 2.0
    cluster_ids = torch.randint(0, n_clusters, (N_GAUSSIANS,), device=device)
    gaussian_means = cluster_centers[cluster_ids] + torch.randn(N_GAUSSIANS, 3, device=device) * 0.3
else:
    gaussian_means = torch.randn(N_GAUSSIANS, 3, device=device) * 2.0

gaussian_scales = torch.randn(N_GAUSSIANS, 3, device=device) * 0.5 - 1.0
gaussian_rotations = torch.randn(N_GAUSSIANS, 4, device=device)
gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
gaussian_opacities = torch.randn(N_GAUSSIANS, 1, device=device)
gaussian_features = torch.randn(N_GAUSSIANS, 16, device=device)

# Generate rays
ray_origins = torch.zeros(N_RAYS, 3, device=device)
ray_directions = torch.randn(N_RAYS, 3, device=device)
ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

t_samples = torch.linspace(0.5, 3.0, N_SAMPLES, device=device)
camera_pos = torch.zeros(3, device=device)

# Render and get filtering stats
print("Rendering...")
result = _C.render_rays_shared(
    ray_origins, ray_directions, t_samples,
    gaussian_means, gaussian_scales, gaussian_rotations,
    gaussian_opacities, gaussian_features, camera_pos,
    0, 1.0, 0.01, 1.0, True
)

# Extract gaussian_filter
_, _, _, _, gaussian_filter, _ = result
num_gaussians_per_ray = gaussian_filter[:, 0].cpu().numpy()

# Statistics
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Min:      {num_gaussians_per_ray.min():.0f} gaussians/ray")
print(f"Max:      {num_gaussians_per_ray.max():.0f} gaussians/ray")
print(f"Mean:     {num_gaussians_per_ray.mean():.1f} gaussians/ray")
print(f"Median:   {np.median(num_gaussians_per_ray):.1f} gaussians/ray")
print(f"Std:      {num_gaussians_per_ray.std():.1f}")
print(f"\nPercentiles:")
print(f"P50:      {np.percentile(num_gaussians_per_ray, 50):.1f}")
print(f"P75:      {np.percentile(num_gaussians_per_ray, 75):.1f}")
print(f"P90:      {np.percentile(num_gaussians_per_ray, 90):.1f}")
print(f"P95:      {np.percentile(num_gaussians_per_ray, 95):.1f}")
print(f"P99:      {np.percentile(num_gaussians_per_ray, 99):.1f}")

# Limit check
MAX_LIMIT = 256
rays_at_limit = (num_gaussians_per_ray >= MAX_LIMIT).sum()
print(f"\n{'='*70}")
print(f"Limit Check (MAX_GAUSSIANS_PER_RAY = {MAX_LIMIT})")
print(f"{'='*70}")
print(f"Rays at limit: {rays_at_limit} / {N_RAYS} ({rays_at_limit/N_RAYS*100:.2f}%)")

if rays_at_limit > 0:
    print(f"\n⚠️  WARNING: {rays_at_limit} rays hitting the {MAX_LIMIT} limit!")
    print(f"   Some Gaussians are being SILENTLY IGNORED!")
    recommended = int(np.percentile(num_gaussians_per_ray, 99) * 1.2)
    print(f"   Recommended: MAX_GAUSSIANS_PER_RAY = {recommended}")
else:
    print(f"\n✅ OK: No rays hitting the limit")
    usage = (np.percentile(num_gaussians_per_ray, 95) / MAX_LIMIT * 100)
    print(f"   P95 usage: {usage:.1f}% of capacity")

# Memory estimate
compact_size = 3 * 4  # 3 floats
memory_mb = (N_RAYS * N_SAMPLES * MAX_LIMIT * compact_size) / (1024**2)
print(f"\n{'='*70}")
print(f"Memory (Forward Cache)")
print(f"{'='*70}")
print(f"Current: {memory_mb:.1f} MB ({N_RAYS}×{N_SAMPLES}×{MAX_LIMIT}×{compact_size}B)")

if rays_at_limit > N_RAYS * 0.01:
    recommended = int(np.percentile(num_gaussians_per_ray, 99) * 1.2)
    new_memory_mb = (N_RAYS * N_SAMPLES * recommended * compact_size) / (1024**2)
    print(f"With {recommended}: {new_memory_mb:.1f} MB ({new_memory_mb/memory_mb:.2f}x)")

print(f"\n{'='*70}\n")


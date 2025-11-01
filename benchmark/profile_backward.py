"""
Backward Pass Profiling
Identify why backward is 6x slower than forward
"""

import torch
import time
import numpy as np

try:
    from gaussian_model.cuda_autograd import create_cuda_render_benchmark_module
    print("✓ CUDA renderer imported")
except ImportError as e:
    print(f"✗ Failed: {e}")
    exit(1)

device = torch.device("cuda:0")

# Practical setup
n_rays = 4096
n_samples = 300
n_gaussians = 10000

print(f"\n{'='*70}")
print(f"Profiling: {n_rays} rays × {n_samples} samples × {n_gaussians} gaussians")
print(f"{'='*70}\n")

# Generate data
ray_origins = torch.randn(n_rays, 3, device=device)
ray_directions = torch.randn(n_rays, 3, device=device)
ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

t_samples = torch.linspace(0.5, 3.0, n_samples, device=device)

gaussian_means = torch.randn(n_gaussians, 3, device=device, requires_grad=True)
gaussian_scales = torch.randn(n_gaussians, 3, device=device, requires_grad=True)
gaussian_rotations = torch.randn(n_gaussians, 4, device=device, requires_grad=True)
gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
gaussian_opacities = torch.randn(n_gaussians, 1, device=device, requires_grad=True)
gaussian_features = torch.randn(n_gaussians, 16, device=device, requires_grad=True)

camera_pos = torch.zeros(3, device=device)

renderer = create_cuda_render_benchmark_module()

# Warmup
for _ in range(3):
    output = renderer(
        ray_origins, ray_directions, t_samples,
        gaussian_means, gaussian_scales, gaussian_rotations,
        gaussian_opacities, gaussian_features, camera_pos,
        0, 1.0, 0.01, 1.0, True
    )
    loss = output.sum()
    loss.backward()
    
    # Zero grads
    gaussian_means.grad.zero_()
    gaussian_scales.grad.zero_()
    gaussian_rotations.grad.zero_()
    gaussian_opacities.grad.zero_()
    gaussian_features.grad.zero_()

torch.cuda.synchronize()

# Benchmark
n_iter = 20

forward_times = []
backward_times = []
total_times = []

for i in range(n_iter):
    # Forward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    output = renderer(
        ray_origins, ray_directions, t_samples,
        gaussian_means, gaussian_scales, gaussian_rotations,
        gaussian_opacities, gaussian_features, camera_pos,
        0, 1.0, 0.01, 1.0, True
    )
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    forward_time = (t1 - t0) * 1000
    
    # Backward
    loss = output.sum()
    
    gaussian_means.grad.zero_()
    gaussian_scales.grad.zero_()
    gaussian_rotations.grad.zero_()
    gaussian_opacities.grad.zero_()
    gaussian_features.grad.zero_()
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    loss.backward()
    
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    backward_time = (t3 - t2) * 1000
    
    total_time = forward_time + backward_time
    
    forward_times.append(forward_time)
    backward_times.append(backward_time)
    total_times.append(total_time)
    
    if (i + 1) % 5 == 0:
        print(f"Iter {i+1}: Forward {forward_time:.2f}ms, Backward {backward_time:.2f}ms, Ratio {backward_time/forward_time:.2f}x")

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Forward:  {np.mean(forward_times):.2f} ± {np.std(forward_times):.2f} ms")
print(f"Backward: {np.mean(backward_times):.2f} ± {np.std(backward_times):.2f} ms")
print(f"Total:    {np.mean(total_times):.2f} ± {np.std(total_times):.2f} ms")
print(f"Ratio:    {np.mean(backward_times)/np.mean(forward_times):.2f}x")
print(f"{'='*70}\n")

# Estimate breakdown
print("ESTIMATED BREAKDOWN:")
print(f"  Recomputation overhead: ~{np.mean(forward_times):.2f} ms ({np.mean(forward_times)/np.mean(backward_times)*100:.1f}%)")
print(f"  Gradient computation:   ~{np.mean(backward_times) - np.mean(forward_times):.2f} ms ({(np.mean(backward_times) - np.mean(forward_times))/np.mean(backward_times)*100:.1f}%)")
print(f"  Memory/atomicAdd:       ~{max(0, np.mean(backward_times) - 2*np.mean(forward_times)):.2f} ms\n")

print("RECOMMENDATIONS:")
if np.mean(backward_times) / np.mean(forward_times) > 4:
    print("  🔴 Backward/Forward ratio > 4x (abnormal!)")
    print("  → Priority 1: Cache forward results")
    print("  → Expected improvement: 2-3x faster backward")
elif np.mean(backward_times) / np.mean(forward_times) > 3:
    print("  🟡 Backward/Forward ratio > 3x (suboptimal)")
    print("  → Consider caching forward results")
    print("  → Expected improvement: 1.5-2x faster backward")
else:
    print("  🟢 Backward/Forward ratio < 3x (normal)")
    print("  → Focus on other optimizations")

print("\nNEXT STEPS:")
print("  1. Implement forward caching (biggest impact)")
print("  2. Profile with: nsys profile python profile_backward.py")
print("  3. Optimize gradient functions")


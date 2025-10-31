"""
CUDA Performance Benchmark for Shared Memory Optimization
Tests the performance improvements from shared memory usage
"""

import torch
import time
import numpy as np

# Import your CUDA renderer
try:
    from submodules.cuda_renderer import render_rays
    print("✓ CUDA renderer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CUDA renderer: {e}")
    print("Please compile CUDA code first: cd submodules && bash cuda_build.sh")
    exit(1)


def benchmark_rendering(n_rays, n_samples, n_gaussians, n_warmup=5, n_iterations=20):
    """
    Benchmark CUDA rendering performance
    
    Args:
        n_rays: Number of rays
        n_samples: Number of samples per ray
        n_gaussians: Number of Gaussians
        n_warmup: Number of warmup iterations
        n_iterations: Number of benchmark iterations
    """
    device = torch.device("cuda:0")
    
    print(f"\n{'='*60}")
    print(f"Benchmark Configuration:")
    print(f"  Rays: {n_rays}")
    print(f"  Samples per ray: {n_samples}")
    print(f"  Gaussians: {n_gaussians}")
    print(f"  Warmup iterations: {n_warmup}")
    print(f"  Benchmark iterations: {n_iterations}")
    print(f"{'='*60}\n")
    
    # Generate random test data
    print("Generating test data...")
    ray_origins = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
    ray_directions = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    
    t_samples = torch.linspace(0.5, 3.0, n_samples, device=device, dtype=torch.float32)
    
    gaussian_means = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32)
    gaussian_scales = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32) * 0.5
    gaussian_rotations = torch.randn(n_gaussians, 4, device=device, dtype=torch.float32)
    gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
    gaussian_opacities = torch.randn(n_gaussians, 1, device=device, dtype=torch.float32)
    
    # SH features (degree 0 = 1 coefficient)
    gaussian_features = torch.randn(n_gaussians, 16, device=device, dtype=torch.float32)
    
    camera_pos = torch.zeros(3, device=device, dtype=torch.float32)
    
    # Parameters
    active_sh_degree = 0
    c = 1.0
    deltaT = 0.01
    scaling_modifier = 1.0
    use_occlusion = True
    
    print("Data generated. Starting benchmark...\n")
    
    # Warmup
    print(f"Warmup ({n_warmup} iterations)...")
    for i in range(n_warmup):
        with torch.no_grad():
            _ = render_rays(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
    torch.cuda.synchronize()
    print("✓ Warmup complete\n")
    
    # Benchmark forward pass
    print(f"Benchmarking forward pass ({n_iterations} iterations)...")
    forward_times = []
    
    for i in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            output = render_rays(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = (end - start) * 1000  # Convert to ms
        forward_times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{n_iterations}: {elapsed:.3f} ms")
    
    # Statistics
    forward_times = np.array(forward_times)
    print(f"\n{'='*60}")
    print("Forward Pass Results:")
    print(f"  Mean: {forward_times.mean():.3f} ms")
    print(f"  Std:  {forward_times.std():.3f} ms")
    print(f"  Min:  {forward_times.min():.3f} ms")
    print(f"  Max:  {forward_times.max():.3f} ms")
    print(f"  Median: {np.median(forward_times):.3f} ms")
    print(f"{'='*60}\n")
    
    # Benchmark backward pass
    print(f"Benchmarking backward pass ({n_iterations} iterations)...")
    
    # Enable gradients
    gaussian_means.requires_grad_(True)
    gaussian_scales.requires_grad_(True)
    gaussian_rotations.requires_grad_(True)
    gaussian_opacities.requires_grad_(True)
    gaussian_features.requires_grad_(True)
    
    backward_times = []
    
    for i in range(n_iterations):
        # Forward pass
        output = render_rays(
            ray_origins, ray_directions, t_samples,
            gaussian_means, gaussian_scales, gaussian_rotations,
            gaussian_opacities, gaussian_features, camera_pos,
            active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
        )
        
        rho_density = output[0]
        loss = rho_density.sum()
        
        # Zero gradients
        if gaussian_means.grad is not None:
            gaussian_means.grad.zero_()
            gaussian_scales.grad.zero_()
            gaussian_rotations.grad.zero_()
            gaussian_opacities.grad.zero_()
            gaussian_features.grad.zero_()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Backward pass
        loss.backward()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = (end - start) * 1000  # Convert to ms
        backward_times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{n_iterations}: {elapsed:.3f} ms")
    
    # Statistics
    backward_times = np.array(backward_times)
    print(f"\n{'='*60}")
    print("Backward Pass Results:")
    print(f"  Mean: {backward_times.mean():.3f} ms")
    print(f"  Std:  {backward_times.std():.3f} ms")
    print(f"  Min:  {backward_times.min():.3f} ms")
    print(f"  Max:  {backward_times.max():.3f} ms")
    print(f"  Median: {np.median(backward_times):.3f} ms")
    print(f"{'='*60}\n")
    
    # Total time
    total_mean = forward_times.mean() + backward_times.mean()
    print(f"Total (Forward + Backward): {total_mean:.3f} ms")
    print(f"Throughput: {1000.0 / total_mean:.2f} iterations/second")
    print(f"\n")
    
    return {
        'forward_mean': forward_times.mean(),
        'forward_std': forward_times.std(),
        'backward_mean': backward_times.mean(),
        'backward_std': backward_times.std(),
        'total_mean': total_mean
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUDA Shared Memory Optimization Benchmark")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA is not available!")
        exit(1)
    
    print(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    
    # Run benchmarks with different configurations
    configs = [
        # (n_rays, n_samples, n_gaussians)
        (1024, 128, 100),    # Small
        (2048, 256, 200),    # Medium
        (4096, 256, 500),    # Large
    ]
    
    results = []
    for n_rays, n_samples, n_gaussians in configs:
        result = benchmark_rendering(
            n_rays=n_rays,
            n_samples=n_samples,
            n_gaussians=n_gaussians,
            n_warmup=5,
            n_iterations=20
        )
        results.append({
            'config': (n_rays, n_samples, n_gaussians),
            'result': result
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)'}")
    print("-"*60)
    for item in results:
        cfg = item['config']
        res = item['result']
        config_str = f"{cfg[0]}x{cfg[1]}x{cfg[2]}"
        print(f"{config_str:<30} {res['forward_mean']:<15.2f} {res['backward_mean']:<15.2f} {res['total_mean']:.2f}")
    print("="*60)
    
    print("\n✓ Benchmark complete!")


"""
CUDA Profiling Script to identify performance bottlenecks
"""

import torch
import time
import numpy as np

try:
    from gaussian_model.cuda_autograd import create_cuda_render_benchmark_module
    print("✓ CUDA renderer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CUDA renderer: {e}")
    exit(1)


def profile_memory_access():
    """
    Profile to see if shared memory is actually helping
    """
    device = torch.device("cuda:0")
    
    # Test different N_samples to see the effect
    configs = [
        (2048, 32, 200),    # Small samples
        (2048, 64, 200),    # Medium samples
        (2048, 128, 200),   # More samples
        (2048, 256, 200),   # Large samples
        (2048, 512, 200),   # Very large samples (exceeds MAX_T_SAMPLES_SHARED)
    ]
    
    print("\n" + "="*80)
    print("PROFILING: Effect of N_samples on Shared Memory Benefit")
    print("="*80)
    
    for n_rays, n_samples, n_gaussians in configs:
        print(f"\n{'='*80}")
        print(f"Config: {n_rays} rays × {n_samples} samples × {n_gaussians} gaussians")
        print(f"{'='*80}")
        
        # Generate test data
        ray_origins = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
        ray_directions = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
        
        t_samples = torch.linspace(0.5, 3.0, n_samples, device=device, dtype=torch.float32)
        
        gaussian_means = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32)
        gaussian_scales = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32) * 0.5
        gaussian_rotations = torch.randn(n_gaussians, 4, device=device, dtype=torch.float32)
        gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
        gaussian_opacities = torch.randn(n_gaussians, 1, device=device, dtype=torch.float32)
        gaussian_features = torch.randn(n_gaussians, 16, device=device, dtype=torch.float32)
        
        camera_pos = torch.zeros(3, device=device, dtype=torch.float32)
        
        active_sh_degree = 0
        c = 1.0
        deltaT = 0.01
        scaling_modifier = 1.0
        use_occlusion = True
        
        # Warmup
        renderer_global = create_cuda_render_benchmark_module(memory_mode='global')
        renderer_shared = create_cuda_render_benchmark_module(memory_mode='shared')
        
        for _ in range(3):
            _ = renderer_global(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
            _ = renderer_shared(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
        
        torch.cuda.synchronize()
        
        # Benchmark
        n_iter = 50
        
        # Global
        times_global = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer_global(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
            torch.cuda.synchronize()
            times_global.append((time.perf_counter() - start) * 1000)
        
        # Shared
        times_shared = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer_shared(
                ray_origins, ray_directions, t_samples,
                gaussian_means, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_features, camera_pos,
                active_sh_degree, c, deltaT, scaling_modifier, use_occlusion
            )
            torch.cuda.synchronize()
            times_shared.append((time.perf_counter() - start) * 1000)
        
        avg_global = np.mean(times_global)
        avg_shared = np.mean(times_shared)
        speedup = avg_global / avg_shared
        
        print(f"Global Memory:  {avg_global:.3f} ms ± {np.std(times_global):.3f}")
        print(f"Shared Memory:  {avg_shared:.3f} ms ± {np.std(times_shared):.3f}")
        print(f"Speedup:        {speedup:.3f}x")
        print(f"Note: N_samples {n_samples} {'>' if n_samples > 512 else '<='} MAX_T_SAMPLES_SHARED (512)")


def analyze_bottleneck():
    """
    Analyze where the actual bottleneck is
    """
    device = torch.device("cuda:0")
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    n_rays = 2048
    n_samples = 256
    n_gaussians = 200
    
    # Generate test data
    ray_origins = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
    ray_directions = torch.randn(n_rays, 3, device=device, dtype=torch.float32)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    
    t_samples = torch.linspace(0.5, 3.0, n_samples, device=device, dtype=torch.float32)
    
    gaussian_means = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32)
    gaussian_scales = torch.randn(n_gaussians, 3, device=device, dtype=torch.float32) * 0.5
    gaussian_rotations = torch.randn(n_gaussians, 4, device=device, dtype=torch.float32)
    gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
    gaussian_opacities = torch.randn(n_gaussians, 1, device=device, dtype=torch.float32)
    gaussian_features = torch.randn(n_gaussians, 16, device=device, dtype=torch.float32)
    
    camera_pos = torch.zeros(3, device=device, dtype=torch.float32)
    
    active_sh_degree = 0
    c = 1.0
    deltaT = 0.01
    scaling_modifier = 1.0
    use_occlusion = True
    
    print("\nMemory Access Estimates:")
    print(f"  t_samples accesses per forward:    {n_rays * n_samples:,}")
    print(f"  Gaussian param accesses:           {n_rays * n_samples * n_gaussians * 15:,}")
    print(f"    (mean: 3, scale: 3, quat: 4, opacity: 1, features: 4 for SH degree 0)")
    print(f"  Ratio (Gaussian/t_samples):        {(n_gaussians * 15):.1f}x")
    
    print("\nInterpretation:")
    print("  - If Gaussian param access >> t_samples access, shared memory benefit is small")
    print("  - Bottleneck is likely Gaussian parameter loading from global memory")
    print("  - To see bigger speedup, need to cache Gaussian parameters too")
    
    # Estimate theoretical speedup
    t_samples_ratio = 1 / (1 + n_gaussians * 15)
    print(f"\nTheoretical Analysis:")
    print(f"  t_samples memory accesses: ~{t_samples_ratio*100:.1f}% of total")
    print(f"  Even with 40x speedup on t_samples, overall speedup: ~{1/(1 - t_samples_ratio*(1-1/40)):.2f}x")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CUDA Performance Profiling")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("✗ CUDA is not available!")
        exit(1)
    
    print(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    
    analyze_bottleneck()
    profile_memory_access()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. t_samples 최적화는 효과가 제한적임 (전체의 ~5% 미만)
   → 주 병목은 Gaussian 파라미터 로딩

2. 더 큰 성능 향상을 위해서는:
   a) Gaussian 파라미터도 shared memory에 캐싱 (복잡함)
   b) Texture memory 사용 (read-only 데이터)
   c) atomicAdd 최적화 (backward pass, 3-5배 향상)
   
3. 현재 L1/L2 cache가 이미 잘 작동하고 있을 가능성:
   - 같은 t_samples가 여러 thread에서 반복 접근
   - Cache hit rate가 높아서 shared memory 이점이 적음

4. NVIDIA Nsight Compute로 정밀 분석 권장:
   nsys profile python benchmark_cuda.py
    """)
    
    print("\n✓ Profiling complete!")


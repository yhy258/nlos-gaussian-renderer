"""
Check Gaussian Filtering Statistics
Analyzes how many Gaussians are filtered per ray to determine optimal MAX_GAUSSIANS_PER_RAY
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from nlos_gaussian_renderer import _C
    CUDA_RENDERER_AVAILABLE = True
except ImportError as e:
    print(f"✗ CUDA renderer not available: {e}")
    CUDA_RENDERER_AVAILABLE = False
    exit(1)

device = torch.device("cuda:0")

def analyze_gaussian_filtering(n_rays, n_samples, n_gaussians, scene_type="random"):
    """
    Analyze Gaussian filtering statistics for different scene configurations
    
    Args:
        n_rays: Number of rays to test
        n_samples: Number of samples per ray
        n_gaussians: Number of Gaussians in scene
        scene_type: 'random', 'clustered', or 'uniform'
    """
    print(f"\n{'='*70}")
    print(f"Scene: {scene_type.upper()} | {n_gaussians} Gaussians | {n_rays} Rays")
    print(f"{'='*70}\n")
    
    # Generate test data
    if scene_type == "random":
        # Random scattered Gaussians
        gaussian_means = torch.randn(n_gaussians, 3, device=device) * 2.0
    elif scene_type == "clustered":
        # Clustered Gaussians (more realistic)
        n_clusters = 10
        cluster_centers = torch.randn(n_clusters, 3, device=device) * 3.0
        cluster_ids = torch.randint(0, n_clusters, (n_gaussians,), device=device)
        gaussian_means = cluster_centers[cluster_ids] + torch.randn(n_gaussians, 3, device=device) * 0.3
    else:  # uniform
        # Uniformly distributed
        gaussian_means = (torch.rand(n_gaussians, 3, device=device) - 0.5) * 4.0
    
    gaussian_scales = torch.randn(n_gaussians, 3, device=device) * 0.5 - 1.0  # log scale
    gaussian_rotations = torch.randn(n_gaussians, 4, device=device)
    gaussian_rotations = gaussian_rotations / gaussian_rotations.norm(dim=1, keepdim=True)
    gaussian_opacities = torch.randn(n_gaussians, 1, device=device)
    gaussian_features = torch.randn(n_gaussians, 16, device=device)
    
    # Generate rays (typical setup)
    ray_origins = torch.zeros(n_rays, 3, device=device)
    ray_directions = torch.randn(n_rays, 3, device=device)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    
    t_samples = torch.linspace(0.5, 3.0, n_samples, device=device)
    camera_pos = torch.zeros(3, device=device)
    
    # Compute bounding boxes
    print("Computing Gaussian bounding boxes...")
    gaussian_bboxes = torch.empty(n_gaussians, 6, device=device)
    
    from submodules.cuda_renderer import _C as cuda_ops
    # We'll use the internal bbox computation if available, otherwise approximate
    try:
        blocks = (n_gaussians + 255) // 256
        # Call internal CUDA function if available
        # For now, let's just call render and get the filter result
    except:
        pass
    
    # Call render to get filtering results
    print("Rendering and analyzing filtering...")
    try:
        result = _C.render_rays_shared(
            ray_origins,
            ray_directions,
            t_samples,
            gaussian_means,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            gaussian_features,
            camera_pos,
            0,  # active_sh_degree
            1.0,  # c
            0.01,  # deltaT
            1.0,  # scaling_modifier
            True  # use_occlusion
        )
        
        # Unpack results
        rho_density, density, transmittance, gaussian_bboxes, gaussian_filter, forward_cache = result
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        return None
    
    # Analyze filtering results
    # gaussian_filter: [N_rays, MAX_GAUSSIANS_PER_RAY+1]
    # First column is count
    num_gaussians_per_ray = gaussian_filter[:, 0].cpu()
    
    print(f"\n{'='*70}")
    print("FILTERING STATISTICS")
    print(f"{'='*70}")
    
    stats = {
        'min': num_gaussians_per_ray.min().item(),
        'max': num_gaussians_per_ray.max().item(),
        'mean': num_gaussians_per_ray.float().mean().item(),
        'median': num_gaussians_per_ray.float().median().item(),
        'std': num_gaussians_per_ray.float().std().item(),
        'p25': torch.quantile(num_gaussians_per_ray.float(), 0.25).item(),
        'p75': torch.quantile(num_gaussians_per_ray.float(), 0.75).item(),
        'p90': torch.quantile(num_gaussians_per_ray.float(), 0.90).item(),
        'p95': torch.quantile(num_gaussians_per_ray.float(), 0.95).item(),
        'p99': torch.quantile(num_gaussians_per_ray.float(), 0.99).item(),
    }
    
    print(f"  Min:        {stats['min']:.0f} gaussians/ray")
    print(f"  Max:        {stats['max']:.0f} gaussians/ray")
    print(f"  Mean:       {stats['mean']:.1f} gaussians/ray")
    print(f"  Median:     {stats['median']:.1f} gaussians/ray")
    print(f"  Std Dev:    {stats['std']:.1f}")
    print(f"\nPercentiles:")
    print(f"  P25:        {stats['p25']:.1f}")
    print(f"  P75:        {stats['p75']:.1f}")
    print(f"  P90:        {stats['p90']:.1f}")
    print(f"  P95:        {stats['p95']:.1f}")
    print(f"  P99:        {stats['p99']:.1f}")
    
    # Check if hitting limit
    max_limit = 256  # Current MAX_GAUSSIANS_PER_RAY
    rays_at_limit = (num_gaussians_per_ray >= max_limit).sum().item()
    
    print(f"\n{'='*70}")
    print("LIMIT ANALYSIS")
    print(f"{'='*70}")
    print(f"  Current MAX_GAUSSIANS_PER_RAY: {max_limit}")
    print(f"  Rays hitting limit:             {rays_at_limit} / {n_rays} ({rays_at_limit/n_rays*100:.1f}%)")
    
    if rays_at_limit > 0:
        print(f"\n  ⚠️  WARNING: {rays_at_limit} rays are hitting the limit!")
        print(f"      This means some Gaussians are being SILENTLY IGNORED!")
        print(f"      Rendering quality may be degraded.")
    else:
        print(f"\n  ✅ OK: No rays hitting the limit")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    recommended_limit = int(stats['p99'] * 1.2)  # 20% margin above p99
    
    if rays_at_limit > n_rays * 0.01:  # More than 1% hitting limit
        print(f"  🔴 INCREASE MAX_GAUSSIANS_PER_RAY to at least: {recommended_limit}")
        print(f"     Current: {max_limit} → Recommended: {recommended_limit}")
        print(f"\n  ⚠️  WARNING: Stack memory may be limited!")
        print(f"     Consider global memory allocation for backward pass.")
    elif stats['p95'] > max_limit * 0.8:  # Using >80% of capacity
        print(f"  🟡 CONSIDER increasing MAX_GAUSSIANS_PER_RAY to: {recommended_limit}")
        print(f"     Current: {max_limit} → Recommended: {recommended_limit}")
    else:
        print(f"  🟢 Current MAX_GAUSSIANS_PER_RAY ({max_limit}) is sufficient")
        print(f"     P99: {stats['p99']:.1f} ({stats['p99']/max_limit*100:.1f}% of capacity)")
    
    # Memory estimation
    print(f"\n{'='*70}")
    print("MEMORY ANALYSIS (Forward Cache)")
    print(f"{'='*70}")
    
    compact_cache_size = 3 * 4  # 3 floats × 4 bytes
    cache_memory_mb = (n_rays * n_samples * max_limit * compact_cache_size) / (1024**2)
    
    print(f"  Compact cache (3 floats): {cache_memory_mb:.1f} MB")
    print(f"  Configuration: {n_rays} rays × {n_samples} samples × {max_limit} gaussians")
    
    if recommended_limit > max_limit:
        new_cache_memory_mb = (n_rays * n_samples * recommended_limit * compact_cache_size) / (1024**2)
        print(f"\n  With recommended limit ({recommended_limit}):")
        print(f"    Cache memory: {new_cache_memory_mb:.1f} MB ({new_cache_memory_mb/cache_memory_mb:.2f}x)")
    
    return stats, num_gaussians_per_ray


def plot_distribution(num_gaussians_per_ray, title="Gaussians per Ray Distribution"):
    """Plot histogram of Gaussians per ray"""
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    counts = num_gaussians_per_ray.numpy()
    plt.hist(counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(counts.mean(), color='red', linestyle='--', label=f'Mean: {counts.mean():.1f}')
    plt.axvline(np.percentile(counts, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(counts, 95):.1f}')
    plt.axvline(256, color='green', linestyle='--', linewidth=2, label='Current Limit: 256')
    plt.xlabel('Gaussians per Ray')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CDF
    plt.subplot(1, 2, 2)
    sorted_counts = np.sort(counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cdf, linewidth=2)
    plt.axvline(256, color='green', linestyle='--', linewidth=2, label='Current Limit: 256')
    plt.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95th percentile')
    plt.axhline(0.99, color='red', linestyle='--', alpha=0.5, label='99th percentile')
    plt.xlabel('Gaussians per Ray')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("./stats")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "gaussian_filtering_stats.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_dir / 'gaussian_filtering_stats.png'}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GAUSSIAN FILTERING STATISTICS ANALYZER")
    print("="*70)
    
    # Test configurations
    configs = [
        # (n_rays, n_samples, n_gaussians, scene_type)
        (4096, 300, 10000, "random"),
        (4096, 300, 50000, "random"),
        (4096, 300, 100000, "random"),
        (4096, 300, 10000, "clustered"),
        (4096, 300, 50000, "clustered"),
    ]
    
    all_results = []
    
    for n_rays, n_samples, n_gaussians, scene_type in configs:
        stats, counts = analyze_gaussian_filtering(n_rays, n_samples, n_gaussians, scene_type)
        if stats is not None:
            all_results.append({
                'n_gaussians': n_gaussians,
                'scene_type': scene_type,
                'stats': stats,
                'counts': counts
            })
    
    # Plot the most realistic case (clustered, 50k gaussians)
    if all_results:
        for result in all_results:
            if result['n_gaussians'] == 50000 and result['scene_type'] == 'clustered':
                plot_distribution(
                    result['counts'],
                    f"Gaussians per Ray ({result['n_gaussians']} Gaussians, {result['scene_type']})"
                )
                break
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


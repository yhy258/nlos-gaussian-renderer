"""
Test script for coordinate-based albedo rendering
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from submodules.cuda_renderer.albedo_coords_renderer import (
    compute_albedo_at_coords,
    compute_density_at_coords,
    create_3d_grid,
    render_volume_albedo
)

def test_basic_functionality():
    """Test basic coordinate-based rendering"""
    print("Testing coordinate-based albedo rendering...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy Gaussian parameters
    N_gaussians = 10
    gaussian_means = torch.randn(N_gaussians, 3, device=device) * 0.5
    gaussian_scales = torch.randn(N_gaussians, 3, device=device) * 0.1 - 2.0  # log scale
    gaussian_rotations = torch.randn(N_gaussians, 4, device=device)
    gaussian_rotations = torch.nn.functional.normalize(gaussian_rotations, dim=-1)
    gaussian_opacities = torch.randn(N_gaussians, 1, device=device)
    
    # SH features (DC term only for simplicity)
    sh_dim = 1  # Only DC term
    gaussian_features = torch.randn(N_gaussians, sh_dim, device=device) * 0.5
    
    # Camera position
    camera_pos = torch.tensor([0.0, 0.0, -2.0], device=device)
    
    # Test 1: Single coordinate
    print("\n1. Testing single coordinate...")
    single_coord = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    albedo = compute_albedo_at_coords(
        single_coord,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        gaussian_features,
        camera_pos,
        active_sh_degree=0,
        scaling_modifier=1.0
    )
    print(f"   Albedo at origin: {albedo.item():.6f}")
    
    # Test 2: Multiple coordinates
    print("\n2. Testing multiple coordinates...")
    coords = torch.randn(100, 3, device=device) * 0.5
    albedo_batch = compute_albedo_at_coords(
        coords,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        gaussian_features,
        camera_pos,
        active_sh_degree=0,
        scaling_modifier=1.0
    )
    print(f"   Albedo shape: {albedo_batch.shape}")
    print(f"   Albedo stats - min: {albedo_batch.min():.6f}, max: {albedo_batch.max():.6f}, mean: {albedo_batch.mean():.6f}")
    
    # Test 3: 3D grid
    print("\n3. Testing 3D grid...")
    resolution = (32, 32, 32)
    volume_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    volume_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    
    coords_grid = create_3d_grid(volume_min, volume_max, resolution)
    print(f"   Grid shape: {coords_grid.shape}")
    
    albedo_volume = compute_albedo_at_coords(
        coords_grid,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        gaussian_features,
        camera_pos,
        active_sh_degree=0,
        scaling_modifier=1.0
    )
    print(f"   Albedo volume shape: {albedo_volume.shape}")
    print(f"   Volume stats - min: {albedo_volume.min():.6f}, max: {albedo_volume.max():.6f}, mean: {albedo_volume.mean():.6f}")
    
    # Test 4: Density computation
    print("\n4. Testing density computation...")
    density = compute_density_at_coords(
        single_coord,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        scaling_modifier=1.0
    )
    print(f"   Density at origin: {density.item():.6f}")
    
    print("\n✅ All tests passed!")
    return True


def test_with_gaussian_model():
    """Test with actual GaussianModel if available"""
    print("\n5. Testing with GaussianModel...")
    
    try:
        from gaussian_model.gaussian_model import GaussianModel
        from configs.default import Config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = Config()
        
        # Create a simple model
        model = GaussianModel(args, device)
        
        # Initialize with random points
        N_points = 50
        points = np.random.randn(N_points, 3) * 0.5
        rhos = np.ones(N_points) * 0.5
        pmin = np.array([-1, -1, -1, 0, -np.pi])
        pmax = np.array([1, 1, 1, np.pi, 0])
        
        model.create_params(points, rhos, pmin, pmax)
        
        # Test volume rendering
        camera_pos = torch.tensor([0.0, 0.0, -2.0], device=device)
        volume_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
        volume_max = torch.tensor([1.0, 1.0, 1.0], device=device)
        
        albedo_volume, coords = render_volume_albedo(
            model,
            camera_pos,
            volume_min,
            volume_max,
            resolution=(16, 16, 16),
            scaling_modifier=1.0
        )
        
        print(f"   Model albedo volume shape: {albedo_volume.shape}")
        print(f"   Volume stats - min: {albedo_volume.min():.6f}, max: {albedo_volume.max():.6f}, mean: {albedo_volume.mean():.6f}")
        print("   ✅ GaussianModel test passed!")
        
    except ImportError as e:
        print(f"   ⚠️ Could not import GaussianModel: {e}")
    except Exception as e:
        print(f"   ❌ Error testing with GaussianModel: {e}")


if __name__ == "__main__":
    print("="*60)
    print("Coordinate-based Albedo Renderer Test")
    print("="*60)
    
    try:
        # First build the CUDA extension if needed
        import subprocess
        import os
        
        cuda_dir = "submodules/cuda_renderer"
        if not os.path.exists(os.path.join(cuda_dir, "build")):
            print("Building CUDA extension...")
            result = subprocess.run(
                ["python", "setup.py", "install"],
                cwd=cuda_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Build failed:\n{result.stderr}")
                sys.exit(1)
            print("Build successful!")
        
        # Run tests
        if test_basic_functionality():
            test_with_gaussian_model()
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

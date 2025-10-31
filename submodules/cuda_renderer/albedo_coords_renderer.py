"""
Coordinate-based albedo renderer for NLOS Gaussians
Evaluates albedo at given 3D coordinates instead of ray marching
"""

import torch
from typing import Optional, Tuple
from gaussian_model.gaussian_model import GaussianModel

def compute_albedo_at_coords(
    coords: torch.Tensor,           # [H, W, D, 3] or any shape [..., 3]
    gaussian_model: GaussianModel,
    camera_pos: torch.Tensor,          # [3]
    active_sh_degree: int = 0,
    scaling_modifier: float = 1.0,
    sigma_threshold: float = 3.0       # Sigma threshold for AABB culling
) -> torch.Tensor:
    """
    Compute albedo at given 3D coordinates.
    
    Args:
        coords: 3D coordinates to evaluate, shape [..., 3]
        gaussian_means: Gaussian center positions [N, 3]
        gaussian_scales: Gaussian scales (before activation) [N, 3]
        gaussian_rotations: Gaussian rotations as quaternions [N, 4]
        gaussian_opacities: Gaussian opacities (before activation) [N, 1]
        gaussian_features: SH features for view-dependent albedo [N, K]
        camera_pos: Camera position for view direction [3]
        active_sh_degree: Active SH degree for rendering
        scaling_modifier: Scale modifier for Gaussians
        sigma_threshold: Number of standard deviations for AABB culling (default 3.0)
        
    Returns:
        Albedo values at the given coordinates, shape [...]
    """
    try:
        from nlos_gaussian_renderer._C import compute_albedo_at_coords as _compute_albedo
    except ImportError:
        raise ImportError(
            "CUDA renderer not built. Please run:\n"
            "cd submodules/cuda_renderer && python setup.py install"
        )
    
    gaussian_means = gaussian_model.get_mu
    gaussian_scales = gaussian_model.get_scaling
    gaussian_rotations = gaussian_model.get_rotation
    gaussian_opacities = gaussian_model.get_opacity
    gaussian_features = gaussian_model.get_features


    # Ensure inputs are contiguous and on the same device
    device = coords.device
    coords = coords.contiguous().to(device)
    gaussian_means = gaussian_means.contiguous().to(device)
    
    # Apply activation functions to scales and opacities
    gaussian_scales_activated = gaussian_scales.contiguous().to(device)
    gaussian_opacities_activated = gaussian_opacities.contiguous().to(device)
    
    # Normalize quaternions
    gaussian_rotations = gaussian_rotations.contiguous().to(device)
    
    gaussian_features = gaussian_features.contiguous().to(device)
    camera_pos = camera_pos.contiguous().to(device)
    
    # Call CUDA kernel with AABB filtering
    albedo = _compute_albedo(
        coords,
        gaussian_means,
        gaussian_scales_activated,
        gaussian_rotations,
        gaussian_opacities_activated,
        gaussian_features,
        camera_pos,
        active_sh_degree,
        scaling_modifier,
        sigma_threshold
    )
    
    return albedo


def compute_density_at_coords(
    coords: torch.Tensor,           # [H, W, D, 3] or any shape [..., 3]
    gaussian_model: GaussianModel,
    scaling_modifier: float = 1.0,
    sigma_threshold: float = 3.0
) -> torch.Tensor:
    """
    Compute density at given 3D coordinates.
    
    Args:
        coords: 3D coordinates to evaluate, shape [..., 3]
        gaussian_means: Gaussian center positions [N, 3]
        gaussian_scales: Gaussian scales (before activation) [N, 3]
        gaussian_rotations: Gaussian rotations as quaternions [N, 4]
        gaussian_opacities: Gaussian opacities (before activation) [N, 1]
        scaling_modifier: Scale modifier for Gaussians
        sigma_threshold: Number of standard deviations for AABB culling (default 3.0)
        
    Returns:
        Density values at the given coordinates, shape [...]
    """
    try:
        from nlos_gaussian_renderer._C import compute_density_at_coords as _compute_density
    except ImportError:
        raise ImportError(
            "CUDA renderer not built. Please run:\n"
            "cd submodules/cuda_renderer && python setup.py install"
        )
    
    # Ensure inputs are contiguous and on the same device
    device = coords.device
    coords = coords.contiguous().to(device)
    gaussian_means = gaussian_model.get_mu.contiguous().to(device)
    gaussian_scales = gaussian_model.get_scaling.contiguous().to(device)
    gaussian_rotations = gaussian_model.get_rotation.contiguous().to(device)
    gaussian_opacities = gaussian_model.get_opacity.contiguous().to(device)
    gaussian_features = gaussian_model.get_features.contiguous().to(device)
    
    # Call CUDA kernel with AABB filtering
    density = _compute_density(
        coords,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        scaling_modifier,
        sigma_threshold
    )
    
    return density


def create_3d_grid(
    volume_min: torch.Tensor,  # [3]
    volume_max: torch.Tensor,  # [3]
    resolution: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Create a 3D coordinate grid for volume evaluation.
    
    Args:
        volume_min: Minimum coordinates of the volume [3]
        volume_max: Maximum coordinates of the volume [3]
        resolution: Grid resolution (H, W, D)
        
    Returns:
        3D coordinate grid of shape [H, W, D, 3]
    """
    H, W, D = resolution
    device = volume_min.device
    
    # Create 1D coordinate arrays
    x = torch.linspace(volume_min[0], volume_max[0], W, device=device)
    y = torch.linspace(volume_min[1], volume_max[1], H, device=device)
    z = torch.linspace(volume_min[2], volume_max[2], D, device=device)
    
    # Create meshgrid
    yy, xx, zz = torch.meshgrid(y, x, z, indexing='ij')
    
    # Stack to create coordinate grid
    coords = torch.stack([xx, yy, zz], dim=-1)
    
    return coords


# Example usage function
def render_volume_albedo(
    model,  # GaussianModel instance
    camera_pos: torch.Tensor,
    volume_min: torch.Tensor,
    volume_max: torch.Tensor,
    resolution: Tuple[int, int, int] = (128, 128, 128),
    scaling_modifier: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render albedo for a 3D volume using coordinate-based evaluation.
    
    Args:
        model: GaussianModel instance with parameters
        camera_pos: Camera position [3]
        volume_min: Minimum volume coordinates [3]
        volume_max: Maximum volume coordinates [3]
        resolution: Volume resolution (H, W, D)
        scaling_modifier: Gaussian scale modifier
        
    Returns:
        albedo_volume: Albedo values [H, W, D]
        coords: 3D coordinate grid [H, W, D, 3]
    """
    # Create 3D coordinate grid
    coords = create_3d_grid(volume_min, volume_max, resolution)
    
    # Get model parameters
    gaussian_means = model.get_mu
    gaussian_scales = model._scaling
    gaussian_rotations = model._rotation
    gaussian_opacities = model._opacity
    gaussian_features = model.get_features
    
    # Compute albedo at coordinates
    albedo_volume = compute_albedo_at_coords(
        coords,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        gaussian_features,
        camera_pos,
        model.active_sh_degree,
        scaling_modifier
    )
    
    return albedo_volume, coords

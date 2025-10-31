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
    
    # Ensure inputs are contiguous and on the same device
    device = coords.device
    coords = coords.contiguous().to(device)
    gaussian_means = gaussian_model.get_mu.contiguous().to(device)
    gaussian_scales = gaussian_model.get_scaling.contiguous().to(device)
    gaussian_rotations = gaussian_model.get_rotation.contiguous().to(device)
    gaussian_opacities = gaussian_model.get_opacity.contiguous().to(device)
    gaussian_features = gaussian_model.get_features.contiguous().to(device)
    camera_pos = camera_pos.contiguous().to(device)
    
    # Call CUDA kernel with AABB filtering
    albedo = _compute_albedo(
        coords,
        gaussian_means,
        gaussian_scales,
        gaussian_rotations,
        gaussian_opacities,
        gaussian_features,
        camera_pos,
        gaussian_model.active_sh_degree,
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
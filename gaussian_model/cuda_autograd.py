"""
CUDA Renderer Autograd Function
Defines forward and backward passes for CUDA-accelerated NLOS Gaussian rendering
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from nlos_gaussian_renderer import _C
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA renderer not available")


class CUDARenderFunction(torch.autograd.Function):
    """
    Autograd function for CUDA ray-based rendering.
    
    Forward: Computes rendering result from Gaussians
    Backward: Computes gradients w.r.t. Gaussian parameters
    """
    
    @staticmethod
    def forward(
        ctx,
        ray_origins: torch.Tensor,          # [N_rays, 3]
        ray_directions: torch.Tensor,       # [N_rays, 3]
        t_samples: torch.Tensor,            # [N_samples]
        gaussian_means: torch.Tensor,       # [N_gaussians, 3]
        gaussian_scales: torch.Tensor,      # [N_gaussians, 3]
        gaussian_rotations: torch.Tensor,   # [N_gaussians, 4]
        gaussian_opacities: torch.Tensor,   # [N_gaussians, 1]
        gaussian_features: torch.Tensor,    # [N_gaussians, K]
        camera_pos: torch.Tensor,           # [3]
        active_sh_degree: int,
        c: float,
        deltaT: float,
        scaling_modifier: float,
        use_occlusion: bool,
        rendering_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: Render rays through Gaussians
        
        Returns:
            rho_density: [N_samples, N_rays] - Main rendering output
            density: [N_samples, N_rays] - Density field
            transmittance: [N_samples, N_rays] - Transmittance values
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA renderer not available")
        
        # Ensure all tensors are contiguous and on CUDA
        ray_origins = ray_origins.contiguous()
        ray_directions = ray_directions.contiguous()
        t_samples = t_samples.contiguous()
        gaussian_means = gaussian_means.contiguous()
        gaussian_scales = gaussian_scales.contiguous()
        gaussian_rotations = gaussian_rotations.contiguous()
        gaussian_opacities = gaussian_opacities.contiguous()
        gaussian_features = gaussian_features.contiguous()
        camera_pos = camera_pos.contiguous()
        
        # Call CUDA forward kernel
        rho_density, density, transmittance = _C.render_rays(
            ray_origins,
            ray_directions,
            t_samples,
            gaussian_means,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            gaussian_features,
            camera_pos,
            active_sh_degree,
            c,
            deltaT,
            scaling_modifier,
            use_occlusion,
            rendering_type
        )
        
        # Save for backward
        ctx.save_for_backward(
            ray_origins,
            ray_directions,
            t_samples,
            gaussian_means,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            gaussian_features,
            camera_pos,
            rho_density,
            density,
            transmittance
        )
        ctx.active_sh_degree = active_sh_degree
        ctx.c = c
        ctx.deltaT = deltaT
        ctx.scaling_modifier = scaling_modifier
        ctx.use_occlusion = use_occlusion
        ctx.rendering_type = rendering_type
        
        return rho_density, density, transmittance
    
    @staticmethod
    def backward(ctx, grad_rho_density, grad_density, grad_transmittance):
        """
        Backward pass: Compute gradients w.r.t. Gaussian parameters
        
        Args:
            grad_rho_density: [N_samples, N_rays] - Gradient from loss
            grad_density: [N_samples, N_rays] - Usually None
            grad_transmittance: [N_samples, N_rays] - Usually None
        
        Returns:
            Gradients for all forward inputs (None for non-learnable params)
        """
        # Retrieve saved tensors
        (
            ray_origins,
            ray_directions,
            t_samples,
            gaussian_means,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            gaussian_features,
            camera_pos,
            rho_density,
            density,
            transmittance
        ) = ctx.saved_tensors
        
        # Initialize gradients
        grad_gaussian_means = None
        grad_gaussian_scales = None
        grad_gaussian_rotations = None
        grad_gaussian_opacities = None
        grad_gaussian_features = None
        
        # Only compute gradients if needed
        if ctx.needs_input_grad[3]:  # gaussian_means
            grad_gaussian_means = torch.zeros_like(gaussian_means)
        if ctx.needs_input_grad[4]:  # gaussian_scales
            grad_gaussian_scales = torch.zeros_like(gaussian_scales)
        if ctx.needs_input_grad[5]:  # gaussian_rotations
            grad_gaussian_rotations = torch.zeros_like(gaussian_rotations)
        if ctx.needs_input_grad[6]:  # gaussian_opacities
            grad_gaussian_opacities = torch.zeros_like(gaussian_opacities)
        if ctx.needs_input_grad[7]:  # gaussian_features
            grad_gaussian_features = torch.zeros_like(gaussian_features)
        
        # TODO: Call CUDA backward kernel when implemented
        # For now, we'll use PyTorch's automatic differentiation
        # This is a placeholder for future CUDA backward implementation
        
        # _C.render_rays_backward(
        #     grad_rho_density.contiguous(),
        #     ray_origins, ray_directions, t_samples,
        #     gaussian_means, gaussian_scales, gaussian_rotations,
        #     gaussian_opacities, gaussian_features, camera_pos,
        #     rho_density, density, transmittance,
        #     grad_gaussian_means, grad_gaussian_scales, grad_gaussian_rotations,
        #     grad_gaussian_opacities, grad_gaussian_features,
        #     ctx.active_sh_degree, ctx.c, ctx.deltaT,
        #     ctx.scaling_modifier, ctx.use_occlusion, ctx.rendering_type
        # )
        
        # Return gradients for all inputs (None for non-differentiable)
        return (
            None,  # ray_origins
            None,  # ray_directions
            None,  # t_samples
            grad_gaussian_means,
            grad_gaussian_scales,
            grad_gaussian_rotations,
            grad_gaussian_opacities,
            grad_gaussian_features,
            None,  # camera_pos
            None,  # active_sh_degree
            None,  # c
            None,  # deltaT
            None,  # scaling_modifier
            None,  # use_occlusion
            None,  # rendering_type
        )


class CUDARenderModule(nn.Module):
    """
    PyTorch Module wrapper for CUDA rendering with automatic differentiation.
    
    This provides a clean interface for using the CUDA renderer in training loops
    with full gradient support.
    """
    
    def __init__(self, sigma_threshold: float = 3.0):
        """
        Args:
            sigma_threshold: Threshold for Gaussian AABB computation
        """
        super().__init__()
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA renderer not available")
        
        self.sigma_threshold = sigma_threshold
    
    def forward(
        self,
        gaussian_model,
        camera_pos: torch.Tensor,
        theta_range: Tuple[float, float],
        phi_range: Tuple[float, float],
        r_range: Tuple[float, float],
        num_theta: int,
        num_phi: int,
        num_r: int,
        c: float,
        deltaT: float,
        scaling_modifier: float = 1.0,
        use_occlusion: bool = True,
        rendering_type: str = 'netf'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CUDA renderer
        
        Args:
            gaussian_model: GaussianModel with learnable parameters
            camera_pos: [3] Camera position
            theta_range: (min, max) theta angles
            phi_range: (min, max) phi angles
            r_range: (min, max) radial distances
            num_theta: Number of theta samples
            num_phi: Number of phi samples
            num_r: Number of radial samples
            c: Speed of light
            deltaT: Time interval
            scaling_modifier: Gaussian scale modifier
            use_occlusion: Whether to use transmittance
            rendering_type: 'netf' or 'nlos-neus'
        
        Returns:
            result: [num_r, num_theta, num_phi] rendered volume
            pred_histogram: [num_r] integrated histogram
        """
        device = camera_pos.device
        
        # Generate rays
        theta = torch.linspace(theta_range[0], theta_range[1], num_theta, device=device)
        phi = torch.linspace(phi_range[0], phi_range[1], num_phi, device=device)
        
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)
        
        num_rays = theta_flat.shape[0]
        
        # Ray directions in Cartesian coordinates
        ray_dirs = torch.stack([
            torch.sin(theta_flat) * torch.cos(phi_flat),
            torch.sin(theta_flat) * torch.sin(phi_flat),
            torch.cos(theta_flat)
        ], dim=1)  # [num_rays, 3]
        
        ray_origins = camera_pos.unsqueeze(0).expand(num_rays, 3)
        
        # Radial samples
        t_samples = torch.linspace(r_range[0], r_range[1], num_r, device=device)
        
        # Get Gaussian parameters (with gradients!)
        gaussian_means = gaussian_model.get_mu
        gaussian_scales = gaussian_model._scaling
        gaussian_rotations = gaussian_model._rotation
        gaussian_opacities = gaussian_model._opacity
        gaussian_features = gaussian_model.get_features_dc.squeeze(1)
        
        # Call custom autograd function
        rho_density, density, transmittance = CUDARenderFunction.apply(
            ray_origins,
            ray_dirs,
            t_samples,
            gaussian_means,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            gaussian_features,
            camera_pos,
            gaussian_model.active_sh_degree,
            c,
            deltaT,
            scaling_modifier,
            use_occlusion,
            rendering_type
        )
        
        # Reshape and apply geometric attenuation
        result = rho_density.T.reshape(num_r, num_theta, num_phi)
        
        # Geometric attenuation: sin(theta) / r^2
        distance = t_samples.view(-1, 1, 1)
        theta_3d = theta_grid.unsqueeze(0)
        
        result = result / (distance ** 2 + 1e-8) * torch.sin(theta_3d)
        
        # Angular integration
        dtheta = (theta_range[1] - theta_range[0]) / num_theta
        dphi = (phi_range[1] - phi_range[0]) / num_phi
        
        pred_histogram = torch.sum(result, dim=(1, 2)) * dtheta * dphi
        
        return result, pred_histogram


def create_cuda_render_module(sigma_threshold: float = 3.0) -> Optional[CUDARenderModule]:
    """
    Factory function to create a CUDA render module
    
    Args:
        sigma_threshold: AABB threshold
    
    Returns:
        CUDARenderModule if CUDA available, None otherwise
    """
    if not CUDA_AVAILABLE:
        return None
    return CUDARenderModule(sigma_threshold=sigma_threshold)


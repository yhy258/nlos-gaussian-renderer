"""
CUDA-accelerated rendering for NLOS Gaussian Model
Integrates the CUDA renderer with the GaussianModel with full autograd support
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from nlos_gaussian_renderer import _C
    CUDA_AVAILABLE = True
except ImportError:
    print("Warning: CUDA extension not compiled. Please run 'python setup.py install'")
    CUDA_AVAILABLE = False

# Import autograd-enabled CUDA rendering module
from .cuda_autograd import CUDARenderModule, create_cuda_render_module


# class NLOSGaussianRenderer:
#     """
#     Ray-based renderer for NLOS Gaussian reconstruction.
    
#     This renderer efficiently computes per-ray contributions from Gaussians
#     by filtering only relevant primitives and computing volume rendering
#     along each ray independently.
#     """
    
#     def __init__(self, sigma_threshold=3.0):
#         """
#         Args:
#             sigma_threshold: Number of standard deviations for AABB size (default: 3.0)
#         """
#         if not CUDA_AVAILABLE:
#             raise RuntimeError("CUDA extension is not available. Cannot use NLOSGaussianRenderer.")
        
#         self.sigma_threshold = sigma_threshold
    
#     def render(
#         self,
#         gaussian_model,
#         camera_pos,
#         theta_range,
#         phi_range,
#         r_range,
#         num_theta,
#         num_phi,
#         num_r,
#         c,
#         deltaT,
#         scaling_modifier=1.0,
#         use_occlusion=True,
#         rendering_type='netf'
#     ):
#         """
#         Render transient histogram using ray-based approach.
        
#         Args:
#             gaussian_model: GaussianModel instance
#             camera_pos: Camera position [3] tensor
#             theta_range: (theta_min, theta_max)
#             phi_range: (phi_min, phi_max)
#             r_range: (r_min, r_max)
#             num_theta: Number of theta samples
#             num_phi: Number of phi samples
#             num_r: Number of radial samples
#             c: Speed of light
#             deltaT: Time interval
#             scaling_modifier: Scale modifier for Gaussians
#             use_occlusion: Whether to use occlusion/transmittance
#             rendering_type: 'netf' or 'nlos-neus'
        
#         Returns:
#             result: [num_r, num_theta, num_phi] rendering result
#             pred_histogram: [num_r] predicted histogram after angular integration
#         """
#         device = camera_pos.device
        
#         # Generate rays for all (theta, phi) combinations
#         theta = torch.linspace(theta_range[0], theta_range[1], num_theta, device=device)
#         phi = torch.linspace(phi_range[0], phi_range[1], num_phi, device=device)
        
#         # Create angular grid
#         theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
#         theta_flat = theta_grid.reshape(-1)  # [num_theta * num_phi]
#         phi_flat = phi_grid.reshape(-1)      # [num_theta * num_phi]
        
#         num_rays = theta_flat.shape[0]
        
#         # Convert spherical to Cartesian directions
#         ray_dirs = torch.stack([
#             torch.sin(theta_flat) * torch.cos(phi_flat),
#             torch.sin(theta_flat) * torch.sin(phi_flat),
#             torch.cos(theta_flat)
#         ], dim=1)  # [num_rays, 3]
        
#         # All rays originate from camera position
#         ray_origins = camera_pos.unsqueeze(0).expand(num_rays, 3)  # [num_rays, 3]
        
#         # Radial samples (ray parameter t)
#         t_samples = torch.linspace(r_range[0], r_range[1], num_r, device=device)
        
#         # Get Gaussian parameters
#         gaussian_means = gaussian_model.get_mu                    # [N_g, 3]
#         gaussian_scales = gaussian_model._scaling                 # [N_g, 3]
#         gaussian_rotations = gaussian_model._rotation             # [N_g, 4]
#         gaussian_opacities = gaussian_model._opacity              # [N_g, 1]
#         gaussian_features = gaussian_model.get_features_dc        # [N_g, 1, K]
        
#         # Reshape features for rendering
#         sh_features = gaussian_features.squeeze(1)  # [N_g, K]
        
#         # Call CUDA rendering function
#         rho_density, density, transmittance = _C.render_rays(
#             ray_origins.contiguous(),
#             ray_dirs.contiguous(),
#             t_samples.contiguous(),
#             gaussian_means.contiguous(),
#             gaussian_scales.contiguous(),
#             gaussian_rotations.contiguous(),
#             gaussian_opacities.contiguous(),
#             sh_features.contiguous(),
#             camera_pos.contiguous(),
#             gaussian_model.active_sh_degree,
#             c,
#             deltaT,
#             scaling_modifier,
#             use_occlusion,
#             rendering_type
#         )
        
#         # Reshape to [num_r, num_theta, num_phi]
#         result = rho_density.T.reshape(num_r, num_theta, num_phi)
        
#         # Apply attenuation factor: sin(theta) / r^2
#         # Distance from camera
#         distance = t_samples.view(-1, 1, 1)  # [num_r, 1, 1]
#         theta_3d = theta_grid.unsqueeze(0)   # [1, num_theta, num_phi]
        
#         result = result / (distance ** 2 + 1e-8) * torch.sin(theta_3d)
        
#         # Compute angular integration
#         dtheta = (theta_range[1] - theta_range[0]) / num_theta
#         dphi = (phi_range[1] - phi_range[0]) / num_phi
        
#         pred_histogram = torch.sum(result, dim=(1, 2)) * dtheta * dphi  # [num_r]
        
#         return result, pred_histogram
    
#     def filter_gaussians(
#         self,
#         ray_origins,
#         ray_directions,
#         gaussian_means,
#         gaussian_bboxes
#     ):
#         """
#         Filter Gaussians that intersect with each ray.
        
#         Args:
#             ray_origins: [N_rays, 3]
#             ray_directions: [N_rays, 3]
#             gaussian_means: [N_gaussians, 3]
#             gaussian_bboxes: [N_gaussians, 2, 3] (min, max)
        
#         Returns:
#             Filtered indices: [N_rays, MAX_GAUSSIANS_PER_RAY+1]
#                 First column is count, rest are indices
#         """
#         return _C.filter_gaussians_per_ray(
#             ray_origins.contiguous(),
#             ray_directions.contiguous(),
#             gaussian_means.contiguous(),
#             gaussian_bboxes.contiguous().view(-1, 6),
#             self.sigma_threshold
#         )


# # Convenience function
# def create_renderer(sigma_threshold=3.0):
#     """Create an NLOSGaussianRenderer instance."""
#     return NLOSGaussianRenderer(sigma_threshold=sigma_threshold)




class GaussianRendererCUDA:
    """
    CUDA-accelerated renderer that integrates with GaussianModel.
    This is a wrapper that provides the same interface as the original
    rendering functions but uses efficient ray-based CUDA kernels with autograd.
    """
    
    def __init__(self, sigma_threshold=3.0):
        """
        Args:
            sigma_threshold: Number of standard deviations for Gaussian AABB (default: 3.0)
        """
        self.use_cuda = CUDA_AVAILABLE
        if self.use_cuda:
            # Use autograd-enabled module for gradient support
            self.renderer = CUDARenderModule(sigma_threshold=sigma_threshold)
        else:
            self.renderer = None
            print("CUDA renderer not available, will use fallback implementation")
    
    def render_transient(
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
        Render transient histogram using CUDA-accelerated ray-based rendering.
        
        Args:
            gaussian_model: GaussianModel instance
            camera_pos: Camera position [3] tensor
            theta_range: (theta_min, theta_max) in radians
            phi_range: (phi_min, phi_max) in radians
            r_range: (r_min, r_max) ray distances
            num_theta: Number of theta samples
            num_phi: Number of phi samples
            num_r: Number of radial samples
            c: Speed of light
            deltaT: Time interval
            scaling_modifier: Scale modifier for Gaussians
            use_occlusion: Whether to use occlusion/transmittance
            rendering_type: 'netf' or 'nlos-neus'
        
        Returns:
            result: [num_r, num_theta, num_phi] rendering result
            pred_histogram: [num_r] predicted histogram after angular integration
        """
        if not self.use_cuda or self.renderer is None:
            raise RuntimeError("CUDA renderer is not available")
        
        # Use autograd-enabled forward pass
        return self.renderer(
            gaussian_model=gaussian_model,
            camera_pos=camera_pos,
            theta_range=theta_range,
            phi_range=phi_range,
            r_range=r_range,
            num_theta=num_theta,
            num_phi=num_phi,
            num_r=num_r,
            c=c,
            deltaT=deltaT,
            scaling_modifier=scaling_modifier,
            use_occlusion=use_occlusion,
            rendering_type=rendering_type
        )
    
    def render_from_spherical_samples(
        self,
        gaussian_model,
        input_points: torch.Tensor,
        camera_pos: torch.Tensor,
        I1: int,
        I2: int,
        num_r: int,
        num_angular: int,
        dtheta: float,
        dphi: float,
        c: float,
        deltaT: float,
        scaling_modifier: float = 1.0,
        use_occlusion: bool = True,
        rendering_type: str = 'netf'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render using pre-computed spherical sample points.
        This provides compatibility with the original spherical_sample_histogram function.
        
        Args:
            gaussian_model: GaussianModel instance
            input_points: [Na, 5] tensor (x, y, z, theta, phi)
            camera_pos: [3] camera position
            I1, I2: Start and end time indices
            num_r: Number of radial samples
            num_angular: Number of angular samples (sqrt of total angular samples)
            dtheta, dphi: Angular increments
            c, deltaT: Speed of light and time interval
            scaling_modifier: Scale modifier
            use_occlusion: Whether to use occlusion
            rendering_type: 'netf' or 'nlos-neus'
        
        Returns:
            result: [num_r, num_angular^2] rendering result
            pred_histogram: [num_r] predicted histogram
        """
        # Extract angular ranges from input_points
        theta_vals = input_points[:, 3]
        phi_vals = input_points[:, 4]
        
        theta_min = theta_vals.min().item()
        theta_max = theta_vals.max().item()
        phi_min = phi_vals.min().item()
        phi_max = phi_vals.max().item()
        
        r_min = I1 * c * deltaT
        r_max = I2 * c * deltaT
        
        # Call CUDA renderer
        result_3d, pred_histogram = self.render_transient(
            gaussian_model=gaussian_model,
            camera_pos=camera_pos,
            theta_range=(theta_min, theta_max),
            phi_range=(phi_min, phi_max),
            r_range=(r_min, r_max),
            num_theta=num_angular,
            num_phi=num_angular,
            num_r=num_r,
            c=c,
            deltaT=deltaT,
            scaling_modifier=scaling_modifier,
            use_occlusion=use_occlusion,
            rendering_type=rendering_type
        )
        
        # Reshape to match original format [num_r, num_angular^2]
        result = result_3d.reshape(num_r, num_angular * num_angular)
        
        return result, pred_histogram


def create_cuda_renderer(sigma_threshold=3.0) -> Optional[GaussianRendererCUDA]:
    """
    Factory function to create a CUDA renderer.
    
    Args:
        sigma_threshold: AABB size threshold (default: 3.0)
    
    Returns:
        GaussianRendererCUDA instance if CUDA is available, None otherwise
    """
    if not CUDA_AVAILABLE:
        print("Warning: CUDA renderer not available")
        return None
    
    return GaussianRendererCUDA(sigma_threshold=sigma_threshold)



"""
CUDA-accelerated rendering for NLOS Gaussian Model
Integrates the CUDA renderer with the GaussianModel
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from cuda_renderer import NLOSGaussianRenderer, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: cuda_renderer not available. Falling back to standard rendering.")


class GaussianRendererCUDA:
    """
    CUDA-accelerated renderer that integrates with GaussianModel.
    This is a wrapper that provides the same interface as the original
    rendering functions but uses efficient ray-based CUDA kernels.
    """
    
    def __init__(self, sigma_threshold=3.0):
        """
        Args:
            sigma_threshold: Number of standard deviations for Gaussian AABB (default: 3.0)
        """
        self.use_cuda = CUDA_AVAILABLE
        if self.use_cuda:
            self.renderer = NLOSGaussianRenderer(sigma_threshold=sigma_threshold)
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
        
        return self.renderer.render(
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



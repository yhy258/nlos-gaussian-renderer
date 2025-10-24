"""
TODO: Since I separated the gaussian_renderer from the main code (in submodules), this file is not used anymore.
      We have to change the file structure. like rendering_cuda.py
Section-based Analytic Rendering Integration for GaussianModel
Wrapper that integrates SectionGaussianRendererCUDA with existing code
"""

import torch
from typing import Tuple, Optional

try:
    from cuda_renderer import SectionGaussianRendererCUDA, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: cuda_renderer not available. Section rendering disabled.")


class GaussianSectionRenderer:
    """
    Wrapper class for section-based analytic rendering.
    
    This provides the same interface as GaussianRendererCUDA but uses
    analytic integration instead of numerical sampling.
    
    Usage:
        renderer = GaussianSectionRenderer()
        result, histogram = renderer.render_transient(
            gaussian_model=model,
            camera_pos=camera_position,
            ...
        )
    """
    
    def __init__(self, sigma_threshold=3.0):
        """
        Args:
            sigma_threshold: Number of standard deviations for Gaussian AABB (default: 3.0)
        """
        self.use_cuda = CUDA_AVAILABLE
        if self.use_cuda:
            self.renderer = SectionGaussianRendererCUDA(sigma_threshold=sigma_threshold)
        else:
            self.renderer = None
            print("CUDA renderer not available, section rendering disabled")
    
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
        Render transient histogram using section-based analytic integration.
        
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
            raise RuntimeError("Section renderer is not available")
        
        return self.renderer.render_transient(
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
        Provides compatibility with the original spherical_sample_histogram function.
        
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
        if not self.use_cuda or self.renderer is None:
            raise RuntimeError("Section renderer is not available")
        
        return self.renderer.render_from_spherical_samples(
            gaussian_model=gaussian_model,
            input_points=input_points,
            camera_pos=camera_pos,
            I1=I1,
            I2=I2,
            num_r=num_r,
            num_angular=num_angular,
            dtheta=dtheta,
            dphi=dphi,
            c=c,
            deltaT=deltaT,
            scaling_modifier=scaling_modifier,
            use_occlusion=use_occlusion,
            rendering_type=rendering_type
        )


def create_section_renderer(sigma_threshold=3.0) -> Optional[GaussianSectionRenderer]:
    """
    Factory function to create a section-based renderer.
    
    Args:
        sigma_threshold: AABB size threshold (default: 3.0)
    
    Returns:
        GaussianSectionRenderer instance if CUDA is available, None otherwise
    """
    if not CUDA_AVAILABLE:
        print("Warning: Section renderer not available")
        return None
    
    return GaussianSectionRenderer(sigma_threshold=sigma_threshold)



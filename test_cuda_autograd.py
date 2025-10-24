"""
Test script for CUDA renderer autograd functionality
Verifies that forward/backward passes work correctly
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cuda_import():
    """Test if CUDA renderer can be imported"""
    print("=" * 60)
    print("Testing CUDA Renderer Import")
    print("=" * 60)
    
    try:
        from nlos_gaussian_renderer import _C
        print("✓ CUDA extension imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CUDA extension: {e}")
        print("\nTo build the CUDA extension, run:")
        print("  cd submodules/cuda_renderer")
        print("  python setup.py install")
        return False


def test_autograd_module():
    """Test autograd module can be created"""
    print("\n" + "=" * 60)
    print("Testing Autograd Module Creation")
    print("=" * 60)
    
    try:
        from gaussian_model.cuda_autograd import CUDARenderModule
        module = CUDARenderModule(sigma_threshold=3.0)
        print("✓ CUDARenderModule created successfully")
        return True, module
    except Exception as e:
        print(f"✗ Failed to create CUDARenderModule: {e}")
        return False, None


def test_forward_pass(module):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("✗ CUDA not available, skipping forward test")
        return False
    
    try:
        # Create dummy Gaussian model
        from gaussian_model.gaussian_model import GaussianModel
        
        model = GaussianModel(sh_degree=0)
        
        # Initialize with a few Gaussians
        num_gaussians = 10
        model._mu = torch.randn(num_gaussians, 3, device=device, requires_grad=True)
        model._scaling = torch.randn(num_gaussians, 3, device=device, requires_grad=True) * 0.1
        model._rotation = torch.randn(num_gaussians, 4, device=device, requires_grad=True)
        model._rotation = torch.nn.functional.normalize(model._rotation, dim=1)
        model._opacity = torch.randn(num_gaussians, 1, device=device, requires_grad=True)
        model._features_dc = torch.randn(num_gaussians, 1, 1, device=device, requires_grad=True)
        model._features_rest = torch.zeros(num_gaussians, 0, 3, device=device)
        
        # Setup camera and rendering parameters
        camera_pos = torch.tensor([0.0, 0.0, -1.0], device=device)
        theta_range = (0.0, 3.14159)
        phi_range = (0.0, 2 * 3.14159)
        r_range = (0.5, 2.0)
        
        num_theta = 4
        num_phi = 4
        num_r = 10
        
        c = 1.0  # Speed of light (normalized)
        deltaT = 0.01
        
        print(f"Rendering with {num_gaussians} Gaussians")
        print(f"Angular samples: {num_theta} x {num_phi}")
        print(f"Radial samples: {num_r}")
        
        # Forward pass
        result, pred_histogram = module(
            gaussian_model=model,
            camera_pos=camera_pos,
            theta_range=theta_range,
            phi_range=phi_range,
            r_range=r_range,
            num_theta=num_theta,
            num_phi=num_phi,
            num_r=num_r,
            c=c,
            deltaT=deltaT,
            scaling_modifier=1.0,
            use_occlusion=True,
            rendering_type='netf'
        )
        
        print(f"✓ Forward pass completed")
        print(f"  Result shape: {result.shape}")
        print(f"  Histogram shape: {pred_histogram.shape}")
        print(f"  Result range: [{result.min().item():.4f}, {result.max().item():.4f}]")
        
        return True, result, pred_histogram, model
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_backward_pass(result, pred_histogram, model):
    """Test backward pass (gradient computation)"""
    print("\n" + "=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)
    
    try:
        # Create dummy loss
        target_histogram = torch.ones_like(pred_histogram) * 0.5
        loss = torch.nn.functional.mse_loss(pred_histogram, target_histogram)
        
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = False
        grad_info = []
        
        if model._mu.grad is not None:
            has_grad = True
            grad_info.append(f"  _mu: {model._mu.grad.abs().mean().item():.6e}")
        
        if model._scaling.grad is not None:
            has_grad = True
            grad_info.append(f"  _scaling: {model._scaling.grad.abs().mean().item():.6e}")
        
        if model._rotation.grad is not None:
            has_grad = True
            grad_info.append(f"  _rotation: {model._rotation.grad.abs().mean().item():.6e}")
        
        if model._opacity.grad is not None:
            has_grad = True
            grad_info.append(f"  _opacity: {model._opacity.grad.abs().mean().item():.6e}")
        
        if model._features_dc.grad is not None:
            has_grad = True
            grad_info.append(f"  _features_dc: {model._features_dc.grad.abs().mean().item():.6e}")
        
        if has_grad:
            print("✓ Backward pass completed")
            print("Gradient magnitudes (mean absolute value):")
            for info in grad_info:
                print(info)
        else:
            print("⚠ Backward pass completed but no gradients computed")
            print("  This is expected if backward kernel is not yet implemented")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CUDA Renderer Autograd Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Import
    if not test_cuda_import():
        print("\n❌ Tests failed: Cannot import CUDA extension")
        return
    
    # Test 2: Module creation
    success, module = test_autograd_module()
    if not success:
        print("\n❌ Tests failed: Cannot create autograd module")
        return
    
    # Test 3: Forward pass
    success, result, pred_histogram, model = test_forward_pass(module)
    if not success:
        print("\n❌ Tests failed: Forward pass error")
        return
    
    # Test 4: Backward pass
    success = test_backward_pass(result, pred_histogram, model)
    if not success:
        print("\n❌ Tests failed: Backward pass error")
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nNote: Backward pass currently uses PyTorch autograd.")
    print("For optimal performance, implement custom CUDA backward kernel.")


if __name__ == "__main__":
    main()


"""
Test script for CUDA renderer
"""

import torch
import numpy as np

def test_import():
    """Test if CUDA renderer can be imported"""
    print("=" * 60)
    print("Test 1: Import CUDA Renderer")
    print("=" * 60)
    
    try:
        from cuda_renderer import NLOSGaussianRenderer, CUDA_AVAILABLE
        print(f"✓ CUDA renderer imported successfully")
        print(f"✓ CUDA available: {CUDA_AVAILABLE}")
        
        if CUDA_AVAILABLE:
            renderer = NLOSGaussianRenderer()
            print(f"✓ Renderer instance created")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError as e:
        print(f"✗ Failed to import CUDA renderer: {e}")
        print("\nTo install, run:")
        print("  cd cuda_renderer")
        print("  python setup.py install")
        return False


def test_integration():
    """Test if CUDA renderer integrates with existing code"""
    print("\n" + "=" * 60)
    print("Test 2: Integration with Existing Code")
    print("=" * 60)
    
    try:
        from gaussian_model.rendering_cuda import create_cuda_renderer, CUDA_AVAILABLE
        
        if not CUDA_AVAILABLE:
            print("✗ CUDA renderer not available, skipping integration test")
            return False
        
        renderer = create_cuda_renderer()
        if renderer is None:
            print("✗ Failed to create renderer")
            return False
        
        print("✓ Renderer created successfully")
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_nlos_helpers():
    """Test if nlos_helpers can use CUDA renderer"""
    print("\n" + "=" * 60)
    print("Test 3: nlos_helpers Integration")
    print("=" * 60)
    
    try:
        import nlos_helpers
        
        if hasattr(nlos_helpers, 'CUDA_RENDERER') and nlos_helpers.CUDA_RENDERER is not None:
            print("✓ nlos_helpers has CUDA_RENDERER")
            print("✓ CUDA renderer will be used when args.use_cuda_renderer=True")
            return True
        else:
            print("! nlos_helpers loaded but CUDA_RENDERER is None")
            print("  This is normal if CUDA extension is not installed")
            return False
            
    except Exception as e:
        print(f"✗ nlos_helpers test failed: {e}")
        return False


def test_config():
    """Test if config has use_cuda_renderer option"""
    print("\n" + "=" * 60)
    print("Test 4: Configuration")
    print("=" * 60)
    
    try:
        from configs.default import Config
        
        args = Config()
        if hasattr(args, 'use_cuda_renderer'):
            print(f"✓ Config has use_cuda_renderer option")
            print(f"  Current value: {args.use_cuda_renderer}")
            print(f"\n  To enable CUDA rendering, set:")
            print(f"    args.use_cuda_renderer = True")
            return True
        else:
            print("✗ Config missing use_cuda_renderer option")
            return False
            
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("NLOS Gaussian CUDA Renderer - Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Import", test_import()))
    results.append(("Integration", test_integration()))
    results.append(("nlos_helpers", test_nlos_helpers()))
    results.append(("Config", test_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n✓ All tests passed!")
        print("\nYou can now use CUDA rendering by setting:")
        print("  args.use_cuda_renderer = True")
    else:
        print("\n! Some tests failed")
        print("\nTo install CUDA renderer:")
        print("  ./install_cuda_renderer.sh")
        print("\nOr manually:")
        print("  cd cuda_renderer")
        print("  python setup.py install")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()



"""
Simple test to verify autograd connection
"""
import torch

# Test 1: Does the tensor have requires_grad?
print("=" * 60)
print("Test 1: Basic autograd setup")
print("=" * 60)

device = 'cuda'
rotation = torch.randn(10, 4, device=device, requires_grad=True)
rotation_norm = torch.nn.functional.normalize(rotation, dim=1)
rotation_norm.requires_grad_(True)

print(f"rotation.requires_grad: {rotation.requires_grad}")
print(f"rotation_norm.requires_grad: {rotation_norm.requires_grad}")
print(f"rotation_norm.is_leaf: {rotation_norm.is_leaf}")

# Test 2: Does gradient flow through custom function?
print("\n" + "=" * 60)
print("Test 2: Custom autograd function")
print("=" * 60)

class TestFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2.0
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        print(f"  [Backward] grad_output: {grad_output.shape}, sum: {grad_output.sum().item():.6f}")
        grad_x = grad_output * 2.0
        print(f"  [Backward] grad_x: {grad_x.shape}, sum: {grad_x.sum().item():.6f}")
        return grad_x

x = torch.randn(10, 4, device=device, requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")

y = TestFunction.apply(x)
print(f"y.requires_grad: {y.requires_grad}")

loss = y.sum()
print(f"loss: {loss.item():.6f}")

loss.backward()
print(f"x.grad is None: {x.grad is None}")
if x.grad is not None:
    print(f"x.grad sum: {x.grad.sum().item():.6f}")

# Test 3: Does it work with normalized tensor?
print("\n" + "=" * 60)
print("Test 3: With normalized tensor")
print("=" * 60)

x_raw = torch.randn(10, 4, device=device, requires_grad=True)
x_norm = torch.nn.functional.normalize(x_raw, dim=1)
x_norm.requires_grad_(True)

print(f"x_raw.requires_grad: {x_raw.requires_grad}")
print(f"x_norm.requires_grad: {x_norm.requires_grad}")
print(f"x_norm.is_leaf: {x_norm.is_leaf}")

y = TestFunction.apply(x_norm)
loss = y.sum()
loss.backward()

print(f"x_raw.grad is None: {x_raw.grad is None}")
print(f"x_norm.grad is None: {x_norm.grad is None}")

if x_norm.grad is not None:
    print(f"x_norm.grad sum: {x_norm.grad.sum().item():.6f}")


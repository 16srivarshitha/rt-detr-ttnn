"""
Basic TTNN test - run directly without pytest
"""
import torch
import ttnn

print("=" * 80)
print("Testing TTNN Basic Operations")
print("=" * 80)

# Test 1: Import
print("\n1. Testing TTNN import...")
print(f"   ✓ TTNN imported successfully")

# Test 2: Device open
print("\n2. Testing device open...")
device = ttnn.open_device(device_id=0)
print(f"   ✓ Device opened: {device}")

# Test 3: Basic tensor operations
print("\n3. Testing basic tensor operations...")
torch_tensor = torch.randn(1, 32, 32, 32)
print(f"   Created torch tensor: {torch_tensor.shape}")

ttnn_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)
print(f"   ✓ Converted to TTNN tensor")

result = ttnn.to_torch(ttnn_tensor)
print(f"   ✓ Converted back to torch: {result.shape}")

assert result.shape == torch_tensor.shape
print(f"   ✓ Shape matches!")

ttnn.close_device(device)
print(f"   ✓ Device closed")

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)

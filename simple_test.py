#!/usr/bin/env python3
"""Simple test for PyTorch and CLIP"""

print("Testing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    exit(1)

print("\nTesting CLIP...")
try:
    import clip
    print("✓ CLIP imported successfully")
    
    # Test model loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP model loaded on {device}")
    
except Exception as e:
    print(f"✗ CLIP test failed: {e}")
    exit(1)

print("\n✓ All tests passed! Ready for Phase 1 implementation.") 
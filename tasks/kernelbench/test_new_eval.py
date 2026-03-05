#!/usr/bin/env python3
"""
Test script for the new KernelBench evaluation that uses official functions.
Tests with both a correct solution (original model) and an incorrect solution.
"""

import sys
import tempfile
from pathlib import Path

# Test 1: Evaluate with the original model as the solution (should pass)
print("=" * 60)
print("TEST 1: Original model as solution (should pass correctness)")
print("=" * 60)

# Create a temporary file with the original model
original_code = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(original_code)
    original_file = f.name

# Run evaluation with only 1 correctness trial and 10 perf trials for speed
import subprocess
result = subprocess.run([
    "python", "evaluate_gpu.py",
    "--task-id", "1_23",  # Softmax task
    "--solution-path", original_file,
    "--num-correct-trials", "1",  # Just 1 for testing
    "--num-perf-trials", "10",    # Just 10 for testing
    "--device", "cpu"  # Use CPU for testing on head node
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
print("Return code:", result.returncode)

# Clean up
Path(original_file).unlink()

print("\n" + "=" * 60)
print("TEST 2: Broken model (should fail correctness)")
print("=" * 60)

# Create a broken model that returns zeros
broken_code = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broken: returns zeros instead of softmax
        return torch.zeros_like(x)
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(broken_code)
    broken_file = f.name

# Run evaluation
result = subprocess.run([
    "python", "evaluate_gpu.py",
    "--task-id", "1_23",
    "--solution-path", broken_file,
    "--num-correct-trials", "1",
    "--num-perf-trials", "10",
    "--device", "cpu"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
print("Return code:", result.returncode)

# Clean up
Path(broken_file).unlink()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✓ Test 1 should show CORRECTNESS PASSED and speedup ~1.0x")
print("✓ Test 2 should show CORRECTNESS FAILED and speedup: 0.0000")
print("✓ Return codes should be 0 for pass, 1 for fail")
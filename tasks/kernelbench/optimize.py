"""
KernelBench Task: 1_10 - 3D tensor matrix multiplication
Level: 1

This file contains the original Model class that needs to be optimized.
Your task is to create a ModelNew class that:
1. Has the EXACT same interface (__init__ and forward signatures)
2. Produces outputs within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
3. Runs faster than the original on GPU

IMPORTANT: You must create a class called "ModelNew" (not "Model")
"""

# Original imports from the task
import torch
import torch.nn as nn

# Additional imports you may need for optimization
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global variables from original task


# Helper functions from original task
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return torch.matmul(A, B)

# Input/Init specifications from original task
def get_inputs():
    A = torch.rand(N, M, K)
    B = torch.rand(K, L)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

###############################################################################
# ORIGINAL MODEL (DO NOT MODIFY - FOR REFERENCE ONLY)
###############################################################################

class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return torch.matmul(A, B)

###############################################################################
# EXAMPLE OPTIMIZATION PATTERN
###############################################################################

# Here's an example of optimization from a similar task:

# === Original Example ===
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

"""

# === Optimized Example ===
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

"""

###############################################################################
# YOUR OPTIMIZED IMPLEMENTATION
###############################################################################


# Level 1 Optimization Strategies:
# - Replace PyTorch operations with custom CUDA kernels
# - Optimize memory access patterns (coalescing)
# - Use shared memory for data reuse
# - Leverage tensor cores for matrix operations
# - Minimize memory transfers between kernels


# TODO: Create your optimized ModelNew class below
# Remember:
# 1. Keep the same __init__ signature as Model
# 2. Keep the same forward signature as Model
# 3. Maintain numerical accuracy within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
# 4. Focus on GPU performance optimization

class ModelNew(nn.Module):
    """Optimized version of Model for 3D tensor matrix multiplication"""

    def __init__(self, *args, **kwargs):
        """Initialize with same signature as Model.__init__"""
        super().__init__()
        # TODO: Implement optimized initialization
        # You can copy from Model and modify, or completely reimplement
        pass

    def forward(self, *args, **kwargs):
        """Forward pass with same signature as Model.forward"""
        # TODO: Implement optimized forward pass
        # This is where your main optimizations should go
        pass

# Optional: Add custom CUDA kernels here
# Example structure:
#
# cuda_source = '''
# #include <torch/extension.h>
# #include <cuda_runtime.h>
#
# __global__ void custom_kernel(...) {
#     // Your CUDA kernel implementation
# }
#
# torch::Tensor custom_op(torch::Tensor input) {
#     // Launch kernel and return result
# }
# '''
#
# cpp_source = '''
# torch::Tensor custom_op(torch::Tensor input);
# '''
#
# custom_module = load_inline(
#     name='custom_ops',
#     cpp_sources=cpp_source,
#     cuda_sources=cuda_source,
#     functions=['custom_op'],
#     verbose=True
# )

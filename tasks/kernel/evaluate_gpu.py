import time
import sys
import os
import pathlib
import importlib
import importlib.util
import traceback
import torch
import torch.nn as nn


########################################################
# Baseline
########################################################
class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, self.weight.T)
        x = x / 2
        x = torch.sum(x, dim=1, keepdim=True)
        x = x * self.scaling_factor
        return x


########################################################
# Weco Solution
########################################################
def load_module_from_path(module_path: str, add_to_sys_modules: bool = False):
    # Clean out all old compiled extensions to prevent namespace collisions during build
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


########################################################
# GPU-Optimized Benchmark
########################################################
os.environ["MAX_JOBS"] = "1"  # number of workers for building with ninja


def get_inputs(B, N, device):
    return torch.randn(B, N, device=device, dtype=torch.float32)


@torch.no_grad()
def bench_gpu(f, inputs, n_warmup, n_rep):
    """GPU-optimized benchmarking function"""
    device_type = inputs.device.type

    # Ensure we're on GPU
    if device_type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device_type}")

    # warm up
    for _ in range(n_warmup):
        f(inputs)
    torch.cuda.synchronize()

    # Use CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(n_rep):
        start_event.record()
        f(inputs)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    # Return median time to be robust to outliers
    return torch.median(torch.tensor(timings)).item()


@torch.no_grad()
def bench_cpu(f, inputs, n_warmup, n_rep):
    """CPU benchmarking function"""
    device_type = inputs.device.type

    # warm up
    for _ in range(n_warmup):
        f(inputs)

    # benchmark
    t_avg = 0.0
    for _ in range(n_rep):
        start_time = time.time()
        f(inputs)
        t_avg += time.time() - start_time

    t_avg /= n_rep
    return t_avg * 1000  # Convert to milliseconds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-path", type=str, required=True)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    # Set GPU if cuda device is specified
    if args.device == "cuda":
        # Use the GPU assigned by Ray/CUDA_VISIBLE_DEVICES
        if not torch.cuda.is_available():
            print("CUDA not available on this system")
            exit(1)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device(args.device)

    # benchmark parameters
    n_correctness_trials = 10
    correctness_tolerance = 1e-5

    # Adjust warmup and rep counts based on device
    if args.device == "cuda":
        n_warmup = 100  # Less warmup needed for GPU
        n_rep = 1000    # More reps for accurate GPU timing
        # Use larger batch sizes for GPU to better utilize parallelism
        batch_size, input_size, hidden_size, scaling_factor = 1024, 512, 1024, 1.5
    else:
        n_warmup = 1000
        n_rep = 5000
        batch_size, input_size, hidden_size, scaling_factor = 128, 10, 20, 1.5

    # load solution module
    try:
        torch.manual_seed(0)
        solution_module = load_module_from_path(args.solution_path, add_to_sys_modules=False)
        solution_model = solution_module.Model(input_size, hidden_size, scaling_factor).to(device)
        assert isinstance(solution_model, nn.Module)
        assert hasattr(solution_model, "forward")
    except Exception:
        print(f"Candidate module initialization failed: {traceback.format_exc()}")
        exit(1)

    torch.manual_seed(0)
    baseline_model = Model(input_size, hidden_size, scaling_factor).to(device)

    # measure correctness
    max_diff_avg = 0
    for _ in range(n_correctness_trials):
        inputs = get_inputs(batch_size, input_size, device)
        optimized_output = solution_model(inputs)
        if torch.isnan(optimized_output).any():
            print("Incorrect solution: NaN detected in optimized model output")
        if torch.isinf(optimized_output).any():
            print("Incorrect solution: Inf detected in optimized model output")
        baseline_output = baseline_model(inputs)
        max_diff_avg += torch.max(torch.abs(optimized_output - baseline_output))
    max_diff_avg /= n_correctness_trials
    print(f"max float diff between values of baseline and optimized model: {max_diff_avg}")
    if max_diff_avg > correctness_tolerance:
        print("Incorrect solution: max float diff is too high")

    # measure performance
    inputs = get_inputs(batch_size, input_size, device)

    # Choose appropriate benchmark function
    bench_fn = bench_gpu if args.device == "cuda" else bench_cpu

    t_avg_baseline = bench_fn(baseline_model, inputs, n_warmup, n_rep)
    print(f"baseline time: {t_avg_baseline:.2f}ms")
    t_avg_optimized = bench_fn(solution_model, inputs, n_warmup, n_rep)
    print(f"optimized time: {t_avg_optimized:.2f}ms")
    print(f"speedup: {t_avg_baseline / t_avg_optimized:.2f}x")
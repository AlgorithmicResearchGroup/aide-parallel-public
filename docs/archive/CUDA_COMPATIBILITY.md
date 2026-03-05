# CUDA and PyTorch Compatibility Guide for H100 GPUs

## H100 GPU Requirements

The NVIDIA H100 (Hopper architecture) requires:
- **Minimum CUDA Version**: 11.8
- **Recommended CUDA Version**: 12.1 or newer
- **Compute Capability**: 9.0

## PyTorch Version Recommendations

### For Best H100 Performance (Recommended)

**PyTorch 2.1.0+ with CUDA 12.1**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Benefits:
- Native H100 support with optimized kernels
- Support for new Tensor Core operations
- Better memory management for 80GB HBM3
- Flash Attention v2 compatibility
- Improved FP8 support

### Alternative Options

**PyTorch 2.1.0+ with CUDA 11.8** (if you have older CUDA drivers)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**PyTorch 2.2.0+ with CUDA 12.1** (latest features)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Checking Your Setup

### 1. Check CUDA Version
```bash
# Check CUDA compiler version
nvcc --version

# Check CUDA runtime version (driver)
nvidia-smi
```

### 2. Check PyTorch CUDA Support
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

# Check H100 detection
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
```

## H100-Specific Optimizations

### 1. Enable TF32 for Better Performance
```python
# Already enabled in evaluate.py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Automatic Mixed Precision (AMP)
```python
# Already used in evaluate.py
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

### 3. Flash Attention (Optional)
For transformer models, Flash Attention can provide significant speedups:
```bash
pip install flash-attn --no-build-isolation
```

### 4. Set CUDA Memory Allocation
```python
# Reduce memory fragmentation on H100
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## Troubleshooting

### Issue: PyTorch doesn't detect GPUs
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: CUDA version mismatch
```bash
# Check all CUDA installations
ls -la /usr/local/cuda*
which nvcc
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### Issue: Out of memory on 80GB H100
```python
# Clear cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Performance Expectations

With proper configuration on H100:
- **Memory Bandwidth**: 3.35 TB/s (vs 1.5 TB/s on A100)
- **FP16 Tensor Core**: 1,979 TFLOPS (3x faster than A100)
- **FP8 Tensor Core**: 3,958 TFLOPS (new capability)
- **Training Speedup**: 2-3x faster than A100 for large models

## Quick Installation

Run the provided script for automatic setup:
```bash
./install_requirements.sh
```

This will:
1. Detect your CUDA version
2. Install appropriate PyTorch version
3. Verify GPU detection
4. Install all dependencies
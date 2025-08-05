import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

# This will tell you the CUDA version PyTorch was built with.
# If it's a CPU-only build, this will be None.
print(f"PyTorch CUDA version: {torch.version.cuda}")
#!/bin/bash
set -ex

# System dependencies for Amazon Linux 2023 (EMR 7.x)
sudo dnf install -y htop

# Python packages via pip3
# The system-managed pip should not be upgraded via pip itself.

# PyTorch with CUDA 11.8 support (compatible with EMR GPU instances)
sudo python3 -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
sudo python3 -m pip install \
    Pillow==10.2.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    pyarrow==14.0.1

# Verify GPU availability
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Bootstrap completed successfully"

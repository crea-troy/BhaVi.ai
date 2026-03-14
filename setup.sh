#!/bin/bash
# BhaVi Setup Script
# Run this first on your machine: bash setup.sh

echo "========================================="
echo "  BhaVi Architecture - Setup Script"
echo "========================================="

# Install PyTorch CPU (optimized for AMD Ryzen)
echo "[1/4] Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Install supporting libraries
echo "[2/4] Installing dependencies..."
pip3 install numpy matplotlib tqdm tensorboard --break-system-packages

# Install memory efficient tools
echo "[3/4] Installing optimization tools..."
pip3 install psutil --break-system-packages

echo "[4/4] Verifying installation..."
python3 -c "
import torch
import numpy as np
print('✅ PyTorch:', torch.__version__)
print('✅ NumPy:', np.__version__)
print('✅ CPU cores available:', torch.get_num_threads())
print('✅ Ready to build BhaVi')
"

echo ""
echo "========================================="
echo "  Setup Complete! Run: python3 main.py"
echo "========================================="

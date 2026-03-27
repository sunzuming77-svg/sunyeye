#!/usr/bin/env bash
# ============================================================
# BAT-mamba: Linux one-click environment installer
# Tested on: Ubuntu 20.04/22.04, CUDA 11.8, Conda >= 23.x
# Usage:
#   chmod +x install_linux.sh
#   ./install_linux.sh
# ============================================================
set -euo pipefail

ENV_NAME="bat-mamba-linux"
PYTHON_VER="3.10"
CUDA_VER="11.8"

echo "============================================"
echo " BAT-mamba Linux Installer"
echo " ENV : $ENV_NAME"
echo " Python: $PYTHON_VER  CUDA: $CUDA_VER"
echo "============================================"

# ---- 0. Verify conda is available ----
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# ---- 1. Verify CUDA toolkit is present ----
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. mamba-ssm CUDA extensions may fail to compile."
    echo "  Install CUDA 11.8 toolkit: https://developer.nvidia.com/cuda-11-8-0-download-archive"
fi

# ---- 2. Remove existing env (optional, comment out to skip) ----
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Existing env '$ENV_NAME' found. Removing..."
    conda env remove -n "$ENV_NAME" -y
fi

# ---- 3. Create conda env from yml ----
echo ""
echo "[1/4] Creating conda environment from environment-linux.yml ..."
conda env create -f environment-linux.yml
echo "Done."

# ---- 4. Activate env for subsequent pip installs ----
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- 5. Verify mamba-ssm CUDA extensions compiled correctly ----
echo ""
echo "[2/4] Verifying mamba-ssm installation ..."
python -c "
import mamba_ssm
print('mamba-ssm version:', mamba_ssm.__version__)
from mamba_ssm.modules.mamba_simple import Mamba
import torch
if torch.cuda.is_available():
    m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(2, 32, 64).cuda()
    y = m(x)
    print('mamba-ssm forward pass OK, output shape:', y.shape)
else:
    print('WARNING: CUDA not available, skipping forward pass test.')
print('mamba-ssm check PASSED.')
"

# ---- 6. Verify fairseq ----
echo ""
echo "[3/4] Verifying fairseq installation ..."
python -c "import fairseq; print('fairseq OK:', fairseq.__version__)"

# ---- 7. Quick model smoke-test ----
echo ""
echo "[4/4] Running project quick_test.py (if present) ..."
if [ -f quick_test.py ]; then
    python quick_test.py && echo "quick_test PASSED." || echo "WARNING: quick_test.py failed — check model/data paths."
else
    echo "quick_test.py not found, skipping."
fi

echo ""
echo "============================================"
echo " Installation complete!"
echo " Activate with:  conda activate $ENV_NAME"
echo " Start training: python main.py --database_path /path/to/PS_data --protocols_path /path/to/PS_data"
echo "============================================"

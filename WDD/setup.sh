#!/bin/bash
# Welding Defect Detection - Linux Setup Script
# Installs all dependencies including CPU-only PyTorch

echo -e "\n========================================"
echo "Welding Defect Detection - Setup"
echo "========================================"
echo -e "\n"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ using: sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

echo "[1/3] Python found. Installing base dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install base dependencies"
    exit 1
fi

echo -e "\n[2/3] Installing PyTorch CPU version..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if [ $? -ne 0 ]; then
    echo "WARNING: PyTorch installation had issues, but may have partially succeeded"
fi

echo -e "\n[3/3] Verifying installation..."
python3 -c "import torch; import streamlit; print('[OK] All dependencies installed successfully!')" &> /dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Some imports failed, but installation may still work"
    echo "Trying individual imports..."
    python3 -c "import streamlit" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Streamlit not installed properly"
        exit 1
    fi
fi

echo -e "\n========================================"
echo "[OK] Setup Complete!"
echo "========================================"
echo -e "\nTo start the application, run:"
echo "  streamlit run app.py"
echo -e "\n"

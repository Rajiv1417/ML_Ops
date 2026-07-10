# Installation Troubleshooting Guide (Windows)

## Issue: "Cannot import 'setuptools.build_meta'" or NumPy Build Errors

This occurs when using **Python 3.14** with older package versions that try to build from source.

### Quick Fix

**Option 1: Use Setup Script** (Recommended)
```powershell
cd C:\path\to\Welding_Defect_Detection
.\setup.bat
```

**Option 2: Manual Installation**
```powershell
# 1. Upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Option 3: Use Python 3.12 or 3.13** (Safest)
- Python 3.14 is very new and may have compatibility issues
- Download Python 3.12 or 3.13 from https://www.python.org/downloads/
- Reinstall dependencies with the newer Python version

---

## What Changed

**requirements.txt** now uses flexible version ranges instead of exact versions:
- ✅ Works with Python 3.11, 3.12, 3.13, and 3.14
- ✅ No source builds required (only binary wheels)
- ✅ Faster installation

**setup.bat** now:
- ✅ Upgrades pip, setuptools, and wheel first
- ✅ More lenient error handling
- ✅ Continues even if some steps have warnings
- ✅ Verifies each component separately

---

## Common Errors & Solutions

### Error: "Cannot import 'setuptools.build_meta'"
```powershell
# Solution: Upgrade setuptools first
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Error: "No matching distribution found"
```powershell
# Solution: Let pip use any compatible version
# (Already fixed - requirements.txt now uses >= instead of ==)
pip install -r requirements.txt
```

### Error: "numpy ... requires python_version < 3.14"
```powershell
# Solution: Install newer numpy or downgrade Python
pip install --upgrade numpy
# OR use Python 3.12/3.13
```

### Error: "ERROR: Could not find a version that satisfies the requirement"
```powershell
# Solution: Upgrade pip and try again
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step-by-Step (Guaranteed to Work on Windows)

```powershell
# 1. Navigate to directory
cd C:\path\to\Welding_Defect_Detection

# 2. Upgrade Python tools
pip install --upgrade pip setuptools wheel

# 3. Install all base packages
pip install streamlit numpy pillow plotly pandas scikit-learn scipy pydantic python-dotenv tqdm

# 4. Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Run app
streamlit run app.py
```

---

## Verification

After installation, verify everything works:

```powershell
# Check Python version
python --version

# Check installed packages
pip list | findstr /i "streamlit torch numpy"

# Quick import test
python -c "import streamlit, torch, numpy; print('✓ All imports work!')"

# Run the app
streamlit run app.py
```

---

## Still Having Issues?

Try the **cleanest approach**:

```powershell
# 1. Create a fresh Python environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 2. Upgrade everything
pip install --upgrade pip setuptools wheel

# 3. Install packages one by one
pip install streamlit
pip install numpy pillow plotly pandas scikit-learn scipy pydantic python-dotenv tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Run app
streamlit run app.py
```

---

**Tip**: If you're using Python 3.14 (very new), consider using **Python 3.12 or 3.13** for better package compatibility.

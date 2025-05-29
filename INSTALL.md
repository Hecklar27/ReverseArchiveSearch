# Installation Guide - Reverse Archive Search

## ✅ Tested & Working Configuration

**Hardware**: NVIDIA GeForce RTX 2080 (7GB VRAM)  
**CUDA**: 11.8  
**PyTorch**: 2.0.1+cu118  
**Status**: Fully functional with GPU acceleration  

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Clone/download the project
cd ReverseArchiveSearch

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Run automated setup
python setup.py
# Choose option 2 for CUDA support
# Choose option 1 for CPU-only
```

### Option 2: Manual Installation

#### For CUDA Support (GPU Acceleration)
```bash
# Install CUDA-enabled PyTorch
pip install -r requirements_cuda.txt --index-url https://download.pytorch.org/whl/cu118

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Run the application
python main.py
```

#### For CPU-Only
```bash
# Install CPU-only PyTorch
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Run the application
python main.py
```

## 📋 Exact Package Versions

### Core Dependencies
- `torch==2.0.1+cu118` (CUDA) or `torch==2.0.1` (CPU)
- `torchvision==0.15.2+cu118` (CUDA) or `torchvision==0.15.2` (CPU)
- `numpy==1.26.4`
- `clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1`
- `Pillow==11.2.1`
- `requests==2.32.3`
- `colorlog==6.9.0`

### Supporting Libraries
- `certifi==2025.4.26`
- `charset-normalizer==3.4.2`
- `colorama==0.4.6`
- `filelock==3.18.0`
- `ftfy==6.3.1`
- `idna==3.10`
- `Jinja2==3.1.6`
- `MarkupSafe==3.0.2`
- `mpmath==1.3.0`
- `networkx==3.4.2`
- `packaging==25.0`
- `regex==2024.11.6`
- `sympy==1.14.0`
- `tqdm==4.67.1`
- `typing_extensions==4.13.2`
- `urllib3==2.4.0`
- `wcwidth==0.2.13`

## 💻 System Requirements

### For CUDA Support
- **GPU**: NVIDIA GTX 10 series or newer (RTX 2080+ recommended)
- **VRAM**: 4GB minimum, 8GB+ recommended
- **CUDA**: 11.8 compatible drivers
- **OS**: Windows 10+, Linux (Ubuntu 18.04+)

### For CPU-Only
- **CPU**: Any modern multi-core processor
- **RAM**: 8GB minimum, 16GB+ recommended
- **OS**: Windows 10+, Linux, macOS

## ⚡ Performance Comparison

| Configuration | Search Time (3,500 images) | Hardware |
|---------------|---------------------------|----------|
| RTX 2080 (CUDA) | ~2-3 minutes | GPU acceleration |
| CPU-only | ~8-10 minutes | CPU processing |

## 🔧 Troubleshooting

### CUDA Issues
1. **CUDA not detected**: Ensure NVIDIA drivers are installed
2. **Memory errors**: Lower batch size or use CPU fallback
3. **Version conflicts**: Use exact versions from requirements

### Installation Issues
1. **Package conflicts**: Use fresh virtual environment
2. **Network errors**: Try different PyTorch index URL
3. **Permission errors**: Run with appropriate permissions

## ✅ Verification

After installation, verify CUDA is working:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Expected output with CUDA:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 2080
```

## 🚀 Usage

1. Launch the application: `python main.py`
2. Select your query image (PNG/JPEG)
3. Load Discord JSON export file
4. Click "Search" and wait for results
5. Double-click results to open Discord links

The application will automatically use GPU acceleration if available! 
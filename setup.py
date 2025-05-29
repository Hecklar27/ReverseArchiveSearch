#!/usr/bin/env python3
"""
Setup script for Reverse Archive Search.
Automates the installation of dependencies including CLIP and CUDA support.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def detect_cuda_preference():
    """Detect if user likely wants CUDA support"""
    has_nvidia = check_nvidia_gpu()
    
    print(f"\n🔍 GPU Detection:")
    if has_nvidia:
        print("✓ NVIDIA GPU detected")
        return True
    else:
        print("❌ No NVIDIA GPU detected (or drivers not installed)")
        return False

def install_pytorch(use_cuda=False):
    """Install PyTorch with or without CUDA"""
    if use_cuda:
        print("\n🚀 Installing PyTorch with CUDA support...")
        # Create a temporary requirements file for CUDA with exact working versions
        cuda_requirements = """torch==2.0.1+cu118
torchvision==0.15.2+cu118
numpy==1.26.4
Pillow==11.2.1
requests==2.32.3
colorlog==6.9.0
certifi==2025.4.26
charset-normalizer==3.4.2
colorama==0.4.6
filelock==3.18.0
ftfy==6.3.1
idna==3.10
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
packaging==25.0
regex==2024.11.6
sympy==1.14.0
tqdm==4.67.1
typing_extensions==4.13.2
urllib3==2.4.0
wcwidth==0.2.13"""
        
        with open("requirements_cuda_temp.txt", "w") as f:
            f.write(cuda_requirements)
        
        success = run_command(
            "pip install -r requirements_cuda_temp.txt --index-url https://download.pytorch.org/whl/cu118",
            "Installing CUDA-enabled PyTorch"
        )
        
        # Clean up temporary file
        try:
            os.remove("requirements_cuda_temp.txt")
        except:
            pass
            
        return success
    else:
        print("\n💻 Installing CPU-only PyTorch...")
        # Create CPU requirements with exact working versions
        cpu_requirements = """torch==2.0.1
torchvision==0.15.2
numpy==1.26.4
Pillow==11.2.1
requests==2.32.3
colorlog==6.9.0
certifi==2025.4.26
charset-normalizer==3.4.2
colorama==0.4.6
filelock==3.18.0
ftfy==6.3.1
idna==3.10
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
packaging==25.0
regex==2024.11.6
sympy==1.14.0
tqdm==4.67.1
typing_extensions==4.13.2
urllib3==2.4.0
wcwidth==0.2.13"""
        
        with open("requirements_cpu_temp.txt", "w") as f:
            f.write(cpu_requirements)
        
        success = run_command("pip install -r requirements_cpu_temp.txt", "Installing CPU-only PyTorch")
        
        # Clean up temporary file
        try:
            os.remove("requirements_cpu_temp.txt")
        except:
            pass
            
        return success

def verify_cuda_installation():
    """Verify if CUDA is working with PyTorch"""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import torch; print('CUDA available:', torch.cuda.is_available()); "
            "print('Device count:', torch.cuda.device_count()); "
            "print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'CPU')"
        ], capture_output=True, text=True, check=True)
        
        print("\n🔧 CUDA Verification:")
        print(result.stdout)
        
        # Check if CUDA is actually available
        cuda_check = subprocess.run([
            sys.executable, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"
        ], capture_output=True)
        
        return cuda_check.returncode == 0
        
    except Exception as e:
        print(f"Error verifying CUDA: {e}")
        return False

def main():
    """Main setup function"""
    print("Reverse Archive Search - Dependency Setup")
    print("=" * 50)
    print("✅ Tested Configuration:")
    print("   Hardware: NVIDIA GeForce RTX 2080 (7GB VRAM)")
    print("   CUDA: 11.8")
    print("   PyTorch: 2.0.1+cu118")
    print("   Status: Fully functional with GPU acceleration")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled. Please activate a virtual environment first.")
            return False
    
    # Detect CUDA preference
    has_nvidia = detect_cuda_preference()
    
    # Ask user for preference
    print(f"\n📦 Installation Options:")
    print("1. CPU-only (works on all systems, slower)")
    print("2. CUDA + GPU (faster, requires NVIDIA GPU)")
    
    if has_nvidia:
        default_choice = "2"
        print(f"\n💡 Recommendation: Option 2 (CUDA) - GPU detected")
    else:
        default_choice = "1"
        print(f"\n💡 Recommendation: Option 1 (CPU) - No compatible GPU detected")
    
    while True:
        choice = input(f"\nEnter choice (1-2) [default: {default_choice}]: ").strip()
        if not choice:
            choice = default_choice
        
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")
    
    use_cuda = choice == "2"
    
    # Install PyTorch
    if not install_pytorch(use_cuda):
        print("Failed to install PyTorch")
        return False
    
    # Install CLIP separately
    if not run_command("pip install git+https://github.com/openai/CLIP.git", "Installing CLIP from GitHub"):
        print("Failed to install CLIP")
        return False
    
    # Verify CUDA if requested
    if use_cuda:
        cuda_working = verify_cuda_installation()
        if not cuda_working:
            print("⚠️  Warning: CUDA installation completed but CUDA is not available.")
            print("This might be due to:")
            print("- Missing or incompatible NVIDIA drivers")
            print("- Missing CUDA toolkit")
            print("- Incompatible GPU")
            print("\nThe application will fall back to CPU processing.")
    
    # Verify installations
    print("\n=== Verifying Installation ===")
    
    packages_to_check = [
        "torch", "torchvision", "numpy", "clip", 
        "PIL", "requests", "colorlog"
    ]
    
    failed_imports = []
    for package in packages_to_check:
        try:
            if package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Some packages failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All dependencies installed successfully!")
    
    if use_cuda:
        print("\n🚀 CUDA-enabled installation complete!")
        print("The application will automatically use GPU acceleration when available.")
    else:
        print("\n💻 CPU-only installation complete!")
        print("For GPU acceleration, re-run setup and choose CUDA option.")
    
    print("\nYou can now run the application with:")
    print("python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
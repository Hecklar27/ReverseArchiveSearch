#!/usr/bin/env python3
"""
Setup script for Reverse Archive Search.
Automates the installation of dependencies including CLIP and CUDA support.
Usage:
    python setup.py          # Normal installation
    python setup.py cleanup  # Remove all dependencies for testing
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
        
        # First install PyTorch packages from CUDA index
        pytorch_requirements = """torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0"""
        
        with open("requirements_pytorch_temp.txt", "w") as f:
            f.write(pytorch_requirements)
        
        success = run_command(
            "pip install -r requirements_pytorch_temp.txt --index-url https://download.pytorch.org/whl/cu118",
            "Installing PyTorch with CUDA support"
        )
        
        # Clean up temporary file
        try:
            os.remove("requirements_pytorch_temp.txt")
        except:
            pass
        
        if not success:
            return False
        
        # Then install other dependencies from standard PyPI
        other_requirements = """numpy
Pillow
requests
colorlog
beautifulsoup4==4.12.3"""
        
        with open("requirements_other_temp.txt", "w") as f:
            f.write(other_requirements)
        
        success = run_command(
            "pip install -r requirements_other_temp.txt",
            "Installing other dependencies"
        )
        
        # Clean up temporary file
        try:
            os.remove("requirements_other_temp.txt")
        except:
            pass
            
        return success
    else:
        print("\n💻 Installing CPU-only PyTorch...")
        
        # Install all packages from standard PyPI for CPU
        cpu_requirements = """torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
numpy
Pillow
requests
colorlog
beautifulsoup4==4.12.3"""
        
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

def cleanup_dependencies():
    """Remove all installed dependencies for testing"""
    print("Reverse Archive Search - Dependency Cleanup")
    print("=" * 50)
    print("🧹 This will remove all installed dependencies")
    print("   Use this for testing the setup process")
    print("=" * 50)
    
    # Confirm cleanup
    response = input("\nAre you sure you want to remove all dependencies? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return False
    
    # List of packages to uninstall
    packages_to_remove = [
        "torch",
        "torchvision", 
        "torchaudio",
        "clip-by-openai",  # CLIP package name
        "numpy",
        "Pillow",
        "requests",
        "colorlog",
        "fsspec",
        "sympy",
        "networkx",
        "jinja2",
        "filelock",
        "typing-extensions",
        "ftfy",
        "regex",
        "tqdm",
        "packaging",
        "beautifulsoup4"
    ]
    
    print(f"\n🗑️  Removing packages: {', '.join(packages_to_remove)}")
    
    # Create uninstall command
    uninstall_cmd = f"pip uninstall -y {' '.join(packages_to_remove)}"
    
    success = run_command(uninstall_cmd, "Uninstalling packages")
    
    # Clean up any temporary files
    temp_files = [
        "requirements_pytorch_temp.txt",
        "requirements_other_temp.txt",
        "requirements_cpu_temp.txt"
    ]
    
    print(f"\n🧹 Cleaning temporary files...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"✓ Removed {temp_file}")
        except Exception as e:
            print(f"⚠️  Could not remove {temp_file}: {e}")
    
    # Clear cache directory if it exists
    cache_dir = Path("cache")
    if cache_dir.exists():
        print(f"\n🗂️  Cache directory found: {cache_dir}")
        response = input("Remove cache directory? (y/N): ")
        if response.lower() == 'y':
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print("✓ Cache directory removed")
            except Exception as e:
                print(f"⚠️  Could not remove cache directory: {e}")
        else:
            print("Cache directory preserved")
    
    if success:
        print("\n✅ Cleanup completed successfully!")
        print("You can now run 'python setup.py' to reinstall dependencies")
    else:
        print("\n⚠️  Cleanup completed with some warnings")
        print("Some packages may not have been installed or already removed")
    
    return True

def main():
    """Main setup function"""
    print("Reverse Archive Search - Dependency Setup")
    print("=" * 50)
    print("✅ Tested Configuration:")
    print("   Hardware: NVIDIA GeForce RTX 2080 (7GB VRAM)")
    print("   CUDA: 11.8")
    print("   PyTorch: 2.6.0 (with CUDA 11.8 support)")
    print("   Status: Fully functional with GPU acceleration")
    print("   Note: Updated from 2.0.1 for current availability")
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
        "PIL", "requests", "colorlog", "beautifulsoup4"
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
    if len(sys.argv) > 1:
        if sys.argv[1] == "cleanup":
            success = cleanup_dependencies()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage:")
            print("  python setup.py          # Normal installation") 
            print("  python setup.py cleanup  # Remove all dependencies")
            sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1) 
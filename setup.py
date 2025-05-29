#!/usr/bin/env python3
"""
Setup script for Reverse Archive Search.
Automates the installation of dependencies including CLIP.
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

def main():
    """Main setup function"""
    print("Reverse Archive Search - Dependency Setup")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled. Please activate a virtual environment first.")
            return False
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        print("Failed to install basic requirements")
        return False
    
    # Install CLIP separately
    if not run_command("pip install git+https://github.com/openai/CLIP.git", "Installing CLIP from GitHub"):
        print("Failed to install CLIP")
        return False
    
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
    print("\nYou can now run the application with:")
    print("python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
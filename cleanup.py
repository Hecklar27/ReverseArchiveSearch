#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for Reverse Archive Search
Removes all dependencies, build artifacts, and cached data completely
so the setup script can be tested from a clean state.

Usage:
    python cleanup.py               # Interactive cleanup
    python cleanup.py --force      # Force cleanup without prompts
    python cleanup.py --help       # Show help
"""

import os
import shutil
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description, ignore_errors=True):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("   ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"   ⚠️ Warning (ignored): {e}")
            return True
        else:
            print(f"   ✗ Error: {e}")
            if e.stderr:
                print(f"   STDERR: {e.stderr}")
            return False
    except Exception as e:
        if ignore_errors:
            print(f"   ⚠️ Warning (ignored): {e}")
            return True
        else:
            print(f"   ✗ Error: {e}")
            return False


def remove_directory(path, description):
    """Remove a directory if it exists"""
    path = Path(path)
    if path.exists() and path.is_dir():
        print(f"🗂️ Removing {description}: {path}")
        try:
            shutil.rmtree(path)
            print("   ✓ Removed")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    else:
        print(f"   ℹ️ {description} not found: {path}")
        return True


def remove_file(path, description):
    """Remove a file if it exists"""
    path = Path(path)
    if path.exists() and path.is_file():
        print(f"📄 Removing {description}: {path}")
        try:
            path.unlink()
            print("   ✓ Removed")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    else:
        print(f"   ℹ️ {description} not found: {path}")
        return True


def remove_pattern(pattern, description):
    """Remove files matching a pattern"""
    import glob
    files = glob.glob(pattern, recursive=True)
    if files:
        print(f"🔍 Removing {description} ({len(files)} files)")
        removed = 0
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                removed += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {file}: {e}")
        print(f"   ✓ Removed {removed}/{len(files)} items")
        return True
    else:
        print(f"   ℹ️ No {description} found")
        return True


def uninstall_packages():
    """Uninstall all project-related packages"""
    packages_to_remove = [
        # Core ML packages
        "torch",
        "torchvision", 
        "torchaudio",
        "clip-by-openai",
        
        # Core Python packages that might have been installed
        "numpy",
        "Pillow",
        "requests",
        "colorlog",
        "beautifulsoup4",
        
        # PyTorch dependencies
        "fsspec",
        "sympy",
        "networkx",
        "jinja2",
        "filelock",
        "typing-extensions",
        "mpmath",
        
        # CLIP dependencies
        "ftfy",
        "regex",
        "tqdm",
        
        # Common dependencies
        "packaging",
        "charset-normalizer",
        "idna",
        "urllib3",
        "certifi",
        "soupsieve",
        "MarkupSafe",
    ]
    
    print(f"📦 Uninstalling packages...")
    print(f"   Packages: {', '.join(packages_to_remove)}")
    
    # Create uninstall command
    uninstall_cmd = f"pip uninstall -y {' '.join(packages_to_remove)}"
    
    return run_command(uninstall_cmd, "Uninstalling Python packages", ignore_errors=True)


def cleanup_pip_cache():
    """Clear pip cache"""
    return run_command("pip cache purge", "Clearing pip cache", ignore_errors=True)


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Comprehensive cleanup for Reverse Archive Search")
    parser.add_argument("--force", action="store_true", help="Force cleanup without prompts")
    parser.add_argument("--keep-venv", action="store_true", help="Keep virtual environment")
    parser.add_argument("--keep-logs", action="store_true", help="Keep log files")
    parser.add_argument("--keep-cache", action="store_true", help="Keep cache directory")
    args = parser.parse_args()
    
    print("🧹 Reverse Archive Search - Comprehensive Cleanup")
    print("=" * 60)
    print("This will remove:")
    print("• All Python dependencies")
    print("• Virtual environment (.venv/)")
    print("• Cache directories")
    print("• Log files")
    print("• Build artifacts")
    print("• Temporary files")
    print("• Python cache files (__pycache__)")
    print("=" * 60)
    
    if not args.force:
        response = input("\nAre you sure you want to proceed with complete cleanup? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return False
    
    print("\n🚀 Starting comprehensive cleanup...")
    
    # Track results
    results = []
    
    # 1. Uninstall Python packages
    print(f"\n{'='*20} PYTHON PACKAGES {'='*20}")
    results.append(uninstall_packages())
    results.append(cleanup_pip_cache())
    
    # 2. Remove virtual environment
    if not args.keep_venv:
        print(f"\n{'='*20} VIRTUAL ENVIRONMENT {'='*20}")
        results.append(remove_directory(".venv", "virtual environment"))
        results.append(remove_directory("venv", "virtual environment (alt name)"))
        results.append(remove_directory("env", "virtual environment (alt name)"))
    
    # 3. Remove cache directories
    if not args.keep_cache:
        print(f"\n{'='*20} CACHE DIRECTORIES {'='*20}")
        results.append(remove_directory("cache", "application cache"))
        results.append(remove_directory(".cache", "hidden cache"))
        results.append(remove_directory("__pycache__", "Python cache (root)"))
        
    # 4. Remove log files
    if not args.keep_logs:
        print(f"\n{'='*20} LOG FILES {'='*20}")
        results.append(remove_file("reverse_archive_search.log", "main log file"))
        results.append(remove_directory("logs", "logs directory"))
        results.append(remove_pattern("*.log", "log files"))
    
    # 5. Remove build artifacts
    print(f"\n{'='*20} BUILD ARTIFACTS {'='*20}")
    results.append(remove_directory("build", "build directory"))
    results.append(remove_directory("dist", "distribution directory"))
    results.append(remove_directory("*.egg-info", "egg info"))
    results.append(remove_pattern("**/__pycache__", "Python cache directories"))
    results.append(remove_pattern("**/*.pyc", "compiled Python files"))
    results.append(remove_pattern("**/*.pyo", "optimized Python files"))
    results.append(remove_pattern("**/*.pyd", "Python extension files"))
    
    # 6. Remove temporary files
    print(f"\n{'='*20} TEMPORARY FILES {'='*20}")
    temp_files = [
        "requirements_pytorch_temp.txt",
        "requirements_other_temp.txt", 
        "requirements_cpu_temp.txt",
        "parsed_messages.pkl",
        "parsed_metadata.pkl",
    ]
    
    for temp_file in temp_files:
        results.append(remove_file(temp_file, f"temporary file"))
    
    results.append(remove_pattern("*.tmp", "temporary files"))
    results.append(remove_pattern("*.temp", "temporary files"))
    results.append(remove_pattern(".*~", "backup files"))
    
    # 7. Remove IDE/editor files
    print(f"\n{'='*20} IDE/EDITOR FILES {'='*20}")
    results.append(remove_directory(".vscode", "VS Code settings"))
    results.append(remove_directory(".idea", "PyCharm settings"))
    results.append(remove_file(".DS_Store", "macOS metadata"))
    results.append(remove_pattern("**/.DS_Store", "macOS metadata files"))
    results.append(remove_file("Thumbs.db", "Windows thumbnail cache"))
    
    # Summary
    print(f"\n{'='*20} CLEANUP SUMMARY {'='*20}")
    success_count = sum(1 for result in results if result)
    total_count = len(results)
    
    if success_count == total_count:
        print("✅ Cleanup completed successfully!")
        print(f"   {success_count}/{total_count} operations completed")
    else:
        print("⚠️ Cleanup completed with some warnings")
        print(f"   {success_count}/{total_count} operations completed")
    
    print("\n🔄 Next steps:")
    print("1. Create/activate a new virtual environment:")
    print("   python -m venv .venv")
    print("   .venv\\Scripts\\activate  # Windows")
    print("   source .venv/bin/activate  # Linux/Mac")
    print("2. Run the setup script:")
    print("   python setup.py")
    
    return success_count == total_count


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1) 
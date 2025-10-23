"""
ALG Build Module
Checks and installs dependencies (Qiskit, NumPy, etc.)
"""

import subprocess
import sys


def check_python_version():
    """Verify Python 3.8+"""
    if sys.version_info < (3, 8):
        print("[ALG BUILD] ERROR: Python 3.8+ required")
        return False
    print(f"[ALG BUILD] Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[ALG BUILD] ✓ {package_name} {version}")
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip"""
    print(f"[ALG BUILD] Installing {package_name}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            check=True,
            timeout=300
        )
        print(f"[ALG BUILD] ✓ {package_name} installed")
        return True
    except subprocess.CalledProcessError:
        print(f"[ALG BUILD] ERROR: Failed to install {package_name}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[ALG BUILD] ERROR: Installation timeout for {package_name}")
        return False


def exec(args=None):
    """Execute build process"""
    print("\n" + "="*70)
    print("ALG BUILD - Dependency Check & Installation")
    print("="*70 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Required packages
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("qiskit", "qiskit"),
        ("qiskit-aer", "qiskit_aer"),
    ]
    
    print("\n[ALG BUILD] Checking dependencies...")
    missing = []
    
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing.append(pkg_name)
    
    if missing:
        print(f"\n[ALG BUILD] Missing packages: {', '.join(missing)}")
        print("[ALG BUILD] Installing...")
        
        for pkg in missing:
            if not install_package(pkg):
                print(f"[ALG BUILD] ERROR: Could not install {pkg}")
                sys.exit(1)
    
    # Create output directories
    import os
    os.makedirs("/var/qallow", exist_ok=True)
    print("\n[ALG BUILD] ✓ Output directory: /var/qallow")
    
    print("\n" + "="*70)
    print("BUILD COMPLETE - All dependencies ready")
    print("="*70 + "\n")

